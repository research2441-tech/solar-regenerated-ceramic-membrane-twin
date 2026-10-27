"""Calibration of the physics-informed twin by reverse-mode differentiation
through multiple-shooting rollouts (JAX). Trains the global rate
parameters psi and, for the hybrid twin, the bounded residual network.

  loss = sum_s sum_t  w_tmp*e_tmp^2 + w_eis*e_eis^2 + w_uv*e_uv^2
         + w_cont*||z_s(T) - z_{s+1}(0)||^2 + w_reg*||theta_nn||^2
"""
import os, pickle, sys, time
import numpy as np
import jax, jax.numpy as jnp
import optax
import twin

jax.config.update("jax_enable_x64", True)
OUT = os.path.dirname(os.path.abspath(__file__))
SEG = 48


def build_segments(runs):
    U, Y, M, A0, link = [], [], [], [], []
    for r in runs:
        n = len(r["J"]) // SEG
        ncip = np.cumsum(r["fC"])
        for s in range(n):
            sl = slice(s * SEG, (s + 1) * SEG)
            u = np.stack([r["J"][sl], r["fR"][sl], r["fC"][sl], np.clip(r["I"][sl], 0, None), r["T"][sl],
                          np.clip(r["tss_f"][sl], 1, None), np.clip(r["uv_f"][sl] / 0.02, 1, None),
                          r["cond"][sl], np.clip(r["turb_f"][sl], 1, None)], -1)
            y = np.stack([r["tmp"][sl], r["eis1"][sl], r["eis2"][sl], r["uv_p"][sl], r["uv_f"][sl] / 0.02], -1)
            m = np.isfinite(y[:, :4]) & np.stack([r["J"][sl] > 0, np.ones(SEG, bool), np.ones(SEG, bool), r["J"][sl] > 0], -1)
            U.append(u); Y.append(np.nan_to_num(y)); M.append(m.astype(float))
            A0.append(0.97 ** (ncip[s * SEG] if s > 0 else 0))
            link.append(1.0 if s < n - 1 else 0.0)
    return (jnp.array(np.array(U)), jnp.array(np.array(Y)), jnp.array(np.array(M)),
            jnp.array(np.array(A0)), jnp.array(np.array(link)))


UK = ["J", "fR", "fC", "I", "T", "tss", "doc", "cond", "turb"]


def rollout(z0, psi, nn, Useg):
    def f(z, u):
        ud = {k: u[i] for i, k in enumerate(UK)}
        zn, Rbar = twin.step(z, psi, ud, nn, jnp, twin.NOM, True)
        tmp = twin.tmp_from_R(Rbar, ud["J"], ud["T"], True, jnp)
        e = (z[:4] @ jnp.array(twin.EIS_G).T)
        return zn, (tmp, zn, e, z[4])
    zT, (tmp, zs, e, th) = jax.lax.scan(f, z0, Useg)
    return zT, tmp, zs, e, th


def make_loss(U, Y, M, A0, link, use_nn, use_eis):
    S = U.shape[0]

    def loss(params):
        psi = params["psi"]; nn = params["nn"] if use_nn else None
        zi = jax.nn.softplus(params["z0"])  # S x 5 (Rc, Ro, Rh, Rs, theta-logit handled below)
        th0 = jax.nn.sigmoid(params["th0"])
        z0 = jnp.concatenate([zi[:, :4], th0[:, None], A0[:, None]], 1)
        zT, tmp, zs, e, th = jax.vmap(lambda z, u: rollout(z, psi, nn, u))(z0, U)
        # eis is computed from the state at the start of each hour; observation after the hour -> use next state
        e_next = jnp.einsum("stk,jk->stj", zs[..., :4], jnp.array(twin.EIS_G))
        uvp = twin.UV_K * Y[..., 4] * (1 - twin.R0_COD - twin.RA_COD * (1 - zs[..., 4]))
        r_tmp = (tmp - Y[..., 0]) / (0.02 * Y[..., 0] + 0.5)
        r_e1 = (e_next[..., 0] - Y[..., 1]) / (0.03 * jnp.abs(Y[..., 1]) + 0.04)
        r_e2 = (e_next[..., 1] - Y[..., 2]) / (0.03 * jnp.abs(Y[..., 2]) + 0.04)
        r_uv = (uvp - Y[..., 3]) / (0.03 * jnp.abs(Y[..., 3]) + 0.002)
        L = jnp.sum(M[..., 0] * r_tmp ** 2) + jnp.sum(M[..., 3] * r_uv ** 2)
        if use_eis:
            L = L + jnp.sum(M[..., 1] * r_e1 ** 2) + jnp.sum(M[..., 2] * r_e2 ** 2)
        cont = zT[:-1, :5] - z0[1:, :5]
        L = L + 50.0 * jnp.sum(link[:-1, None] * cont ** 2 / (0.05 + jnp.abs(z0[1:, :5])))
        if use_nn:
            L = L + 1e-1 * sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(nn))
        L = L + 10.0 * jnp.sum((psi - jnp.array(PSI0)) ** 2)
        return L / jnp.sum(M[..., 0])
    return loss


PSI0 = [twin.NOM["log_ac"], twin.NOM["log_ao"], twin.NOM["log_kph"], twin.NOM["log_as"]]


def train(runs, use_nn=True, use_eis=True, steps=2500, seed=0, verbose=True):
    U, Y, M, A0, link = build_segments(runs)
    S = U.shape[0]
    rng = np.random.default_rng(seed)
    nn0 = twin.nn_init(rng)
    params = {"psi": jnp.array(PSI0) + 0.0, "nn": {k: jnp.array(v) for k, v in nn0.items()},
              "z0": jnp.full((S, 4), -1.0), "th0": jnp.zeros(S)}
    loss = make_loss(U, Y, M, A0, link, use_nn, use_eis)
    opt = optax.adam(optax.cosine_decay_schedule(0.03, steps, 0.05))
    st = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss))

    @jax.jit
    def upd(p, s):
        l, g = vg(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l
    t0 = time.time(); hist = []
    for i in range(steps):
        params, st, l = upd(params, st)
        if i % 100 == 0 or i == steps - 1:
            hist.append((i, float(l)))
            if verbose:
                print(i, round(float(l), 4), np.round(np.array(params["psi"]), 3), round(time.time() - t0, 1), flush=True)
    out = {"psi": np.array(params["psi"]), "nn": {k: np.array(v) for k, v in params["nn"].items()} if use_nn else None,
           "hist": hist, "train_time_s": time.time() - t0, "n_seg": S}
    return out


if __name__ == "__main__":
    runs = pickle.load(open(os.path.join(OUT, "dev_logs.pkl"), "rb"))
    res = {}
    for name, nn_, eis_ in (("hybrid", True, True), ("physics", False, True), ("hybrid_noeis", True, False)):
        print("==", name)
        res[name] = train(runs, nn_, eis_, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 2500)
    pickle.dump(res, open(os.path.join(OUT, "twin_params.pkl"), "wb"))
