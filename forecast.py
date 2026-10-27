"""Experiment E1: fouling-progression and organic-breakthrough forecasting.

Test logs come from closed-loop operation under the event-triggered policy
in three climates and two loading scenarios (eval seeds), none of which is
used for training. Forecast origins every 3 h; horizons 6, 12, 24 h; the
future action sequence is known (it is what a scheduler would evaluate),
the exogenous inputs are forecast from past observations only.
"""
import os, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import jax, jax.numpy as jnp, optax
import sim, controllers, twin, mpc, devdata
from membrane_design import viscosity

OUT = os.path.dirname(os.path.abspath(__file__))
HZ = (6, 12, 24)
EVERY = 3
RM = twin.RM[True]


def r_obs(a):
    tmp, J, T = a["tmp"], a["J"], a["T"]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = tmp * 1e3 / (viscosity(T) * J / 3.6e6) / 1e12 - RM
    R[(J <= 0) | ~np.isfinite(R)] = np.nan
    # carry forward the last measured fouling resistance
    out = R.copy(); last = 0.0
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return R, out


def uv_ratio(a):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a["uv_p"] / a["uv_f"]
    r[(a["J"] <= 0)] = np.nan
    out = r.copy(); last = 0.4
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return r, out


def obs_list(a):
    keys = ["J", "I", "T", "tss_f", "uv_f", "cond", "turb_f", "tmp", "eis1", "eis2", "uv_p", "Vp", "Vf", "Qin", "cip_left"]
    L = []
    for i in range(len(a["J"])):
        o = {k: float(a[k][i]) for k in keys}
        o["mode"] = "C" if a["isC"][i] else ("R" if a["fR"][i] else ("F" if a["J"][i] > 0 else "I"))
        L.append(o)
    return L


def future_summary(a, t, k, Ifc):
    sl = slice(t + 1, t + 1 + k)
    vol = np.sum(a["J"][sl]) / 50.0
    reg = np.sum(a["fR"][sl])
    regI = np.sum(a["fR"][sl] * Ifc[:k]) / 1000.0
    cip = float(np.sum(a["fC"][sl]) > 0)
    return [vol, reg, regI, cip]


def features_now(a, t, Rf, Uf):
    lo = max(0, t - 24)
    return [Rf[t], Uf[t], np.mean(a["J"][lo:t + 1]) / 50.0, np.sum(a["fR"][lo:t + 1]),
            a["T"][t], np.log(max(a["tss_f"][t], 1)), np.log(max(a["uv_f"][t] / 0.02, 1)), a["cond"][t] / 1000.0,
            np.log(max(a["turb_f"][t], 1)), (t % 24) / 24.0]


def build_tabular(runs):
    X = {k: [] for k in HZ}; Y = {k: [] for k in HZ}; YU = {k: [] for k in HZ}
    for a in runs:
        R, Rf = r_obs(a); U, Uf = uv_ratio(a)
        n = len(a["J"])
        ghi = np.clip(a["I"], 0, None)
        for t in range(48, n - 25, EVERY):
            Ifc = np.array([ghi[t + j - 23] if t + j - 23 < n else 0 for j in range(1, 25)])  # yesterday same hour
            f0 = features_now(a, t, Rf, Uf)
            for k in HZ:
                if not np.isfinite(R[t + k]):
                    continue
                X[k].append(f0 + future_summary(a, t, k, Ifc)); Y[k].append(R[t + k] - Rf[t])
                YU[k].append(U[t + k] - Uf[t] if np.isfinite(U[t + k]) else np.nan)
    return {k: np.array(X[k]) for k in HZ}, {k: np.array(Y[k]) for k in HZ}, {k: np.array(YU[k]) for k in HZ}


# ---------------------------------------------------------------- GRU
def gru_init(key, n_in, n_h=24, n_f=4):
    ks = jax.random.split(key, 6)
    s = 0.2
    return {"Wz": jax.random.normal(ks[0], (n_in + n_h, n_h)) * s, "Wr": jax.random.normal(ks[1], (n_in + n_h, n_h)) * s,
            "Wh": jax.random.normal(ks[2], (n_in + n_h, n_h)) * s, "bz": jnp.zeros(n_h), "br": jnp.zeros(n_h), "bh": jnp.zeros(n_h),
            "W1": jax.random.normal(ks[3], (n_h + n_f, 32)) * s, "b1": jnp.zeros(32),
            "W2": jax.random.normal(ks[4], (32, 2)) * s, "b2": jnp.zeros(2)}


def gru_apply(p, seq, fut):
    def cell(h, x):
        xh = jnp.concatenate([x, h])
        z = jax.nn.sigmoid(xh @ p["Wz"] + p["bz"]); r = jax.nn.sigmoid(xh @ p["Wr"] + p["br"])
        hh = jnp.tanh(jnp.concatenate([x, r * h]) @ p["Wh"] + p["bh"])
        return (1 - z) * h + z * hh, None
    h, _ = jax.lax.scan(cell, jnp.zeros(p["bz"].shape[0]), seq)
    q = jnp.tanh(jnp.concatenate([h, fut]) @ p["W1"] + p["b1"])
    return q @ p["W2"] + p["b2"]


def seq_feats(a, Rf, Uf):
    return np.stack([Rf, Uf, a["J"] / 50.0, a["fR"], a["isC"], np.clip(a["I"], 0, None) / 500.0, (a["T"] - 20) / 10,
                     np.log(np.clip(a["tss_f"], 1, None) / 60), np.log(np.clip(a["uv_f"], 1e-3, None) / 0.02 / 40),
                     a["cond"] / 1000.0, np.log(np.clip(a["turb_f"], 1, None) / 60)], -1)


def build_seq(runs, k, W=48):
    S, F, Y = [], [], []
    for a in runs:
        R, Rf = r_obs(a); U, Uf = uv_ratio(a); sf = seq_feats(a, Rf, Uf); n = len(a["J"])
        ghi = np.clip(a["I"], 0, None)
        for t in range(W, n - 25, EVERY):
            if not np.isfinite(R[t + k]):
                continue
            Ifc = np.array([ghi[t + j - 23] for j in range(1, 25)])
            S.append(sf[t - W + 1:t + 1]); F.append(future_summary(a, t, k, Ifc))
            Y.append([R[t + k] - Rf[t], (U[t + k] - Uf[t]) if np.isfinite(U[t + k]) else 0.0])
    return np.array(S), np.array(F), np.array(Y)


def train_gru(runs, k, steps=1500, seed=0):
    S, F, Y = build_seq(runs, k)
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    Yn = (Y - mu) / sd
    p = gru_init(jax.random.PRNGKey(seed), S.shape[2])
    f = jax.vmap(gru_apply, in_axes=(None, 0, 0))
    opt = optax.adam(3e-3); st = opt.init(p)
    S, F, Yn = jnp.array(S), jnp.array(F), jnp.array(Yn)
    rng = np.random.default_rng(seed)

    @jax.jit
    def upd(p, st, idx):
        l, g = jax.value_and_grad(lambda q: jnp.mean((f(q, S[idx], F[idx]) - Yn[idx]) ** 2))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    for i in range(steps):
        idx = jnp.array(rng.integers(0, S.shape[0], 256))
        p, st, l = upd(p, st, idx)
    return p, mu, sd


# ---------------------------------------------------------------- twin forecasts
class _Hist:
    pass


def twin_forecasts(a, twp, use_eis, rng_seed=0, alpha=0.1, n_ens=32, open_loop=False, init=None):
    """Returns dict k -> arrays (t, pred_R, sd_R, lo_R, hi_R, pred_U, sd_U) and conformal bounds."""
    rng = np.random.default_rng(rng_seed)
    kf = twin.EnKF(n_ens, rng, True, twp["psi"], twp["nn"], use_eis=use_eis)
    ol = obs_list(a); n = len(ol)
    helper = mpc.TwinMPC(twp, n_ens=n_ens); helper.rng = rng
    aci = {k: (mpc.ACI(alpha / 2, q0=1.64), mpc.ACI(alpha / 2, q0=1.64)) for k in HZ}  # upper/lower for R
    aciu = {k: (mpc.ACI(alpha / 2, q0=1.64), mpc.ACI(alpha / 2, q0=1.64)) for k in HZ}
    if init is not None:
        for k in HZ:
            aci[k][0].s = list(init["R"][k]); aci[k][1].s = list(-np.array(init["R"][k]))
            aciu[k][0].s = list(init["U"][k]); aciu[k][1].s = list(-np.array(init["U"][k]))
    scores = {"R": {k: [] for k in HZ}, "U": {k: [] for k in HZ}}
    est = []
    pend = []; res = {k: [] for k in HZ}
    R, Rf = r_obs(a); U, Uf = uv_ratio(a)
    for t in range(n):
        o = ol[t]
        u = twin.u_from_obs(o, 0); kf.predict(u)
        if not open_loop:
            kf.update(o)
        est.append(kf.Z.mean(0).copy())
        helper.o_hist = ol[:t + 1]
        # resolve pending forecasts whose target is t
        for item in [p for p in pend if p[0] == t]:
            _, k, mR, sR, mU, sU, t0 = item
            if np.isfinite(R[t]):
                up, lo = aci[k]
                qu, ql = up.q(), lo.q()
                res[k].append((t0, mR, sR, mR - ql * sR, mR + qu * sR, R[t], mU, sU, U[t], aciu[k][0].q(), aciu[k][1].q()))
                up.update((R[t] - mR) / sR); lo.update((mR - R[t]) / sR); scores["R"][k].append((R[t] - mR) / sR)
                if np.isfinite(U[t]):
                    aciu[k][0].update((U[t] - mU) / sU); aciu[k][1].update((mU - U[t]) / sU); scores["U"][k].append((U[t] - mU) / sU)
        pend = [p for p in pend if p[0] > t]
        if t >= 48 and t % EVERY == 0 and t < n - 25:
            exo = helper._exo_forecast(t + 1, 24)
            Imult = np.clip(rng.lognormal(0, 0.35, n_ens), 0.2, 1.6)
            Z = kf.Z.copy(); Psi = kf.Psi
            for j in range(1, 25):
                oj = ol[t + j]
                uu = dict(J=np.full(n_ens, oj["J"]), fR=np.full(n_ens, 1.0 if oj["mode"] == "R" else 0.0),
                          fC=np.full(n_ens, 1.0 if (oj["mode"] == "C" and oj["cip_left"] == 0) else 0.0),
                          I=exo["I"][j - 1] * Imult, T=np.full(n_ens, exo["T"][j - 1]), tss=np.full(n_ens, exo["tss"][j - 1]),
                          doc=np.full(n_ens, exo["doc"][j - 1]), cond=np.full(n_ens, exo["cond"][j - 1]),
                          turb=np.full(n_ens, exo["turb"][j - 1]))
                Z, Rbar = twin.step(Z, Psi, uu, twp["nn"], np, twin.NOM, True)
                if j in HZ:
                    th = Z[:, 4]
                    Upred = 1 - twin.R0_COD - twin.RA_COD * (1 - th)
                    Upred = Upred * twin.UV_K / 0.02
                    pend.append((t + j, j, Rbar.mean(), Rbar.std() + 0.02, Upred.mean(), Upred.std() + 0.003, t))
                    res_raw = (np.percentile(Rbar, 5), np.percentile(Rbar, 95))
                    pend[-1] = pend[-1] + ()
    return {k: np.array(v) for k, v in res.items()}, scores, np.array(est)


def metrics(y, p, lo=None, hi=None, alpha=0.1):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    out = dict(rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))), n=int(m.sum()))
    if lo is not None:
        lo, hi = lo[m], hi[m]
        cov = np.mean((y >= lo) & (y <= hi)); w = np.mean(hi - lo)
        wink = np.mean((hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi))
        out.update(cov=float(cov), width=float(w), winkler=float(wink))
    return out


def test_logs():
    runs = []
    for site in ("PHX", "GSO", "CHI"):
        for seed, d0 in ((1, 30), (2, 150), (3, 270)):
            for sc in ("normal", "shock"):
                ctrl = controllers.ThresholdSolar(tmp_reg=55.0)
                recs, obs, pl, ctx = sim.run(ctrl, site, d0, 45, sc, seed)
                a = devdata.logs_to_arrays(obs); a["meta"] = dict(site=site, seed=seed, scen=sc)
                a["true"] = {k: np.array([r[k] for r in recs]) for k in ("Rc", "Ro", "Rh", "Rs", "theta")}
                runs.append(a)
    return runs


if __name__ == "__main__":
    t0 = time.time()
    dev = pickle.load(open(os.path.join(OUT, "dev_logs.pkl"), "rb"))
    twp = pickle.load(open(os.path.join(OUT, "twin_params.pkl"), "rb"))
    tests = test_logs()
    print("logs", len(tests), round(time.time() - t0, 1), flush=True)
    # ---- tabular GBM (point + quantile)
    Xd, Yd, YUd = build_tabular(dev)
    gbm = {}
    for k in HZ:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(Xd[k], Yd[k])
        ql = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, max_iter=300, learning_rate=0.05, random_state=0).fit(Xd[k], Yd[k])
        qh = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=300, learning_rate=0.05, random_state=0).fit(Xd[k], Yd[k])
        mu_ = np.isfinite(YUd[k])
        mU = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xd[k][mu_], YUd[k][mu_])
        gbm[k] = (m, ql, qh, mU)
    print("gbm", round(time.time() - t0, 1), flush=True)
    gru = {k: train_gru(dev, k) for k in HZ}
    print("gru", round(time.time() - t0, 1), flush=True)
    # conformal calibration set: twin forecasts on three development logs
    inits = {}
    for key, eis in (("physics", True), ("hybrid_noeis", False), ("hybrid", True)):
        sc = {"R": {k: [] for k in HZ}, "U": {k: [] for k in HZ}}
        for j in (0, 7, 14):
            _, s_, _ = twin_forecasts(dev[j], twp[key], eis, rng_seed=100 + j)
            for v in sc:
                for k in HZ:
                    sc[v][k] += s_[v][k]
        inits[key] = sc
    print("calib", round(time.time() - t0, 1), flush=True)
    rows = []; srows = []
    for i, a in enumerate(tests):
        meta = a["meta"]
        R, Rf = r_obs(a); U, Uf = uv_ratio(a)
        Xt, Yt, YUt = build_tabular([a])
        # origins for tabular rows, in the same order as build_tabular
        n = len(a["J"])
        for k in HZ:
            orig = [t for t in range(48, n - 25, EVERY) if np.isfinite(R[t + k])]
            orig = np.array(orig)
            yR = R[orig + k]; yU = U[orig + k]
            # persistence of fouling resistance
            pers = Rf[orig]
            rows.append(dict(model="Persistence", k=k, var="R", **meta, **metrics(yR, pers)))
            rows.append(dict(model="Persistence", k=k, var="U", **meta, **metrics(yU, Uf[orig])))
            # cake-law (linear in filtered volume) extrapolation over the last 24 h
            pc = []
            for t in orig:
                lo = max(0, t - 24)
                vv = np.cumsum(a["J"][lo:t + 1]); rr = R[lo:t + 1]; mm = np.isfinite(rr)
                if mm.sum() >= 4 and np.ptp(vv[mm]) > 0:
                    b = np.polyfit(vv[mm], rr[mm], 1)[0]
                else:
                    b = 0.0
                pc.append(Rf[t] + max(b, 0) * np.sum(a["J"][t + 1:t + 1 + k]))
            rows.append(dict(model="Cake-law", k=k, var="R", **meta, **metrics(yR, np.array(pc))))
            m, ql, qh, mU = gbm[k]
            pg = Rf[orig] + m.predict(Xt[k]); lo = Rf[orig] + ql.predict(Xt[k]); hi = Rf[orig] + qh.predict(Xt[k])
            rows.append(dict(model="GBM-Q", k=k, var="R", **meta, **metrics(yR, pg, lo, hi)))
            rows.append(dict(model="GBM-Q", k=k, var="U", **meta, **metrics(yU, Uf[orig] + mU.predict(Xt[k]))))
            S, F, Y = build_seq([a], k)
            p, mu, sd = gru[k]
            pr = np.array(jax.vmap(gru_apply, in_axes=(None, 0, 0))(p, jnp.array(S), jnp.array(F))) * sd + mu
            rows.append(dict(model="GRU", k=k, var="R", **meta, **metrics(yR, Rf[orig] + pr[:, 0])))
            rows.append(dict(model="GRU", k=k, var="U", **meta, **metrics(yU, Uf[orig] + pr[:, 1])))
        for name, key, eis, ol_ in (("RIS-open-loop", "physics", True, True), ("Twin-physics", "physics", True, False),
                                    ("Twin-hybrid-noEIS", "hybrid_noeis", False, False), ("Proposed twin", "hybrid", True, False)):
            res, _, est = twin_forecasts(a, twp[key], eis, rng_seed=i, open_loop=ol_, init=inits[key])
            tr = a["true"]
            for j, nm in enumerate(("Rc", "Ro", "Rh", "Rs", "theta")):
                srows.append(dict(model=name, state=nm, **meta, rmse=float(np.sqrt(np.mean((est[:, j] - tr[nm]) ** 2))),
                                  mean_true=float(np.mean(tr[nm]))))
            for k in HZ:
                r = res[k]
                if len(r) == 0:
                    continue
                t0_, mR, sR, loR, hiR, yR, mU, sU, yU, qUu, qUl = r.T
                rows.append(dict(model=name, k=k, var="R", **meta, **metrics(yR, mR, loR, hiR)))
                # raw ensemble interval (Gaussian, no conformal) for the proposed twin
                if name == "Proposed twin":
                    rows.append(dict(model="Twin ensemble (raw)", k=k, var="R", **meta,
                                     **metrics(yR, mR, mR - 1.645 * sR, mR + 1.645 * sR)))
                    rows.append(dict(model=name, k=k, var="U", **meta, **metrics(yU, mU, mU - qUl * sU, mU + qUu * sU)))
                    rows.append(dict(model="Twin ensemble (raw)", k=k, var="U", **meta,
                                     **metrics(yU, mU, mU - 1.645 * sU, mU + 1.645 * sU)))
                else:
                    rows.append(dict(model=name, k=k, var="U", **meta, **metrics(yU, mU)))
        print(i, meta, round(time.time() - t0, 1), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "forecast_results.csv"), index=False)
    ds = pd.DataFrame(srows); ds.to_csv(os.path.join(OUT, "state_est_results.csv"), index=False)
    print(ds.groupby(["state", "model"]).rmse.mean().round(4).to_string())
    print(df.groupby(["var", "k", "model"])[["rmse", "mae", "cov", "width", "winkler"]].mean().round(4).to_string())
