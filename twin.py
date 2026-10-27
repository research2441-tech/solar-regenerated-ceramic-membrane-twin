"""Physics-informed digital twin of the photocatalytic ceramic membrane.

State z = [Rc, Ro, Rs, theta, a]  (cake, photo-reversible organic, irreversible
scale, adsorbent site occupancy, photocatalyst activity)
Parameters psi = [log alpha_c, log alpha_o, log k_ph, log alpha_s] (online
random-walk parameters in the ensemble Kalman filter).

The same hourly transition is written once with an array backend `xp`
(numpy for filtering/control, jax.numpy for training by reverse-mode
differentiation through multi-step rollouts).
"""
import numpy as np
from membrane_design import select_design, intrinsic_resistance

D = select_design()
RM = {True: intrinsic_resistance(D["w_b"], D["Ts"], True) / 1e12,
      False: intrinsic_resistance(D["w_b"], D["Ts"], False) / 1e12}

NOM = dict(log_ac=np.log(0.28), log_ao=np.log(0.035), log_kph=np.log(0.6), log_as=np.log(6e-5),
           eta_bw=0.82, J_crit=45.0, Ro_max=12.0, gamma_T=0.045, cond0=400.0,
           K_R=2.0, beta=0.7, I_ref=400.0, tau_h=0.85, eps_t=0.06, eps_p=0.15, d_ch=0.6,
           kappa_c=1.5, kappa_s=0.4, q_cap=110.0, k_theta=0.25, k_corr=2e-5,
           k_hard=0.020, k_ph_hard=0.35)
# EIS observation gains (from a separate characterisation step)
EIS_G = np.array([[0.12, 0.30, 0.19, 0.50],    # eis1 <- Rc, Ro, Rh, Rs
                  [0.80, 0.03, 0.61, 0.05]])   # eis2
UV_K = 0.012  # permeate UV254 per mg C L-1
R0_COD, RA_COD = 0.35, 0.40
PPCP_FORM = {"tcs": (0.15, 0.85, 0.7), "mep": (0.05, 0.70, 1.2), "bp3": (0.10, 0.75, 1.0)}
PPCP_W = {"tcs": 1.8, "mep": 4.5, "bp3": 1.8}


def viscosity(T, xp=np):
    return 2.414e-5 * 10 ** (247.8 / (T + 133.15))


def nn_init(rng, n_in=6, n_h=16, n_out=2):
    return {"W1": rng.normal(0, 0.3, (n_in, n_h)), "b1": np.zeros(n_h),
            "W2": rng.normal(0, 0.05, (n_h, n_out)), "b2": np.zeros(n_out)}


def nn_feat(u, xp=np):
    return xp.stack([u["J"] / 50.0, (u["T"] - 20.0) / 10.0, xp.log(u["tss"] / 60.0),
                     xp.log(u["doc"] / 40.0), xp.log(u["turb"] / 60.0), (u["cond"] - 600.0) / 300.0], axis=-1)


def nn_mult(nn, u, xp=np):
    if nn is None:
        return 1.0, 1.0
    f = nn_feat(u, xp)
    h = xp.tanh(f @ nn["W1"] + nn["b1"])
    o = 0.7 * xp.tanh(h @ nn["W2"] + nn["b2"])
    m = xp.exp(o)
    return m[..., 0], m[..., 1]


def light(I, mode_R, mode_F, turb, Rc, xp=np, P=NOM):
    att = xp.where(mode_R > 0.5, xp.exp(-P["eps_p"] * P["d_ch"]),
                   xp.where(mode_F > 0.5, xp.exp(-P["eps_t"] * turb * P["d_ch"]),
                            xp.exp(-P["eps_t"] * 0.5 * turb * P["d_ch"])))
    return I * P["tau_h"] * att * xp.exp(-P["kappa_c"] * Rc)


def step(z, psi, u, nn=None, xp=np, P=NOM, coated=True):
    """One-hour transition. z [...,6] = [Rc, Ro, Rh, Rs, theta, a]; psi [...,4];
    u dict of arrays (J flux, fR regen flag, fC cip-completion flag, I
    irradiance, T, tss, doc, cond, turb). Returns new z and the hour-mean
    total fouling resistance (for TMP)."""
    Rc, Ro, Rh, Rs, th, a = z[..., 0], z[..., 1], z[..., 2], z[..., 3], z[..., 4], z[..., 5]
    ac, ao, kph, as_ = xp.exp(psi[..., 0]), xp.exp(psi[..., 1]), xp.exp(psi[..., 2]), xp.exp(psi[..., 3])
    J = u["J"]; filt = (J > 0) * 1.0
    mc, mo = nn_mult(nn, u, xp)
    eta = P["eta_bw"] * xp.exp(-0.05 * Rh)
    m = u["tss"] * J / 1000.0  # g m-2 h-1
    # two half-hour filtration cycles, each closed by a backwash
    after = (1 - eta) ** 2 * Rc + ac * mc * m / 2 * ((1 - eta) ** 2 + (1 - eta))
    Rc_mean = 0.5 * (Rc + after) + ac * mc * m / 4.0
    Rc_n = xp.where(filt > 0.5, after, xp.where(u["fR"] > 0.5, (1 - eta) * Rc, Rc))
    hard = P["k_hard"] * xp.where(filt > 0.5, Rc_mean, Rc_n)
    Rc_n = Rc_n - hard
    dRo = ao * mo * u["doc"] * J / 1000.0 * (xp.maximum(J, 1e-6) / P["J_crit"]) ** 0.6 * \
        xp.exp(P["gamma_T"] * (u["T"] - 20.0)) * (1 - Ro / P["Ro_max"])
    dRs = as_ * J * xp.maximum(u["cond"] - P["cond0"], 0.0) / 1000.0 * xp.exp(0.03 * (u["T"] - 20.0))
    dth = J * u["doc"] / 1000.0 / P["q_cap"] * (1 - th) * (1.0 if coated else 4.0)
    if coated:
        Ieff = light(u["I"], u["fR"], filt, u["turb"], xp.where(u["fR"] > 0.5, Rc_n, Rc_mean), xp, P)
        g = a * xp.exp(-P["kappa_s"] * Rs) * ((xp.maximum(Ieff, 0.0) + 1e-3) / P["I_ref"]) ** P["beta"]
        g = g * (1 - u["fC"])
    else:
        Ieff = 0.0 * J; g = 0.0 * J
    Ro_n = xp.maximum(Ro + dRo - kph * g * Ro / (P["K_R"] + Ro), 0.0)
    Rh_n = xp.maximum(Rh + hard - P["k_ph_hard"] / 0.6 * kph * g * Rh / (P["K_R"] + Rh), 0.0)
    th_n = xp.clip(th + dth - P["k_theta"] * g * th, 0.0, 1.0)
    a_n = a - P["k_corr"] * Ieff / 1000.0 * a
    Rs_n = Rs + dRs
    cip = u["fC"]
    Rc_n = xp.where(cip > 0.5, 0.05 * Rc_n, Rc_n); Ro_n = xp.where(cip > 0.5, 0.10 * Ro_n, Ro_n)
    Rh_n = xp.where(cip > 0.5, 0.15 * Rh_n, Rh_n)
    Rs_n = xp.where(cip > 0.5, 0.35 * Rs_n, Rs_n); th_n = xp.where(cip > 0.5, 0.30 * th_n, th_n)
    a_n = xp.where(cip > 0.5, 0.97 * a_n, a_n)
    zn = xp.stack([Rc_n, Ro_n, Rh_n, Rs_n, th_n, a_n], axis=-1)
    return zn, Rc_mean + Ro + Rh + Rs


def tmp_from_R(Rtot, J, T, coated=True, xp=np):
    return viscosity(T, xp) * J / 3.6e6 * (RM[coated] + Rtot) * 1e12 / 1e3


def obs_eis(z, xp=np):
    R = z[..., :4]
    return R @ EIS_G.T


def cod_perm(th, cod, cod_dis, xp=np, coated=True):
    ra = RA_COD if coated else 0.15
    return (cod - cod_dis) * 0.03 + cod_dis * (1 - R0_COD - ra * (1 - th))


def uv_perm(th, doc, xp=np, coated=True):
    ra = RA_COD if coated else 0.15
    return UV_K * doc * (1 - R0_COD - ra * (1 - th))


def ppcp_removal(th, xp=np, coated=True):
    num = 0.0; den = 0.0
    for k, (b0, ba, n) in PPCP_FORM.items():
        ba_e = ba if coated else 0.35 * ba
        r = 1 - (1 - b0) * (1 - ba_e * (1 - th) ** n)
        num = num + PPCP_W[k] * r; den += PPCP_W[k]
    return num / den


# ----------------------------------------------------------------------
class EnKF:
    """Joint state-parameter ensemble Kalman filter (stochastic, perturbed
    observations) with covariance inflation and physical projection."""

    def __init__(self, n_ens, rng, coated=True, psi0=None, nn=None, use_eis=True, use_uv=True,
                 q_state=(0.02, 0.01, 0.006, 0.002, 0.01), q_par=0.01, infl=1.03):
        self.N = n_ens; self.rng = rng; self.coated = coated; self.nn = nn
        self.use_eis = use_eis; self.use_uv = use_uv
        psi0 = np.array(psi0 if psi0 is not None else [NOM["log_ac"], NOM["log_ao"], NOM["log_kph"], NOM["log_as"]])
        self.Z = np.zeros((n_ens, 6)); self.Z[:, 5] = 1.0 if coated else 0.0
        self.Z[:, :4] = np.abs(rng.normal(0.0, 0.05, (n_ens, 4)))
        self.psi0 = psi0.copy()
        self.Psi = psi0[None, :] + rng.normal(0, 0.15, (n_ens, 4))
        self.qs = np.array(q_state); self.qp = q_par; self.infl = infl
        self.last_Rbar = np.zeros(n_ens)

    def predict(self, u):
        uu = {k: np.full(self.N, v, dtype=float) for k, v in u.items()}
        Zn, Rbar = step(self.Z, self.Psi, uu, self.nn, np, NOM, self.coated)
        Zn[:, :5] += self.rng.normal(0, 1, (self.N, 5)) * self.qs * (0.2 + np.abs(Zn[:, :5]) ** 0.5)
        self.Psi = np.clip(self.Psi + self.rng.normal(0, self.qp, self.Psi.shape), self.psi0 - 1.2, self.psi0 + 1.2)
        self.Z = self._project(Zn); self.last_Rbar = Rbar; self.last_u = u
        return Rbar

    def _project(self, Z):
        Z[:, :4] = np.maximum(Z[:, :4], 0.0); Z[:, 4] = np.clip(Z[:, 4], 0, 1)
        Z[:, 5] = np.clip(Z[:, 5], 0, 1)
        return Z

    def update(self, o):
        """o: observation dict for the hour just simulated."""
        u = self.last_u
        H, y, r = [], [], []
        if u["J"] > 0 and np.isfinite(o.get("tmp", np.nan)):
            H.append(tmp_from_R(self.last_Rbar, u["J"], u["T"], self.coated)); y.append(o["tmp"])
            r.append((0.01 * o["tmp"] + 0.3) ** 2 + (0.03 * o["tmp"]) ** 2)
        if self.use_eis and np.isfinite(o.get("eis1", np.nan)):
            e = obs_eis(self.Z)
            H.append(e[:, 0]); y.append(o["eis1"]); r.append((0.03 * max(o["eis1"], 0.1)) ** 2 + 0.03 ** 2 + 0.02)
            H.append(e[:, 1]); y.append(o["eis2"]); r.append((0.03 * max(o["eis2"], 0.1)) ** 2 + 0.03 ** 2 + 0.02)
        if self.use_uv and u["J"] > 0 and np.isfinite(o.get("uv_p", np.nan)):
            H.append(uv_perm(self.Z[:, 4], o["uv_f"] / 0.02, coated=self.coated)); y.append(o["uv_p"])
            r.append((0.03 * o["uv_p"]) ** 2 + 1e-5)
        if not H:
            return
        Hx = np.stack(H, 1); y = np.array(y); R = np.diag(r)
        X = np.concatenate([self.Z[:, :5], self.Psi], 1)
        Xm = X.mean(0); X = Xm + self.infl * (X - Xm)
        A = X - X.mean(0); HA = Hx - Hx.mean(0)
        Pxy = A.T @ HA / (self.N - 1); Pyy = HA.T @ HA / (self.N - 1) + R
        K = Pxy @ np.linalg.inv(Pyy)
        Yp = y[None, :] + self.rng.multivariate_normal(np.zeros(len(y)), R, self.N)
        X = X + (Yp - Hx) @ K.T
        Z = np.concatenate([X[:, :5], self.Z[:, 5:6]], 1); Psi = X[:, 5:].copy()
        if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(Psi))):
            return
        self.Z = self._project(Z); self.Psi = np.clip(Psi, self.psi0 - 1.2, self.psi0 + 1.2)

    def cip_reset(self):
        pass


def u_from_obs(o, prev_cip_left):
    """Exogenous/action vector for the hour that produced observation o."""
    fC = 1.0 if (o["mode"] == "C" and o["cip_left"] == 0) else 0.0
    return dict(J=float(o["J"]), fR=1.0 if o["mode"] == "R" else 0.0, fC=fC, I=max(float(o["I"]), 0.0),
                T=float(o["T"]), tss=max(float(o["tss_f"]), 1.0), doc=max(float(o["uv_f"]) / 0.02, 1.0),
                cond=float(o["cond"]), turb=max(float(o["turb_f"]), 1.0))
