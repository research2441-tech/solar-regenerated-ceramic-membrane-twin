"""Risk-aware solar-regeneration scheduler driven by the digital twin.

Every `replan` hours the scheduler enumerates candidate 24-h plans
(flux level x regeneration window x chemical-clean decision), rolls each
plan through the twin ensemble with scenario irradiance forecasts, and
selects the plan with minimum expected cost subject to conformal upper
bounds on TMP and on permeate organic quality.

Conformal calibration is action-conditional: past ensemble snapshots are
replayed under the actions that were actually applied, using only the
exogenous forecasts that were available at the time, and the normalised
residuals update adaptive quantiles (adaptive conformal inference).
"""
import numpy as np
import twin, weather
from twin import step, tmp_from_R, NOM
from plant import DEFAULT, FLUX_LEVELS
from controllers import LevelControl, CAP

AREA = DEFAULT["area"]
LAGS = (3, 6, 12, 24)
COST = dict(elec=0.12, cip=5.8, unmet=3.0, org=2.0, tmp_risk=5.0, corr=60.0 * 2e-5,
            term_rev=0.05, term_s=0.40, term_th=0.60, recirc_kw=0.025)
TMP_LIM = 120.0; COD_LIM = 70.0; PPCP_LIM = 0.60


class ACI:
    """Adaptive conformal inference on one-sided normalised scores."""

    def __init__(self, alpha=0.1, gamma=0.01, window=500, q0=1.64):
        self.alpha = alpha; self.gamma = gamma; self.at = alpha; self.s = []; self.window = window; self.q0 = q0

    def q(self):
        if len(self.s) < 30:
            return self.q0
        lev = min(max(1 - self.at, 0.0), 0.999)
        return float(np.quantile(np.array(self.s[-self.window:]), lev))

    def update(self, score):
        err = 1.0 if score > self.q() else 0.0
        self.at = float(np.clip(self.at + self.gamma * (self.alpha - err), 0.001, 0.5))
        self.s.append(score)


class TwinMPC:
    name = "DT-RMPC"; coated = True

    def __init__(self, tw_params, n_ens=32, replan=2, use_eis=True, conformal=True, robust=True,
                 solar_forecast=True, alpha=0.1, seed=0, name=None, extra_W=2.0, calib_init=None, cost=None):
        self.twp = tw_params; self.N = n_ens; self.replan = replan; self.use_eis = use_eis
        self.conformal = conformal; self.robust = robust; self.solar_forecast = solar_forecast
        self.alpha = alpha; self.seed = seed; self.extra_W = extra_W if use_eis else 0.0
        self.calib_init = calib_init
        self.cost = dict(COST); self.cost.update(cost or {})
        if name:
            self.name = name

    # ------------------------------------------------------------------
    def reset(self, ctx, plant):
        self.ctx = ctx; self.plant = plant
        self.rng = np.random.default_rng(self.seed + 17)
        self.kf = twin.EnKF(self.N, self.rng, True, self.twp["psi"], self.twp["nn"], use_eis=self.use_eis)
        self.o_hist = []; self.plan = None; self.plan_t = -999
        self.aci_tmp = {k: ACI(self.alpha) for k in LAGS}; self.aci_uv = {k: ACI(self.alpha) for k in LAGS}
        if self.calib_init is not None:
            for k in LAGS:
                self.aci_tmp[k].s = list(self.calib_init["tmp"][k]); self.aci_uv[k].s = list(self.calib_init["th"][k])
        self.snap = {}; self.lc = LevelControl(); self.calib = []
        self.t_dec = []

    # ------------------------------------------------------------------
    def _exo_forecast(self, h, H):
        """Exogenous forecasts available at hour h for hours h..h+H-1."""
        ob = self.o_hist
        ghi_hist = np.array([max(o["I"], 0.0) for o in ob])
        if self.solar_forecast:
            I = weather.day_ahead_forecast(ghi_hist, H, h, self.rng, 1)[0]
        else:
            hod = (np.arange(h, h + H)) % 24
            I = np.where((hod >= 8) & (hod <= 16), 450.0, 0.0)
        hod = np.arange(h, h + H) % 24
        out = {"I": I}
        for k_o, k in (("tss_f", "tss"), ("uv_f", "doc"), ("cond", "cond"), ("turb_f", "turb"), ("T", "T"), ("Qin", "Qin")):
            v = np.array([o[k_o] for o in ob[-168:]]) if ob else np.array([60.0])
            hh = (np.arange(len(ob))[-168:]) % 24 if ob else np.array([0])
            prof = np.array([np.nanmean(v[hh == q]) if np.any(hh == q) else np.nanmean(v) for q in range(24)])
            x = prof[hod]
            if k == "doc":
                x = x / 0.02
            out[k] = np.maximum(x, 1e-3)
        out["T"] = np.full(H, ob[-1]["T"]) if ob else np.full(H, 25.0)
        # inferred reuse demand: Vp(t-1) + production(t) - Vp(t)
        if len(ob) > 2:
            dem = []
            for i in range(max(1, len(ob) - 168), len(ob)):
                prod = ob[i]["J"] * AREA / 1000.0 * (1.0 - 0.048)
                dem.append(max(ob[i - 1]["Vp"] + prod - ob[i]["Vp"], 0.0))
            dem = np.array(dem); hh = np.arange(len(dem)) + max(1, len(ob) - 168)
            prof = np.array([dem[(hh % 24) == q].mean() if np.any((hh % 24) == q) else dem.mean() for q in range(24)])
            out["dem"] = prof[hod]
        else:
            out["dem"] = np.full(H, 0.04)
        return out

    # ------------------------------------------------------------------
    def _plans(self, h, exo, H=24):
        I = exo["I"]
        cand = [k for k in range(H) if I[k] >= 200.0]
        if len(cand) > 6:
            cand = list(np.array(cand)[np.linspace(0, len(cand) - 1, 6).round().astype(int)])
        plans = []
        est = self.kf.Z.mean(0)
        allow_cip = (tmp_from_R(est[0] + est[1] + est[2] + est[3], 50.0, exo["T"][0]) > 70.0) or est[3] > 1.0
        for lev in (1, 2, 3):
            for hi in (0.95, 0.75):
                for s in [None] + cand:
                    for d in ((0,) if s is None else (1, 2, 3, 4)):
                        for cip in ((0, 1) if allow_cip else (0,)):
                            plans.append((lev, s, d, cip, hi))
        return plans

    def _schedule(self, plans, exo, H=24):
        """Deterministic tank/mode schedule for every plan (P x H)."""
        P = len(plans)
        mode = np.zeros((P, H), int)  # 0 idle, 1 filter, 2 regen, 3 cip, 4 cip-complete
        J = np.zeros((P, H)); unmet = np.zeros(P)
        o = self.o_hist[-1]
        cip_left = o["cip_left"]
        for p, (lev, s, d, cip, hi) in enumerate(plans):
            Vp, Vf = o["Vp"], o["Vf"]; on = self.lc.on
            cl = cip_left if cip_left > 0 else (2 if cip else 0)
            for k in range(H):
                Vf = min(Vf + exo["Qin"][k], DEFAULT["feed_tank"])
                if cl > 0:
                    cl -= 1; mode[p, k] = 4 if cl == 0 else 3
                elif s is not None and s <= k < s + d:
                    mode[p, k] = 2
                else:
                    if Vp < self.lc.lo * CAP:
                        on = True
                    elif Vp > hi * CAP:
                        on = False
                    if on and Vf > 0.05:
                        mode[p, k] = 1; J[p, k] = FLUX_LEVELS[lev]
                        q = min(J[p, k] * AREA / 1000.0, Vf - 0.02); J[p, k] = q / (AREA / 1000.0)
                        Vf -= q; Vp += q * (1 - 0.048)
                Vp = min(Vp, CAP)
                dmd = exo["dem"][k]; sv = min(dmd, Vp); Vp -= sv; unmet[p] += dmd - sv
        return mode, J, unmet

    def _rollout(self, plans, exo, H=24):
        P = len(plans); N = self.N
        mode, J, unmet = self._schedule(plans, exo, H)
        Z = np.repeat(self.kf.Z[None], P, 0).reshape(P * N, 6)
        Psi = np.repeat(self.kf.Psi[None], P, 0).reshape(P * N, 4)
        Imult = np.clip(self.rng.lognormal(0, 0.35, N), 0.2, 1.6)
        Imult = np.tile(Imult, P)
        tmp = np.zeros((P * N, H)); th = np.zeros((P * N, H)); Rterm = None
        E = np.zeros(P * N); dose = np.zeros(P * N)
        for k in range(H):
            Jk = np.repeat(J[:, k], N); mk = np.repeat(mode[:, k], N)
            u = dict(J=Jk, fR=(mk == 2) * 1.0, fC=(mk == 4) * 1.0, I=exo["I"][k] * Imult, T=np.full(P * N, exo["T"][k]),
                     tss=np.full(P * N, exo["tss"][k]), doc=np.full(P * N, exo["doc"][k]),
                     cond=np.full(P * N, exo["cond"][k]), turb=np.full(P * N, exo["turb"][k]))
            th[:, k] = Z[:, 4]
            Z, Rbar = step(Z, Psi, u, self.twp["nn"], np, NOM, True)
            t = tmp_from_R(Rbar, Jk, u["T"])
            tmp[:, k] = np.where(Jk > 0, t, 0.0)
            E += t * 1e3 * Jk * AREA / 1000.0 / 0.5 / 3.6e6
            dose += u["I"] * NOM["tau_h"] * (mk == 2) / 1000.0
        tmp = tmp.reshape(P, N, H); th = th.reshape(P, N, H); Z = Z.reshape(P, N, 6)
        return mode, J, unmet, tmp, th, Z, E.reshape(P, N).mean(1), dose.reshape(P, N).mean(1)

    def _qk(self, aci, H):
        qs = np.array([aci[k].q() for k in LAGS])
        return np.interp(np.arange(1, H + 1), LAGS, qs)

    def _evaluate(self, plans, exo, H=24):
        mode, J, unmet, tmp, th, Z, E, dose = self._rollout(plans, exo, H)
        med = tmp.mean(1); sd = tmp.std(1) + 0.5
        thm = th.mean(1); ths = th.std(1) + 0.01
        doc = exo["doc"][None, :]
        cod_dis = doc * 2.9
        if self.conformal:
            qT = self._qk(self.aci_tmp, H)[None]; qU = self._qk(self.aci_uv, H)[None]
        else:
            qT = qU = np.full((1, H), 1.2816)
        if self.robust:
            tmpU = med + qT * sd
            # uv_p = K*doc*(1-r0-ra(1-th)): uv-score bound translated to theta bound
            thU = np.clip(thm + qU * ths, 0, 1)
        else:
            tmpU = med; thU = thm
        codU = cod_dis * 0.03 + cod_dis * (1 - twin.R0_COD - twin.RA_COD * (1 - thU))
        ppL = twin.ppcp_removal(thU)
        filt = J > 0
        vol = J * AREA / 1000.0
        org_bad = filt * (np.maximum(codU - COD_LIM, 0) / 10.0 + np.maximum(PPCP_LIM - ppL, 0) / 0.05)
        c = self.cost
        cip_n = np.array([p[3] for p in plans]) * 1.0
        Zm = Z.mean(1)
        cost = (c["elec"] * (E + c["recirc_kw"] * (mode == 2).sum(1)) + c["cip"] * cip_n + c["unmet"] * unmet +
                c["org"] * (vol * org_bad).sum(1) + c["tmp_risk"] * (filt & (tmpU > TMP_LIM)).sum(1) +
                c["corr"] * dose +
                c["term_rev"] * (Zm[:, 0] + Zm[:, 1] + Zm[:, 2]) + c["term_s"] * Zm[:, 3] + c["term_th"] * Zm[:, 4] * 10)
        return cost, dict(med=med, sd=sd, thm=thm, ths=ths, mode=mode)

    # ------------------------------------------------------------------
    def act(self, h):
        if not self.o_hist:
            return "F", 2
        o = self.o_hist[-1]
        if o["cip_left"] > 0:
            return "C", 2
        if self.plan is None or h - self.plan_t >= self.replan:
            import time
            t0 = time.time()
            exo = self._exo_forecast(h, 24)
            plans = self._plans(h, exo)
            cost, info = self._evaluate(plans, exo)
            b = int(np.argmin(cost))
            self.plan = (plans[b], info["mode"][b]); self.plan_t = h
            self.t_dec.append(time.time() - t0)
            self.snap[h] = (self.kf.Z.copy(), self.kf.Psi.copy(), exo)
        (lev, s, d, cip, hi), mrow = self.plan
        k = h - self.plan_t
        m = mrow[k] if k < len(mrow) else 1
        if m in (3, 4):
            return "C", 2
        if m == 2:
            return "R", lev
        Vp, Vf = o["Vp"], o["Vf"]
        self.lc.hi = hi
        return ("F", lev) if self.lc.want(Vp, Vf) else ("I", lev)

    def observe(self, h, o):
        u = twin.u_from_obs(o, 0)
        self.kf.predict(u); self.kf.update(o)
        self.o_hist.append(o)
        if h not in self.snap and h % 1 == 0:
            exo = self._exo_forecast(h + 1, 24) if len(self.o_hist) > 1 else None
            if exo is not None:
                self.snap[h + 1] = (self.kf.Z.copy(), self.kf.Psi.copy(), exo)
        self._calibrate(h, o)
        for k in list(self.snap.keys()):
            if k < h - 30:
                del self.snap[k]

    def _calibrate(self, h, o):
        """Replay snapshots taken `lag` hours ago under the realised actions."""
        if o["J"] <= 0:
            return
        for lag in LAGS:
            t0 = h + 1 - lag
            if t0 not in self.snap:
                continue
            Z, Psi, exo = self.snap[t0]
            Imult = np.clip(self.rng.lognormal(0, 0.35, self.N), 0.2, 1.6)
            Zr = Z.copy()
            for j in range(lag):
                oj = self.o_hist[t0 + j] if t0 + j < len(self.o_hist) else o
                u = dict(J=np.full(self.N, oj["J"]), fR=np.full(self.N, 1.0 if oj["mode"] == "R" else 0.0),
                         fC=np.full(self.N, 1.0 if (oj["mode"] == "C" and oj["cip_left"] == 0) else 0.0),
                         I=exo["I"][j] * Imult, T=np.full(self.N, exo["T"][j]), tss=np.full(self.N, exo["tss"][j]),
                         doc=np.full(self.N, exo["doc"][j]), cond=np.full(self.N, exo["cond"][j]), turb=np.full(self.N, exo["turb"][j]))
                Zr, Rbar = step(Zr, Psi, u, self.twp["nn"], np, NOM, True)
            if np.isfinite(o["tmp"]):
                t = tmp_from_R(Rbar, o["J"], exo["T"][lag - 1])
                sc = (o["tmp"] - t.mean()) / (t.std() + 0.5)
                self.aci_tmp[lag].update(sc)
                self.calib.append(("tmp", lag, o["tmp"], t.mean(), t.std() + 0.5, h))
            if np.isfinite(o["uv_p"]):
                doc = o["uv_f"] / 0.02
                th_obs = (o["uv_p"] / (twin.UV_K * doc) - 1 + twin.R0_COD + twin.RA_COD) / twin.RA_COD
                sc = (th_obs - Zr[:, 4].mean()) / (Zr[:, 4].std() + 0.01)
                self.aci_uv[lag].update(sc)
                self.calib.append(("th", lag, th_obs, Zr[:, 4].mean(), Zr[:, 4].std() + 0.01, h))


def calib_scores(calib):
    out = {"tmp": {k: [] for k in LAGS}, "th": {k: [] for k in LAGS}}
    for var, lag, y, m, sd, h in calib:
        out[var][lag].append((y - m) / sd)
    return out
