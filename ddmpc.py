"""Data-driven MPC baseline: identical plan set, cost and schedule as the
twin scheduler, but the fouling/breakthrough model is a pair of one-step
gradient-boosted regressors trained on the commissioning logs (no physics,
no fouling fractions, point forecasts)."""
import os, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import twin, forecast
from mpc import TwinMPC, COST, TMP_LIM, COD_LIM, PPCP_LIM, AREA
from membrane_design import viscosity

OUT = os.path.dirname(os.path.abspath(__file__))


def onestep_dataset(runs):
    X, YR, YU = [], [], []
    for a in runs:
        R, Rf = forecast.r_obs(a); U, Uf = forecast.uv_ratio(a)
        n = len(a["J"]); sreg = 99.0; scip = 999.0
        for t in range(1, n - 1):
            sreg = 0.0 if a["fR"][t] else sreg + 1; scip = 0.0 if a["isC"][t] else scip + 1
            x = [Rf[t], Uf[t], a["J"][t + 1] / 50.0, a["fR"][t + 1], a["fC"][t + 1], max(a["I"][t + 1], 0) / 500.0,
                 a["T"][t], np.log(max(a["tss_f"][t], 1)), np.log(max(a["uv_f"][t] / 0.02, 1)), a["cond"][t] / 1000.0,
                 np.log(max(a["turb_f"][t], 1)), min(sreg, 72) / 24.0, min(scip, 720) / 240.0]
            X.append(x); YR.append(Rf[t + 1] - Rf[t]); YU.append(Uf[t + 1] - Uf[t])
    return np.array(X), np.array(YR), np.array(YU)


def train(runs):
    X, YR, YU = onestep_dataset(runs)
    mR = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X, YR)
    mU = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X, YU)
    return mR, mU


class DDMPC(TwinMPC):
    name = "DD-MPC"

    def __init__(self, models, replan=2, seed=0):
        super().__init__({"psi": None, "nn": None}, n_ens=2, replan=replan, use_eis=False, conformal=False,
                         robust=False, seed=seed, extra_W=0.0)
        self.mR, self.mU = models

    def reset(self, ctx, plant):
        self.ctx = ctx; self.plant = plant; self.rng = np.random.default_rng(self.seed + 17)
        self.o_hist = []; self.plan = None; self.plan_t = -999; self.snap = {}
        from controllers import LevelControl
        self.lc = LevelControl(); self.t_dec = []; self.sreg = 99.0; self.scip = 999.0
        self.Rlast = 0.0; self.Ulast = 0.4
        self.kf = _Dummy()

    def observe(self, h, o):
        self.o_hist.append(o)
        self.sreg = 0.0 if o["mode"] == "R" else self.sreg + 1
        self.scip = 0.0 if o["mode"] == "C" else self.scip + 1
        if o["J"] > 0 and np.isfinite(o["tmp"]):
            self.Rlast = o["tmp"] * 1e3 / (viscosity(o["T"]) * o["J"] / 3.6e6) / 1e12 - twin.RM[True]
        if o["J"] > 0 and np.isfinite(o["uv_p"]):
            self.Ulast = o["uv_p"] / max(o["uv_f"], 1e-6)
        est = np.array([0.0, max(self.Rlast, 0), 0.0, 0.0, 0.0, 1.0])
        self.kf.Z = est[None]

    def _evaluate(self, plans, exo, H=24):
        mode, J, unmet = self._schedule(plans, exo, H)
        P = len(plans)
        R = np.full(P, self.Rlast); U = np.full(P, self.Ulast)
        sreg = np.full(P, self.sreg); scip = np.full(P, self.scip)
        E = np.zeros(P); tmp = np.zeros((P, H)); th = np.zeros((P, H))
        for k in range(H):
            x = np.stack([R, U, J[:, k] / 50.0, (mode[:, k] == 2) * 1.0, (mode[:, k] == 4) * 1.0, np.full(P, exo["I"][k] / 500.0),
                          np.full(P, exo["T"][k]), np.full(P, np.log(exo["tss"][k])), np.full(P, np.log(exo["doc"][k])),
                          np.full(P, exo["cond"][k] / 1000.0), np.full(P, np.log(exo["turb"][k])),
                          np.minimum(sreg, 72) / 24.0, np.minimum(scip, 720) / 240.0], 1)
            R = np.maximum(R + self.mR.predict(x), 0.0); U = np.clip(U + self.mU.predict(x), 0.0, 1.0)
            sreg = np.where(mode[:, k] == 2, 0.0, sreg + 1); scip = np.where(mode[:, k] >= 3, 0.0, scip + 1)
            t = twin.tmp_from_R(R, J[:, k], exo["T"][k])
            tmp[:, k] = np.where(J[:, k] > 0, t, 0.0)
            E += t * 1e3 * J[:, k] * AREA / 1000.0 / 0.5 / 3.6e6
            th[:, k] = np.clip((U * 0.02 / twin.UV_K - 1 + twin.R0_COD + twin.RA_COD) / twin.RA_COD, 0, 1)
        cod_dis = exo["doc"][None, :] * 2.9
        codU = cod_dis * 0.03 + cod_dis * (1 - twin.R0_COD - twin.RA_COD * (1 - th))
        ppL = twin.ppcp_removal(th)
        filt = J > 0; vol = J * AREA / 1000.0
        org_bad = filt * (np.maximum(codU - COD_LIM, 0) / 10.0 + np.maximum(PPCP_LIM - ppL, 0) / 0.05)
        c = COST; cip_n = np.array([p[3] for p in plans]) * 1.0
        cost = (c["elec"] * (E + c["recirc_kw"] * (mode == 2).sum(1)) + c["cip"] * cip_n + c["unmet"] * unmet +
                c["org"] * (vol * org_bad).sum(1) + c["tmp_risk"] * (filt & (tmp > TMP_LIM)).sum(1) +
                c["term_rev"] * R + c["term_th"] * th[:, -1] * 10)
        return cost, dict(mode=mode)


class _Dummy:
    def __init__(self):
        self.Z = np.zeros((1, 6)); self.Psi = np.zeros((1, 4))


if __name__ == "__main__":
    runs = pickle.load(open(os.path.join(OUT, "dev_logs.pkl"), "rb"))
    pickle.dump(train(runs), open(os.path.join(OUT, "ddmpc_models.pkl"), "wb"))
    print("ok")
