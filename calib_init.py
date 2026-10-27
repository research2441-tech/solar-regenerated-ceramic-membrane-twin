"""Conformal calibration set from commissioning-period closed-loop runs
(Greensboro, development seeds) for each twin variant."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pickle, numpy as np
import sim, mpc
OUT = os.path.dirname(os.path.abspath(__file__))
tp = pickle.load(open(os.path.join(OUT, "twin_params.pkl"), "rb"))
res = {}
for key, eis in (("hybrid", True), ("physics", True), ("hybrid_noeis", False)):
    allc = {"tmp": {k: [] for k in mpc.LAGS}, "th": {k: [] for k in mpc.LAGS}}
    for seed, d0, sc in ((100, 60, "normal"), (101, 200, "shock"), (102, 300, "light")):
        c = mpc.TwinMPC(tp[key], use_eis=eis, seed=seed)
        sim.run(c, "GSO", d0, 30, sc, seed)
        s = mpc.calib_scores(c.calib)
        for v in s:
            for k in s[v]:
                allc[v][k] += s[v][k]
    res[key] = allc
    print(key, {v: {k: (len(allc[v][k]), round(float(np.quantile(allc[v][k], 0.9)), 2)) for k in mpc.LAGS} for v in allc})
pickle.dump(res, open(os.path.join(OUT, "calib_init.pkl"), "wb"))
