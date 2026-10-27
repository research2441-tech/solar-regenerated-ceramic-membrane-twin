"""Closed-loop experiments E2-E7 (main comparison, ablation, robustness,
sensitivity, long-horizon durability). Results are appended as JSON lines
to results/<exp>.jsonl; one line per run."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, sys, time
import numpy as np
from multiprocessing import Pool
import sim, controllers, mpc, ddmpc

OUT = os.path.dirname(os.path.abspath(__file__))
SITES = ("PHX", "GSO", "CHI")
SCENS = ("normal", "shock", "light")
SEASONS = ((1, 15), (2, 105), (3, 195), (4, 285))
NDAYS = 60
_TP = None; _DD = None; _CI = None


def _load():
    global _TP, _DD, _CI
    if _TP is None:
        _TP = pickle.load(open(os.path.join(OUT, "twin_params.pkl"), "rb"))
        _DD = pickle.load(open(os.path.join(OUT, "ddmpc_models.pkl"), "rb"))
        _CI = pickle.load(open(os.path.join(OUT, "calib_init.pkl"), "rb"))


def make(key, seed, **kw):
    _load()
    if key == "CONV-MF":
        return controllers.Conventional()
    if key == "PC-FIX":
        return controllers.FixedSolar()
    if key == "PC-TRIG":
        return controllers.ThresholdSolar()
    if key == "DD-MPC":
        return ddmpc.DDMPC(_DD, seed=seed)
    if key == "DT-MPC":
        return mpc.TwinMPC(_TP["hybrid"], conformal=False, robust=False, seed=seed, name="DT-MPC")
    if key == "DT-RMPC":
        return mpc.TwinMPC(_TP["hybrid"], seed=seed, calib_init=_CI["hybrid"], **kw)
    if key == "-EIS":
        return mpc.TwinMPC(_TP["hybrid_noeis"], use_eis=False, seed=seed, name="-EIS", calib_init=_CI["hybrid_noeis"])
    if key == "-Residual":
        return mpc.TwinMPC(_TP["physics"], seed=seed, name="-Residual", calib_init=_CI["physics"])
    if key == "-Conformal":
        return mpc.TwinMPC(_TP["hybrid"], conformal=False, robust=True, seed=seed, name="-Conformal")
    if key == "-SolarFcst":
        return mpc.TwinMPC(_TP["hybrid"], solar_forecast=False, seed=seed, name="-SolarFcst", calib_init=_CI["hybrid"])
    raise KeyError(key)


def job(args):
    exp, key, site, d0, scen, seed, ndays, kw, envkw, trace = args
    t0 = time.time()
    ctrl = make(key, seed, **kw)
    recs, obs, pl, ctx = sim.run(ctrl, site, d0, ndays, scen, seed, **envkw)
    s = sim.summarize(recs, pl, ctx, getattr(ctrl, "extra_W", 0.0))
    s = {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in s.items()}
    row = dict(exp=exp, method=key, site=site, day=d0, scen=scen, seed=seed, ndays=ndays, kw=kw, env=str(envkw),
               wall_s=time.time() - t0, **s)
    if hasattr(ctrl, "t_dec") and ctrl.t_dec:
        row["dec_ms"] = float(np.mean(ctrl.t_dec) * 1000); row["dec_ms_p95"] = float(np.percentile(ctrl.t_dec, 95) * 1000)
    if hasattr(ctrl, "calib"):
        c = ctrl.calib
        for var in ("tmp", "th"):
            for lag in mpc.LAGS:
                sc = [x for x in c if x[0] == var and x[1] == lag]
                if sc:
                    row[f"calib_{var}_{lag}_n"] = len(sc)
    if trace:
        keep = ("mode", "level", "J", "tmp", "Rc", "Ro", "Rh", "Rs", "theta", "a", "I", "cod_p", "ppcp", "Vp", "served", "demand", "org_ok")
        tr = {k: [r[k] for r in recs] for k in keep}
        if hasattr(ctrl, "calib"):
            tr["calib"] = ctrl.calib
        if hasattr(ctrl, "aci_tmp"):
            tr["q_tmp"] = {k: v.q() for k, v in ctrl.aci_tmp.items()}
            tr["q_uv"] = {k: v.q() for k, v in ctrl.aci_uv.items()}
        pickle.dump(tr, open(os.path.join(OUT, f"{exp}_{key}_{site}_{scen}_{seed}.pkl"), "wb"))
    return row


def tasks(exp):
    T = []
    if exp == "main":
        for key in ("CONV-MF", "PC-FIX", "PC-TRIG", "DD-MPC", "DT-MPC", "DT-RMPC"):
            for site in SITES:
                for scen in SCENS:
                    for seed, d0 in SEASONS:
                        tr = (scen == "shock" and seed == 1)
                        T.append(("main", key, site, d0, scen, seed, NDAYS, {}, {}, tr))
    elif exp == "ablation":
        for key in ("-EIS", "-Residual", "-Conformal", "-SolarFcst"):
            for site in SITES:
                for scen in SCENS:
                    for seed, d0 in SEASONS:
                        T.append(("ablation", key, site, d0, scen, seed, NDAYS, {}, {}, False))
    elif exp == "robust":
        conds = {"noise3x": dict(sensor=dict(tmp_sd=0.03, uv_sd=0.06, eis_sd=0.09)),
                 "drift": dict(sensor=dict(drift=0.08)),
                 "missing20": dict(sensor=dict(miss=0.20)),
                 "lowsun": dict(irr_scale=0.6),
                 "deact": dict(params=dict(delta_a=0.09, k_corr=1.0e-4))}
        for cname, envkw in conds.items():
            for key in ("PC-FIX", "PC-TRIG", "DD-MPC", "DT-MPC", "DT-RMPC"):
                for scen in ("normal", "shock"):
                    for seed, d0 in SEASONS:
                        T.append(("robust_" + cname, key, "GSO", d0, scen, seed, NDAYS, {}, envkw, False))
        for key in ("PC-FIX", "PC-TRIG", "DT-MPC", "DT-RMPC"):
            for scen in ("normal", "shock"):
                for seed, d0 in SEASONS:
                    T.append(("robust_SDP", key, "SDP", d0, scen, seed, NDAYS, {}, {}, False))
    elif exp == "sens":
        grid = ([dict(alpha=a) for a in (0.05, 0.2)] + [dict(replan=r) for r in (1, 4)] + [dict(n_ens=n) for n in (16, 64)] +
                [dict(cost=dict(org=1.0)), dict(cost=dict(org=4.0)), dict(cost=dict(cip=11.6))])
        for kw in grid:
            for site in ("GSO", "CHI"):
                for scen in ("normal", "shock"):
                    for seed, d0 in SEASONS:
                        T.append(("sens", "DT-RMPC", site, d0, scen, seed, NDAYS, kw, {}, False))
    elif exp == "long":
        for key in ("CONV-MF", "PC-FIX", "PC-TRIG", "DT-RMPC"):
            for site in ("GSO", "CHI"):
                T.append(("long", key, site, 0, "normal", 11, 365, {}, {}, True))
    return T


if __name__ == "__main__":
    exp = sys.argv[1]
    pass
    T = tasks(exp)
    done = set()
    fn = os.path.join(OUT, f"{exp}.jsonl")
    if os.path.exists(fn):
        for line in open(fn):
            r = json.loads(line); done.add((r["exp"], r["method"], r["site"], r["scen"], r["seed"], json.dumps(r["kw"], sort_keys=True)))
    T = [t for t in T if (t[0], t[1], t[2], t[4], t[5], json.dumps(t[7], sort_keys=True)) not in done]
    # longest jobs first
    T.sort(key=lambda t: (0 if "MPC" in t[1] or t[1].startswith("-") else 1, -t[6]))
    print(exp, len(T), "jobs", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", "2"))) as pool, open(fn, "a") as f:
        for i, row in enumerate(pool.imap_unordered(job, T)):
            f.write(json.dumps(row) + "\n"); f.flush()
            print(i, row["exp"], row["method"], row["site"], row["scen"], row["seed"], round(row["wall_s"], 1), round(time.time() - t0), flush=True)
