import os
"""Development (commissioning) logs used to calibrate the twin and train
data-driven baselines: Greensboro climate only, event-triggered operation."""
import numpy as np, pickle, os
import sim, controllers

OUT = os.path.dirname(os.path.abspath(__file__))
KEYS = ["J", "I", "T", "tss_f", "uv_f", "cond", "turb_f", "tmp", "eis1", "eis2", "uv_p", "Vp", "Vf", "Qin", "cip_left"]


def logs_to_arrays(obs):
    d = {k: np.array([o[k] for o in obs], dtype=float) for k in KEYS}
    d["fR"] = np.array([1.0 if o["mode"] == "R" else 0.0 for o in obs])
    d["fC"] = np.array([1.0 if (o["mode"] == "C" and o["cip_left"] == 0) else 0.0 for o in obs])
    d["isC"] = np.array([1.0 if o["mode"] == "C" else 0.0 for o in obs])
    return d


def generate(site="GSO", seeds=(100, 101, 102), days=(0, 120, 240), n_days=60, scens=("normal", "light"), policy=None):
    runs = []
    for seed in seeds:
        for d0 in days:
            for sc in scens:
                ctrl = policy() if policy else controllers.ThresholdSolar(tmp_reg=50.0 + 10 * (seed % 3))
                recs, obs, pl, ctx = sim.run(ctrl, site, d0, n_days, sc, seed)
                a = logs_to_arrays(obs); a["true"] = {k: np.array([r[k] for r in recs]) for k in ("Rc", "Ro", "Rh", "Rs", "theta", "a", "tmp", "cod_p", "ppcp")}
                a["meta"] = dict(site=site, seed=seed, day=d0, scen=sc)
                runs.append(a)
    return runs


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs = generate()
    pickle.dump(runs, open(os.path.join(OUT, "dev_logs.pkl"), "wb"))
    print(len(runs), "dev runs,", sum(len(r["J"]) for r in runs), "hours")
