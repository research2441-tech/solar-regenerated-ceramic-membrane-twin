import os
"""Statistics and summary tables for the closed-loop experiments."""
import json, os
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
MAIN_ORDER = ["CONV-MF", "PC-FIX", "PC-TRIG", "DD-MPC", "DT-MPC", "DT-RMPC"]
METRICS = ["demand_met", "org_ok", "ppcp", "cod_p", "tmp_mean", "tmp_p95", "norm_perm", "n_cip", "reg_h_d",
           "sec_kwh_m3", "cost_m3", "gwp_m3", "a_end", "life_coat_y", "basic_ok", "lrv_ec", "las_p", "prod_m3_d"]
HIGHER_BETTER = {"demand_met", "org_ok", "ppcp", "norm_perm", "a_end", "life_coat_y", "basic_ok", "lrv_ec"}


def load(exp):
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"{exp}.jsonl"))]
    return pd.DataFrame(rows)


def block_key(df):
    return df.site + "|" + df.scen + "|" + df.seed.astype(str)


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(run, 1.0)
    return adj


def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired(df, ref, others, metric):
    df = df.assign(block=block_key(df))
    P = df.pivot_table(index="block", columns="method", values=metric)
    res = []
    for m in others:
        ok = P[[ref, m]].dropna()
        x, y = ok[ref].values, ok[m].values
        try:
            p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue if np.any(x != y) else 1.0
        except ValueError:
            p = 1.0
        diff = x - y
        bs = np.random.default_rng(0).choice(diff, (4000, len(diff)), replace=True).mean(1)
        res.append(dict(metric=metric, ref=ref, other=m, n=len(diff), mean_diff=float(diff.mean()),
                        ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)), p=float(p),
                        r_rb=rank_biserial(x, y), wins=int(np.sum((x > y) if metric in HIGHER_BETTER else (x < y)))))
    out = pd.DataFrame(res); out["p_holm"] = holm(out.p.values)
    return out


def friedman(df, methods, metric):
    df = df.assign(block=block_key(df))
    P = df.pivot_table(index="block", columns="method", values=metric)[methods].dropna()
    s = stats.friedmanchisquare(*[P[m].values for m in methods])
    sign = -1 if metric in HIGHER_BETTER else 1
    ranks = (sign * P).rank(axis=1).mean()
    return float(s.statistic), float(s.pvalue), ranks


def summary(df, methods, metrics=METRICS):
    g = df.groupby("method")[metrics]
    return g.mean().loc[methods], g.std().loc[methods]


if __name__ == "__main__":
    df = load("main")
    print(df.groupby("method").size())
    m, s = summary(df, [x for x in MAIN_ORDER if x in set(df.method)])
    pd.set_option("display.width", 250)
    print(m.round(3).to_string())
    print(df.groupby(["site", "method"]).org_ok.mean().unstack().round(3))
    print(df.groupby(["scen", "method"]).org_ok.mean().unstack().round(3))
    if "DT-RMPC" in set(df.method):
        for met in ("org_ok", "tmp_mean", "cost_m3", "sec_kwh_m3", "n_cip", "norm_perm", "ppcp"):
            print(paired(df, "DT-RMPC", [x for x in MAIN_ORDER if x != "DT-RMPC" and x in set(df.method)], met).round(4).to_string())
