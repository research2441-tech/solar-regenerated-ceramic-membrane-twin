import os
"""Result figures (grayscale)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "axes.titlesize": 7.2})
OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(os.path.abspath(__file__))
GRAYS = ["#000000", "#3a3a3a", "#6a6a6a", "#8f8f8f", "#b0b0b0", "#d0d0d0"]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "----"]
MK = ["o", "s", "^", "v", "D", "P", "X", "*"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), "-"]
NAMES = {"CONV-MF": "CONV-MF", "PC-FIX": "PC-FIX", "PC-TRIG": "PC-TRIG", "DD-MPC": "DD-MPC", "DT-MPC": "DT-MPC", "DT-RMPC": "DT-RMPC"}


def fig_forecast():
    df = pd.read_csv(f"{RES}/forecast_results.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.985, bottom=0.2, top=0.87))
    models = ["Persistence", "Cake-law", "GBM-Q", "GRU", "RIS-open-loop", "Twin-physics", "Twin-hybrid-noEIS", "Proposed twin"]
    for ai, var in enumerate(("R", "U")):
        ax = axs[ai]
        for i, m in enumerate(models):
            s = df[(df.model == m) & (df["var"] == var)].groupby("k").rmse.mean()
            if len(s) == 0:
                continue
            ax.plot(s.index, s.values, color="black" if m == "Proposed twin" else GRAYS[min(i, 4)], ls=LS[i], marker=MK[i],
                    ms=3.2, lw=1.2 if m == "Proposed twin" else 0.8, mfc="white" if m != "Proposed twin" else "black", label=m)
        ax.set_xticks([6, 12, 24]); ax.set_xlabel("lead time (h)"); ax.grid(alpha=0.3, lw=0.4)
        ax.set_ylabel(r"RMSE of $R_f$ ($10^{12}$ m$^{-1}$)" if var == "R" else "RMSE of UV$_{254}$ ratio (–)")
        ax.set_title("(a) fouling resistance" if var == "R" else "(b) permeate UV$_{254}$ / feed UV$_{254}$")
    axs[0].legend(fontsize=5.0, ncol=2, loc="upper left", handlelength=2.6, columnspacing=0.8)
    ax = axs[2]
    pts = [("GBM-Q", "R", "s"), ("Twin ensemble (raw)", "R", "v"), ("Proposed twin", "R", "o"), ("RIS-open-loop", "R", "^")]
    for m, var, mk in pts:
        s_ = df[(df.model == m) & (df["var"] == var)].groupby("k")[["cov", "width"]].mean()
        ax.plot(s_.width, s_["cov"], color="black", ls="-", lw=0.6, zorder=1)
        fc = "black" if m == "Proposed twin" else "white"
        ax.scatter(s_.width, s_["cov"], s=[10, 22, 40], marker=mk, facecolors=fc, edgecolors="black", linewidths=0.7, zorder=3, label=m)
    ax.axhline(0.9, color="black", lw=0.6, ls=":")
    ax.text(ax.get_xlim()[0] if False else 0.05, 0.905, "target 0.90", fontsize=5.2, transform=ax.get_yaxis_transform(), va="bottom")
    ax.set_xlabel(r"mean 90% interval width ($10^{12}$ m$^{-1}$)"); ax.set_ylabel("empirical coverage")
    ax.set_title("(c) interval calibration, $R_f$"); ax.grid(alpha=0.3, lw=0.4); ax.set_xlim(0, 4.0); ax.set_ylim(0.45, 1.03)
    ax.legend(fontsize=5.0, loc="lower right", markerscale=1.0)
    fig.savefig(f"{OUT}/fig4_forecast.png", dpi=400); plt.close(fig)


def fig_trajectory(site="CHI", scen="shock", seed=1, d0=10, nd=10):
    A = pickle.load(open(f"{RES}/main_DT-RMPC_{site}_{scen}_{seed}.pkl", "rb"))
    B = pickle.load(open(f"{RES}/main_PC-FIX_{site}_{scen}_{seed}.pkl", "rb"))
    Cn = pickle.load(open(f"{RES}/main_CONV-MF_{site}_{scen}_{seed}.pkl", "rb"))
    sl = slice(d0 * 24, (d0 + nd) * 24); t = np.arange(sl.stop - sl.start) / 24.0 + d0
    fig, axs = plt.subplots(4, 1, figsize=(7.16, 4.2), sharex=True, gridspec_kw=dict(hspace=0.48, left=0.075, right=0.865, bottom=0.08, top=0.95))
    ax = axs[0]
    I = np.array(A["I"])[sl]
    ax.fill_between(t, 0, I, color="#d9d9d9", step="mid", lw=0)
    ax.plot(t, I, color="#7a7a7a", lw=0.5, drawstyle="steps-mid")
    for tr, y, lab in ((A, 1.06, "DT-RMPC"), (B, 1.24, "PC-FIX")):
        m = np.array(tr["mode"])[sl]
        r = np.where(m == "R")[0]
        ax.scatter(t[r], np.full(len(r), y * I.max()), marker="|", s=18, c="black", lw=0.9)
        ax.text(1.008, (y * I.max()) / (1.36 * I.max()), lab, fontsize=5.4, ha="left", va="center", transform=ax.transAxes)
        cip = np.where(m == "C")[0]
        if len(cip):
            ax.scatter(t[cip], np.full(len(cip), y * I.max()), marker="x", s=10, c="black", lw=0.7)
    ax.set_ylim(0, 1.36 * I.max()); ax.set_ylabel("GHI (W m$^{-2}$)")
    ax.set_title(f"(a) irradiance and regeneration hours (ticks), {site}, {scen} loading", loc="left")
    for ai, (key, lab) in enumerate((("tmp", "TMP (kPa)"), ("theta", r"occupancy $\theta$"), ("ppcp", "PPCP removal (%)"))):
        ax = axs[ai + 1]
        for tr, nm, ls, c in ((Cn, "CONV-MF", ":", "#6a6a6a"), (B, "PC-FIX", "--", "#3a3a3a"), (A, "DT-RMPC", "-", "black")):
            y = np.array(tr[key], dtype=float)[sl]
            J = np.array(tr["J"])[sl]
            if key in ("tmp", "ppcp"):
                y = np.where(J > 0, y, np.nan)
            if key == "ppcp":
                y = 100 * y
            ax.plot(t, y, ls=ls, color=c, lw=0.9 if nm == "DT-RMPC" else 0.8, label=nm)
        ax.set_ylabel(lab); ax.grid(alpha=0.3, lw=0.4)
        if key == "ppcp":
            ax.axhline(60, color="black", lw=0.6, ls=(0, (1, 1))); ax.text(t[-1], 61, "target", fontsize=5.2, ha="right", va="bottom")
        ax.set_title(["(b) transmembrane pressure", "(c) adsorbent site occupancy", "(d) PPCP removal in permeate"][ai], loc="left")
    axs[1].legend(fontsize=5.6, ncol=1, loc="upper left", bbox_to_anchor=(1.005, 1.0), handlelength=2.2)
    axs[-1].set_xlabel("day of run")
    fig.savefig(f"{OUT}/fig5_trajectory.png", dpi=400); plt.close(fig)


def fig_main_box():
    df = analysis.load("main")
    order = analysis.MAIN_ORDER
    mets = [("org_ok", "organic-compliant volume (%)", 100), ("tmp_mean", "mean TMP (kPa)", 1), ("norm_perm", "normalised permeability (–)", 1),
            ("sec_kwh_m3", "specific energy (kWh m$^{-3}$)", 1), ("n_cip", "chemical cleans per 60 d", 1), ("cost_m3", "screening cost (USD m$^{-3}$)", 1)]
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.4), gridspec_kw=dict(hspace=0.62, wspace=0.42, left=0.075, right=0.985, bottom=0.13, top=0.96))
    for ax, (m, lab, sc) in zip(axs.flat, mets):
        data = [df[df.method == k][m].values * sc for k in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.0),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(["#ffffff", "#e6e6e6", "#cccccc", "#b3b3b3", "#8c8c8c", "#4d4d4d"][i])
        for i, dd in enumerate(data):
            x = np.random.default_rng(i).normal(i + 1, 0.06, len(dd))
            ax.scatter(x, dd, s=3, c="black", alpha=0.45, lw=0, zorder=3)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=5.6)
        ax.set_ylabel(lab, fontsize=6.2); ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.savefig(f"{OUT}/fig6_main.png", dpi=400); plt.close(fig)


def fig_breakdown():
    df = analysis.load("main")
    order = analysis.MAIN_ORDER
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw=dict(wspace=0.22, left=0.07, right=0.985, bottom=0.15, top=0.78))
    fills = ["#ffffff", "#e6e6e6", "#cccccc", "#b3b3b3", "#8c8c8c", "#4d4d4d"]
    for ax, col, cats, title in ((axs[0], "site", ["PHX", "GSO", "CHI"], "(a) by climate"),
                                 (axs[1], "scen", ["light", "normal", "shock"], "(b) by loading scenario")):
        g = df.groupby([col, "method"]).org_ok.agg(["mean", "std"]).reset_index()
        w = 0.13
        for i, m in enumerate(order):
            vals = [g[(g[col] == c) & (g.method == m)]["mean"].values[0] * 100 for c in cats]
            err = [g[(g[col] == c) & (g.method == m)]["std"].values[0] * 100 for c in cats]
            x = np.arange(3) + (i - 2.5) * w
            ax.bar(x, vals, w * 0.92, color=fills[i], edgecolor="black", lw=0.5, hatch=HATCH[i] if i in (1, 3) else "", label=m)
            ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Phoenix", "Greensboro", "Chicago"] if col == "site" else ["light", "mixed", "shock"])
        ax.set_ylabel("organic-compliant volume (%)"); ax.set_title(title, loc="left"); ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_ylim(0, 100)
    axs[0].legend(fontsize=5.6, ncol=6, loc="lower left", bbox_to_anchor=(0.0, 1.13), columnspacing=1.0, handlelength=1.6)
    fig.savefig(f"{OUT}/fig7_breakdown.png", dpi=400); plt.close(fig)


def fig_ablation():
    ab = analysis.load("ablation"); mn = analysis.load("main")
    full = mn[mn.method == "DT-RMPC"]
    variants = ["-EIS", "-Residual", "-Conformal", "-SolarFcst"]
    ab = pd.concat([ab, mn[mn.method.isin(["DT-MPC"])]])
    variants = variants + ["DT-MPC"]
    labels = ["no EIS", "no residual", "fixed 1.28σ", "no GHI fcst.", "no bounds"]
    mets = [("org_ok", "Δ organic-compliant\nvolume (pts)", 100), ("tmp_mean", "Δ mean TMP (kPa)", 1), ("cost_m3", "Δ cost (USD m$^{-3}$)", 1)]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw=dict(wspace=0.5, left=0.08, right=0.99, bottom=0.3, top=0.93))
    key = ["site", "scen", "seed"]
    F = full.set_index(key)
    for ax, (m, lab, sc) in zip(axs, mets):
        means, lo, hi = [], [], []
        for v in variants:
            V = ab[ab.method == v].set_index(key)
            dlt = (V[m] - F[m]).dropna().values * sc
            bs = np.random.default_rng(0).choice(dlt, (2000, len(dlt))).mean(1)
            means.append(dlt.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        x = np.arange(len(variants))
        ax.bar(x, means, 0.62, color="#bdbdbd", edgecolor="black", lw=0.5)
        ax.errorbar(x, means, yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.6)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.4)
        ax.set_ylabel(lab, fontsize=6.2); ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.savefig(f"{OUT}/fig8_ablation.png", dpi=400); plt.close(fig)


def fig_robust():
    rows = []
    for c in ("noise3x", "drift", "missing20", "lowsun", "deact", "SDP"):
        fn = f"{RES}/robust.jsonl"
    rb = analysis.load("robust")
    mn = analysis.load("main")
    base = mn[(mn.site == "GSO") & (mn.scen.isin(["normal", "shock"]))].assign(exp="nominal")
    allr = pd.concat([base, rb])
    conds = [("nominal", "nominal"), ("robust_noise3x", "3× noise"), ("robust_drift", "TMP drift"), ("robust_missing20", "20% missing"),
             ("robust_lowsun", "60% irradiance"), ("robust_deact", "fast deactivation"), ("robust_SDP", "Sand Point")]
    order = ["PC-FIX", "PC-TRIG", "DD-MPC", "DT-MPC", "DT-RMPC"]
    fills = ["#e6e6e6", "#cccccc", "#b3b3b3", "#8c8c8c", "#4d4d4d"]
    fig, axs = plt.subplots(2, 1, figsize=(7.16, 3.1), sharex=True, gridspec_kw=dict(hspace=0.12, left=0.075, right=0.985, bottom=0.1, top=0.85))
    for ax, (m, lab, sc) in zip(axs, (("org_ok", "organic-compliant volume (%)", 100), ("tmp_mean", "mean TMP (kPa)", 1))):
        w = 0.15
        for i, meth in enumerate(order):
            vals = []
            for c, _ in conds:
                s = allr[(allr.exp == c) & (allr.method == meth)][m]
                vals.append(s.mean() * sc if len(s) else np.nan)
            x = np.arange(len(conds)) + (i - 2) * w
            ax.bar(x, vals, w * 0.92, color=fills[i], edgecolor="black", lw=0.5, hatch="////" if i == 1 else "", label=meth)
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.3, lw=0.4)
    axs[1].set_xticks(range(len(conds))); axs[1].set_xticklabels([c[1] for c in conds], fontsize=6.2)
    axs[0].legend(fontsize=5.6, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.03))
    fig.savefig(f"{OUT}/fig9_robust.png", dpi=400); plt.close(fig)


def fig_long():
    order = ["CONV-MF", "PC-FIX", "PC-TRIG", "DT-RMPC"]
    st = {"CONV-MF": (":", "#6a6a6a"), "PC-FIX": ("--", "#3a3a3a"), "PC-TRIG": ("-.", "#8f8f8f"), "DT-RMPC": ("-", "black")}
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.1), sharex=True, gridspec_kw=dict(hspace=0.25, wspace=0.38, left=0.07, right=0.985, bottom=0.12, top=0.88))
    from membrane_design import viscosity
    import twin
    for r, site in enumerate(("GSO", "CHI")):
        for m in order:
            fn = f"{RES}/long_{m}_{site}_normal_11.pkl"
            if not os.path.exists(fn):
                continue
            tr = pickle.load(open(fn, "rb"))
            J = np.array(tr["J"]); tmp = np.array(tr["tmp"], dtype=float); n = len(J)
            day = np.arange(n) // 24
            Rm = twin.RM[m != "CONV-MF"]
            with np.errstate(invalid="ignore", divide="ignore"):
                L = np.where(J > 0, 1.0, np.nan)
            Rf = np.array(tr["Rc"]) + np.array(tr["Ro"]) + np.array(tr["Rh"]) + np.array(tr["Rs"])
            perm = Rm / (Rm + Rf)
            dm = pd.Series(perm).groupby(day).mean()
            ls, c = st[m]
            axs[r, 0].plot(dm.index, dm.values, ls=ls, color=c, lw=0.8, label=m)
            org = pd.Series(np.array(tr["org_ok"], dtype=float) * (J > 0)).groupby(day).sum() / pd.Series((J > 0) * 1.0).groupby(day).sum().replace(0, np.nan)
            axs[r, 1].plot(org.index, org.rolling(7, min_periods=1).mean().values * 100, ls=ls, color=c, lw=0.8)
            mode = np.array(tr["mode"]); prev = np.concatenate([np.array(["I"]), mode[:-1]])
            cip = np.cumsum((mode == "C") & (prev != "C"))
            axs[r, 2].plot(np.arange(n) / 24.0, cip, ls=ls, color=c, lw=0.8)
        axs[r, 0].set_ylabel(f"{site}: norm. permeability")
        axs[r, 1].set_ylabel("org. compliant (%, 7-d)")
        axs[r, 2].set_ylabel("cumulative CIP")
        for a in axs[r]:
            a.grid(alpha=0.3, lw=0.4)
    for a in axs[1]:
        a.set_xlabel("day of year")
    axs[0, 0].legend(fontsize=5.4, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.25))
    fig.savefig(f"{OUT}/fig10_long.png", dpi=400); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:]
    for w in which:
        globals()["fig_" + w]()
