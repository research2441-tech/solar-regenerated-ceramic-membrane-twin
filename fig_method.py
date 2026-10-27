import os
"""Fig. 2 (membrane design space) and Fig. 3 (fouling fractions, actions and
impedance observation model)."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import membrane_design as md

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
OUT = os.path.dirname(os.path.abspath(__file__))


def fig2():
    W, T, eps, d, Lp, sf = md.design_space()
    sel = md.select_design()
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.95, left=0.085, right=0.975, bottom=0.2, top=0.88))
    ax = axs[0]
    cs = ax.contourf(W, T, Lp, levels=np.arange(0, 700, 50), cmap="Greys", alpha=0.85)
    c2 = ax.contour(W, T, sf, levels=[12.0], colors="black", linestyles="--", linewidths=0.9)
    ax.clabel(c2, fmt=r"$\sigma_f$=12 MPa", fontsize=5.5, inline_spacing=2, manual=[(8, 1060)])
    ax.plot(sel["w_b"], sel["Ts"], marker="*", ms=9, mfc="white", mec="black", mew=0.8, ls="none")
    ax.annotate("selected", (sel["w_b"], sel["Ts"]), xytext=(9.0, 1215), fontsize=5.8,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", mutation_scale=6))
    cb = fig.colorbar(cs, ax=ax, fraction=0.07, pad=0.03); cb.set_label(r"$L_p$ (LMH bar$^{-1}$)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_xlabel(r"biochar $w_b$ (wt%)"); ax.set_ylabel(r"sintering $T_s$ ($^\circ$C)"); ax.set_title("(a) permeability", fontsize=7)
    ax = axs[1]
    cs = ax.contourf(W, T, sf, levels=np.arange(0, 40, 3), cmap="Greys", alpha=0.85)
    c3 = ax.contour(W, T, eps, levels=[0.35, 0.40, 0.45, 0.50], colors="black", linewidths=0.6)
    ax.clabel(c3, fmt=lambda v: rf"$\varepsilon$={v:.2f}", fontsize=5.0, inline_spacing=1, manual=[(4.2, 1000), (8.3, 1000), (12.5, 1000), (16.7, 1000)])
    ax.plot(sel["w_b"], sel["Ts"], marker="*", ms=9, mfc="white", mec="black", mew=0.8, ls="none")
    cb = fig.colorbar(cs, ax=ax, fraction=0.07, pad=0.03); cb.set_label(r"$\sigma_f$ (MPa)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    ax.set_xlabel(r"biochar $w_b$ (wt%)"); ax.set_ylabel(r"sintering $T_s$ ($^\circ$C)"); ax.set_title("(b) flexural strength, porosity", fontsize=7)
    ax = axs[2]
    dp = np.logspace(np.log10(0.05), np.log10(3.0), 120)
    lrv = [md.sieving_lrv(x) for x in dp]
    ax.semilogx(dp, lrv, "k-", lw=1.0)
    for x, lab, off in ((0.65, r"$E.\ coli$ (0.65 $\mu$m)", (0.12, 3.2)), (2.0, r"MP (2 $\mu$m)", (0.7, 5.3))):
        y = md.sieving_lrv(x)
        ax.plot([x], [y], "ko", ms=3.5, mfc="white")
        ax.annotate(lab, (x, y), xytext=off, fontsize=5.6, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xlabel(r"particle diameter ($\mu$m)"); ax.set_ylabel("sieving LRV (–)"); ax.set_ylim(0, 7.2)
    ax.set_title("(c) separation layer", fontsize=7); ax.grid(alpha=0.3, lw=0.4)
    fig.savefig(f"{OUT}/fig2_design.png", dpi=400); plt.close(fig)


def fig3():
    fig = plt.figure(figsize=(7.16, 2.55))
    ax = fig.add_axes([0.165, 0.17, 0.40, 0.66])
    # removal per event at reference conditions, computed with the plant rate laws
    I_eff = 600 * 0.85 * np.exp(-0.15 * 0.6)
    g = (I_eff / 400.0) ** 0.7
    def lh(R0, k, hours=3.0, K=2.0, n=180):
        R = R0
        for _ in range(n):
            R -= k * g * R / (K + R) * hours / n
        return 1 - R / R0
    rem = np.array([[0.82, 0.0, 0.03, 0.0, 0.0],
                    [0.82, lh(1.0, 0.35), lh(2.0, 0.6), 0.0, 1 - np.exp(-0.25 * g * 3.0)],
                    [0.95, 0.85, 0.90, 0.65, 0.70]])
    cols = [r"$R_c$", r"$R_h$", r"$R_o$", r"$R_s$", r"$\theta$"]
    rows = ["backwash\n(30 s, every 0.5 h)", "solar regeneration\n(3 h, 600 W m$^{-2}$)", "chemical clean\n(2 h, NaOCl + citric)"]
    for i_ in range(3):
        for j_ in range(5):
            v = rem[i_, j_]
            y = 2 - i_
            if v > 0.005:
                ax.scatter([j_], [y], s=30 + 330 * v, c=[[1 - 0.75 * v] * 3], edgecolors="black", linewidths=0.6, zorder=3)
                ax.text(j_, y - 0.46, f"{100 * v:.0f}%", ha="center", va="center", fontsize=5.8)
            else:
                ax.text(j_, y, "–", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels(cols, fontsize=7.5)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(rows, fontsize=6)
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-1.0, 2.6)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.grid(alpha=0.25, lw=0.4)
    ax.text(2.0, -0.85, r"fraction of each fouling state removed per event; $\Delta P=\mu J\,(R_m+R_c+R_h+R_o+R_s)$",
            ha="center", va="center", fontsize=5.6)
    # EIS circuit panel (b)
    ax2 = fig.add_axes([0.63, 0.17, 0.35, 0.70])
    w_ = np.logspace(-1, 6, 600) * 2 * np.pi

    def Z(rm, rc, tm=1e-4, tc=0.1, rs=0.2):
        return rs + rm / (1 + 1j * w_ * tm) + rc / (1 + 1j * w_ * tc)
    cases = [("clean", 1.0, 0.02, "-"), (r"cake-dominated ($R_c$=1.2)", 1.0 + 0.0, 0.8 * 1.2 ** 0.9, "--"),
             (r"pore-dominated ($R_o$=2.0)", 1.0 + 0.30 * 2.0, 0.05, ":")]
    for lab, rm, rc, ls in cases:
        z = Z(rm, rc)
        ax2.plot(z.real, -z.imag, "k", ls=ls, lw=1.0, label=lab)
    ax2.set_xlabel(r"Re $Z$ (norm.)"); ax2.set_ylabel(r"$-$Im $Z$ (norm.)")
    ax2.set_aspect("equal", adjustable="datalim"); ax2.grid(alpha=0.3, lw=0.4)
    ax2.legend(fontsize=5.4, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax2.set_ylim(-0.05, 1.25)
    ax2.annotate("membrane arc\n" + r"$y_1$: $R_o$, $R_s$", (0.9, 0.38), xytext=(0.35, 0.85), fontsize=5.4,
                 arrowprops=dict(arrowstyle="-", lw=0.5))
    ax2.annotate("cake arc\n" + r"$y_2$: $R_c$, $R_h$", (1.9, 0.42), xytext=(1.55, 1.05), fontsize=5.4,
                 arrowprops=dict(arrowstyle="-", lw=0.5))
    fig.text(0.005, 0.95, "(a)", fontsize=7); fig.text(0.60, 0.95, "(b)", fontsize=7)
    fig.savefig(f"{OUT}/fig3_fractions.png", dpi=400); plt.close(fig)


if __name__ == "__main__":
    fig2(); fig3()
