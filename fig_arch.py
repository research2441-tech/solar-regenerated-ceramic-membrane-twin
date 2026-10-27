"""Fig. 1: system architecture (grayscale, no overlaps)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
OUT = os.path.dirname(os.path.abspath(__file__))


def box(ax, x, y, w, h, title, lines, fill="#FFFFFF", lw=0.9, ls="-", tsize=7.2, lsize=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec="black", lw=lw, ls=ls))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center", fontsize=tsize, fontweight="bold")
    n = len(lines)
    top = y + h - 5.4
    step = (h - 7.0) / max(n, 1)
    step = min(step, 3.2)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - i * step, t, ha="center", va="center", fontsize=lsize)


def arrow(ax, pts, ls="-", lw=0.9, text=None, tpos=None, tsize=5.9, ha="center", va="bottom", rot=0):
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        a = FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-|>" if last else "-", mutation_scale=8, lw=lw,
                            ls=ls, color="black", shrinkA=0, shrinkB=0)
        ax.add_patch(a)
    if text:
        ax.text(tpos[0], tpos[1], text, fontsize=tsize, ha=ha, va=va, rotation=rot,
                bbox=dict(fc="white", ec="none", pad=0.4))


def main():
    fig = plt.figure(figsize=(7.16, 4.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")
    G1, G2 = "#F2F2F2", "#E0E0E0"
    # ---------------- offline calibration lane (top)
    ax.text(1.0, 62.6, "Offline calibration (commissioning logs)", fontsize=6.6, style="italic", va="center")
    box(ax, 2, 50, 20, 10.5, "Commissioning logs", [r"$\mathcal{D}$: 18 runs, 60 d each", "event-triggered operation"], G1)
    box(ax, 30, 50, 25, 10.5, "Multiple-shooting rollouts", [r"$z_{s,t+1}=f(z_{s,t},u_{s,t};\psi,\phi)$", "48-h segments, JAX"], G1)
    box(ax, 63, 50, 22, 10.5, "Calibration loss", [r"TMP, EIS, UV$_{254}$ residuals", "+ continuity + priors"], G1)
    arrow(ax, [(22.5, 55.2), (29.5, 55.2)])
    arrow(ax, [(55.5, 55.2), (62.5, 55.2)])
    # dashed reverse-mode gradient: loss -> twin parameters (psi, phi)
    arrow(ax, [(74, 49.5), (74, 45.5), (60, 45.5), (60, 41.2)], ls=(0, (3, 2)), text=r"reverse-mode gradient $\nabla_{\psi,\phi}\mathcal{L}$",
          tpos=(67.0, 46.1), tsize=5.9)
    # ---------------- online loop (middle)
    box(ax, 5, 19, 19, 22, "Membrane module", ["RHA–kaolin support", "(biochar-templated)", r"g-C$_3$N$_4$/TiO$_2$ layer",
                                                "filter + backwash", "solar regeneration", "chemical clean (CIP)"], G2)
    box(ax, 28, 21, 11.5, 18, "Sensors", ["TMP, $J$, $T$", r"UV$_{254}$ in/out", "turbidity, EC", r"EIS $y_1, y_2$", "irradiance"], "#FFFFFF")
    box(ax, 43, 19, 24, 22, "Fraction-resolved twin", [r"$z=[R_c,R_o,R_h,R_s,\theta,a]$", "physics rate laws", r"$\times$ bounded residual $g_\phi$",
                                                        "ensemble Kalman filter", r"joint $(z,\psi)$ update"], G1)
    box(ax, 73, 19, 25, 22, "Risk-aware scheduler", ["candidate 24-h plans", "flux, regeneration window,", "CIP, tank set-point",
                                                      "conformal bounds on", r"TMP and $\theta$; min. cost"], G2)
    arrow(ax, [(24.5, 30), (27.5, 30)])
    arrow(ax, [(39.5, 30), (42.5, 30)])
    arrow(ax, [(67.5, 30), (72.5, 30)], text="ensemble", tpos=(70.0, 31.4), tsize=5.6)
    # actions back to the plant (above the middle row, below the top lane)
    arrow(ax, [(98.4, 30.0), (99.6, 30.0), (99.6, 0.9), (2.2, 0.9), (2.2, 30.0), (4.5, 30.0)],
          text=r"control actions $a_t$", tpos=(33.6, 0.9), tsize=5.9, va="center")
    # ---------------- bottom row
    box(ax, 5, 3, 19, 11, "Permeate tank", ["toilet flushing,", "garden reuse"], "#FFFFFF")
    box(ax, 43, 3, 24, 11, "Action-conditional replay", [r"adaptive conformal $q_k$", "k = 3, 6, 12, 24 h"], G1)
    box(ax, 73, 3, 25, 11, "Exogenous forecasts", ["irradiance (past days),", "influent and demand profile"], "#FFFFFF")
    arrow(ax, [(14.5, 18.5), (14.5, 14.5)], text="permeate", tpos=(15.4, 16.0), tsize=5.6, ha="left", va="center")
    arrow(ax, [(33.8, 20.5), (33.8, 10.0), (42.5, 10.0)], text="outcomes", tpos=(34.6, 13.6), tsize=5.6, ha="left", va="center")
    arrow(ax, [(55, 18.5), (55, 14.5)], text="snapshots", tpos=(55.9, 16.5), tsize=5.6, ha="left", va="center")
    arrow(ax, [(67.5, 11.5), (70.0, 11.5), (70.0, 24.0), (72.5, 24.0)], text=r"$q_k$", tpos=(69.4, 17.5), tsize=6.0, ha="right", va="center")
    arrow(ax, [(72.5, 6.0), (67.5, 6.0)])
    arrow(ax, [(85.5, 14.5), (85.5, 18.5)])
    # external inputs to the module
    ax.text(10.0, 45.2, "greywater", fontsize=5.8, va="center", ha="left")
    arrow(ax, [(9.2, 46.9), (9.2, 41.5)])
    ax.text(20.6, 45.2, "sunlight", fontsize=5.8, va="center", ha="left")
    arrow(ax, [(19.8, 46.9), (19.8, 41.5)])
    fig.savefig(f"{OUT}/fig1_architecture.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    main()
