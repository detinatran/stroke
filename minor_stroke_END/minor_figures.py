"""Figures for the minor-stroke END analysis (12.4 cm wide, fonts >= 6.3 pt)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "outputs_minor"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
W_IN = 12.4 / 2.54
BLUE, ORANGE, VIOLET, INK2, GRID = "#2a78d6", "#eb6834", "#4a3aa7", "#52514e", "#e4e3df"
COL = {"admissionSBP": BLUE, "Blood glucose on admission": ORANGE}
MK = {"admissionSBP": "o", "Blood glucose on admission": "s"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.0})


def tag(ax, s, x=-0.2, y=1.12):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


d = pd.read_csv(os.path.join(RES, "minor_dose_response.csv"))
m = pd.read_csv(os.path.join(RES, "minor_main_policies.csv"))
fig, axes = plt.subplots(1, 3, figsize=(W_IN, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1.0], wspace=1.0))
for k, e in enumerate(["admissionSBP", "Blood glucose on admission"]):
    ax = axes[k]; s = d[d.Exposure == e].sort_values("Delta")
    x = np.r_[0, s.Delta]; y = np.r_[0, 100 * s.RD]
    ax.fill_between(x, np.r_[0, 100 * s.Boot_lo], np.r_[0, 100 * s.Boot_hi], color=COL[e], alpha=0.18, lw=0)
    ax.plot(x, y, marker=MK[e], ms=3, lw=1.1, color=COL[e])
    ax.axhline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
    ax.set_title(f"{s.Label.iloc[0]}\n(floor {s.Floor.iloc[0]:g} {s.Unit.iloc[0]})", fontsize=7)
    ax.set_xlabel(f"Reduction ({s.Unit.iloc[0]})"); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    tag(ax, "AB"[k], x=-0.42 if k == 0 else -0.25)
axes[0].set_ylabel("Change in END risk (pp)")
ax = axes[2]
r = m[m.Exposure == "DAPT"].set_index("Policy")
labs = ["everyone on DAPT", "no one on DAPT", "DAPT vs no DAPT"]
yy = np.arange(len(labs))[::-1]
for y_, l in zip(yy, labs):
    v = r.loc[l]
    ax.errorbar(100 * v.RD, y_, xerr=[[100 * (v.RD - v.Boot_lo)], [100 * (v.Boot_hi - v.RD)]], fmt="^", ms=3.2, color=VIOLET, capsize=2, lw=0.9)
ax.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
ax.set_yticks(yy); ax.set_yticklabels(["all DAPT", "no DAPT", "DAPT − none"])
ax.set_xlabel("Change (pp)"); ax.set_title("DAPT (exploratory,\nsee timing caveat)", fontsize=7)
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
tag(ax, "C", x=-0.85)
fig.savefig(os.path.join(OUT, "Figure_minor_END.png")); fig.savefig(os.path.join(OUT, "Figure_minor_END.pdf"))
from PIL import Image
print("saved; width", round(Image.open(os.path.join(OUT, "Figure_minor_END.png")).size[0] / 600, 2), "in (type area 4.88)")

# ---------------------------------------------------------------- Figure 2: baseline-risk strata and ablations
st = pd.read_csv(os.path.join(RES, "minor_risk_strata.csv"))
ab = pd.read_csv(os.path.join(RES, "minor_ablation.csv"))
fig = plt.figure(figsize=(W_IN, 2.3))
gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=1.25, left=0.08, right=0.98, bottom=0.2, top=0.88)
a = fig.add_subplot(gs[0, 0])
for k, e in enumerate(["admissionSBP", "Blood glucose on admission"]):
    s = st[st.Exposure == e]
    x = s.Tertile.values + (k - 0.5) * 0.2
    a.errorbar(x, 100 * s.DR_RD, yerr=[100 * (s.DR_RD - s.DR_lo), 100 * (s.DR_hi - s.DR_RD)], fmt=MK[e], ms=3, color=COL[e],
               capsize=1.6, lw=0.8, label={"admissionSBP": "SBP −20", "Blood glucose on admission": "Glucose −2"}[e])
a.axhline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
a.set_xticks([1, 2, 3]); a.set_xticklabels(["low", "mid", "high"]); a.set_xlim(0.5, 3.5)
lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
a.set_xlabel("Baseline-risk tertile"); a.set_ylabel("DR change in END risk (pp)")
a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
a.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
tag(a, "A", x=-0.42, y=1.04)
b = fig.add_subplot(gs[0, 1])
short = {"PRIMARY": "primary (DR)", "− outcome": "IPW only", "− density": "g-computation only", "linear": "linear nuisances",
         "minimal": "minimal adjustment", "no ratio": "no ratio truncation"}
rows = []
for e in ["admissionSBP", "Blood glucose on admission"]:
    s = ab[ab.Exposure == e]
    for key, lab in short.items():
        r = s[s.Configuration.str.startswith(key)].iloc[0]
        rows.append((e, lab, r.RD, r.RD_lo, r.RD_hi))
yy = np.arange(len(rows))[::-1]
for y_, (e, lab, rd_, lo, hi) in zip(yy, rows):
    b.errorbar(100 * rd_, y_, xerr=[[100 * (rd_ - lo)], [100 * (hi - rd_)]], fmt=MK[e], ms=2.8, color=COL[e], capsize=1.5, lw=0.8)
b.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
xl = b.get_xlim(); b.set_xlim(xl[0], xl[1] + 1.1)
b.set_yticks(yy); b.set_yticklabels([r[1] for r in rows], fontsize=6.3)
b.axhline(len(short) - 0.5, color=GRID, lw=0.6)
b.text(0.98, yy[0] - 0.5, "SBP\n−20 mmHg", transform=b.get_yaxis_transform(), fontsize=6.3, color=BLUE, va="center", ha="right")
b.text(0.98, yy[-1] + 3.0, "Glucose\n−2 mmol/L", transform=b.get_yaxis_transform(), fontsize=6.3, color=ORANGE, va="center", ha="right")
b.set_xlabel("Change in END risk (pp)")
b.xaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
tag(b, "B", x=-0.62, y=1.04)
fig.savefig(os.path.join(OUT, "Figure_minor_strata_ablation.png")); fig.savefig(os.path.join(OUT, "Figure_minor_strata_ablation.pdf"))
print("saved fig2; width", round(Image.open(os.path.join(OUT, "Figure_minor_strata_ablation.png")).size[0] / 600, 2), "in")
