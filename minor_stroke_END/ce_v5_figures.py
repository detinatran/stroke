import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOSMOTE = "--nosmote" in sys.argv                     # drop the SMOTE arm (not reported in the paper)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = ARGS[0] if len(ARGS) > 0 else "outputs_ce_v5"
OUT = ARGS[1] if len(ARGS) > 1 else "figures"
SUFFIX = "_nosmote" if NOSMOTE else ""
os.makedirs(OUT, exist_ok=True)
W_IN = 12.4 / 2.54
BLUE, ORANGE, GREEN, VIOLET, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "savefig.dpi": 600, "savefig.bbox": "standard"})

def err(est, lo, hi):
    return [[max(est - lo, 0.0)], [max(hi - est, 0.0)]]

def tag(ax, s, x=-0.3, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

pol = pd.read_csv(os.path.join(SRC, "ce_v5_policies.csv"))
bins = pd.read_csv(os.path.join(SRC, "ce_v5_calibration_bins.csv"))
perf = pd.read_csv(os.path.join(SRC, "ce_v5_model_performance.csv")).set_index("Model")
if NOSMOTE:
    pol = pol[~pol.Policy.str.contains("SMOTE")].reset_index(drop=True)
    bins = bins[~bins.Model.str.contains("SMOTE")]
    perf = perf[~perf.index.str.contains("SMOTE")]
short = {"CE: threshold, uncalibrated": "CE, threshold (uncal.)", "CE: threshold": "CE, threshold", "CE: goal 10%": "CE, goal −10%", "CE: goal 20%": "CE, goal −20%", "CE: goal 30%": "CE, goal −30%",
         "Rule: SBP -20, floor 140": "Rule: SBP \u221220 (all)", "Tree: DR-learned": "Tree (DR-learned)"}
colors = {"CE: threshold, uncalibrated": INK2, "CE: threshold": "#6ea8f0", "CE: goal 10%": BLUE, "CE: goal 20%": VIOLET, "CE: goal 30%": "#8a7bd8", "Tree: DR-learned": GREEN, "Rule: SBP -20, floor 140": ORANGE}
for p in pol.Policy:
    if p.startswith("CE: monotone+recal, SBP drop"):
        short[p] = "CE, mono + recal., capped"
        colors[p] = "#6ea8f0"
fig = plt.figure(figsize=(W_IN, 3.15))
gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.0], width_ratios=[1.25, 1.0], left=0.30, right=0.985, bottom=0.11, top=0.965,
                      hspace=0.75, wspace=0.42)

a = fig.add_subplot(gs[0, :])
y = np.arange(len(pol))[::-1]
for y_, r in zip(y, pol.itertuples()):
    a.errorbar(r.Cohort_RD_pp, y_, xerr=err(r.Cohort_RD_pp, r.Boot_lo_pp, r.Boot_hi_pp), fmt="o", ms=3.4,
               color=colors.get(r.Policy, VIOLET), capsize=2, lw=0.9)
a.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
a.set_yticks(y)
a.set_yticklabels([short.get(p, p) for p in pol.Policy])
a.set_xlabel("Change in END risk, whole cohort (pp)")
a.xaxis.grid(True, color=GRID, lw=0.5)
a.set_axisbelow(True)
fig.text(0.015, 0.94, "A", fontsize=9, fontweight="bold", va="bottom")

b = fig.add_subplot(gs[1, 0])
ce = pol[pol.Policy.isin(["CE: threshold, uncalibrated", "CE: threshold", "CE: goal 10%"])].reset_index(drop=True)   # stable per-recipient arms; the rest are in the table
yy = np.arange(len(ce))[::-1] * 1.0
for y_, r in zip(yy, ce.itertuples()):
    b.errorbar(r.Promise_pp, y_ + 0.15, xerr=err(r.Promise_pp, r.Promise_lo, r.Promise_hi), fmt="D", ms=3, color=INK2,
               capsize=2, lw=0.9, label="model promise" if y_ == yy[0] else None)
    b.errorbar(r.Per_recipient_pp, y_ - 0.15, xerr=err(r.Per_recipient_pp, r.PerRec_lo, r.PerRec_hi), fmt="o", ms=3.4,
               color=VIOLET, capsize=2, lw=0.9, label="causal estimate" if y_ == yy[0] else None)
b.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
b.set_yticks(yy)
ylab = {"CE: threshold, uncalibrated": "Threshold\n(uncal.)", "CE: threshold": "Threshold", "CE: goal 10%": "Goal −10%", "CE: goal 20%": "Goal −20%", "CE: goal 30%": "Goal −30%"}
b.set_yticklabels([ylab.get(p, p) for p in ce.Policy])
b.set_ylim(-0.7, len(ce) - 0.3)
b.set_xlabel("Change per recipient (pp)")
b.legend(loc="lower center", handletextpad=0.2, borderaxespad=0.1, fontsize=6.3, bbox_to_anchor=(0.45, 1.0), ncol=2, columnspacing=0.8)
b.xaxis.grid(True, color=GRID, lw=0.5)
b.set_axisbelow(True)
fig.text(0.015, 0.47, "B", fontsize=9, fontweight="bold", va="bottom")

c = fig.add_subplot(gs[1, 1])
lab = {"XGBoost": "Uncalibrated", "XGBoost monotone + recalibrated": "Recal.", "XGBoost monotone + recalibrated, with DBP": "Recal. + DBP"}
mcol = {"XGBoost": INK2, "XGBoost monotone + recalibrated": "#6ea8f0", "XGBoost monotone + recalibrated, with DBP": BLUE}
for m, g in bins.groupby("Model", sort=False):
    col = mcol.get(m, INK2)
    c.plot(g.pred, g.obs, marker="o", ms=2.4, lw=0.9, color=col, label=f"{lab.get(m, m)}  {perf.loc[m, 'Cal_slope']:.2f}")
hi = max(bins.pred.max(), bins.obs.max()) * 1.05
c.plot([0, hi], [0, hi], color=INK2, ls=(0, (2, 2)), lw=0.6)
c.set_xlabel("Predicted risk")
c.set_ylabel("Observed END")
c.set_ylim(-0.01, 0.56); c.set_yticks([0, 0.1, 0.2, 0.3])
c.legend(title="calibration slope", title_fontsize=6.3, loc="upper left", fontsize=6.3, handletextpad=0.2,
         borderaxespad=0.1, labelspacing=0.2, handlelength=1.2)
c.yaxis.grid(True, color=GRID, lw=0.5)
c.set_axisbelow(True)
fig.text(0.665, 0.47, "C", fontsize=9, fontweight="bold", va="bottom")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Figure_ce_v5{SUFFIX}.{ext}"), bbox_inches=None, pad_inches=0)
print("saved", os.path.join(OUT, f"Figure_ce_v5{SUFFIX}.png"))
