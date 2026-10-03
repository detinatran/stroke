"""Figure for the counterfactual-explanation analysis (12.4 cm wide, fonts >= 6.3 pt).
(A) Changes asked for by the explanations; (B) risk reduction the model promises vs the causal estimate;
(C) population MTP dose-response for systolic BP (from minor_stroke_counterfactual.py)."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CE = sys.argv[1] if len(sys.argv) > 1 else "outputs_ce"
RES = sys.argv[2] if len(sys.argv) > 2 else "outputs_minor"
OUT = sys.argv[3] if len(sys.argv) > 3 else "figures"
os.makedirs(OUT, exist_ok=True)
W_IN = 12.4 / 2.54
BLUE, ORANGE, GREEN, VIOLET, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "savefig.dpi": 600, "savefig.bbox": "standard"})


def tag(ax, s, x=-0.3, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


ce = pd.read_csv(os.path.join(CE, "ce_individual.csv"))
pol = pd.read_csv(os.path.join(CE, "ce_causal_check.csv")).set_index("Policy")
S = json.load(open(os.path.join(CE, "ce_summary.json")))
v = ce[ce.status == "valid"]
n, nv = S["n"], len(v)
fig, axes = plt.subplots(1, 3, figsize=(W_IN, 2.3), gridspec_kw=dict(width_ratios=[1, 0.95, 0.9], wspace=1.05))

# A: requested changes
a = axes[0]
only_s = (v.dSBP < 0) & (v.dGLU >= 0); only_g = (v.dSBP >= 0) & (v.dGLU < 0); both = (v.dSBP < 0) & (v.dGLU < 0)
for m_, c_, mk, lab in [(only_s, BLUE, "o", "SBP"), (only_g, ORANGE, "s", "glucose"), (both, VIOLET, "^", "both")]:
    a.scatter(-v.dSBP[m_], -v.dGLU[m_], s=7, color=c_, marker=mk, alpha=0.75, lw=0, label=f"{lab} ({int(m_.sum())})")
a.set_xlabel("SBP decrease (mmHg)"); a.set_ylabel("Glucose decrease (mmol/L)")
a.set_title(f"Requested\nchanges", fontsize=7)
a.legend(loc="upper right", markerscale=1.3, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25)
a.set_xlim(-3, 95); a.set_ylim(-0.5, 16.5)
tag(a, "A", x=-0.42)

# B: promised vs causal, per recipient (pp)
b = axes[1]
rng = np.random.default_rng(0)
bm = np.array([rng.choice(v.model_change.values, nv).mean() for _ in range(2000)])
rows = [("model", 100 * v.model_change.mean(), 100 * np.quantile(bm, 0.025), 100 * np.quantile(bm, 0.975), INK2, "D")]
for key, lab, c_ in [("explanation policy (both)", "causal: both", VIOLET), ("SBP part only", "causal: SBP", BLUE),
                     ("glucose part only", "causal: glucose", ORANGE)]:
    r = pol.loc[key]; k = n / nv
    rows.append((lab, 100 * r.RD * k, 100 * r.Boot_lo * k, 100 * r.Boot_hi * k, c_, "o"))
yy = np.arange(len(rows))[::-1]
for y_, (lab, est, lo, hi, c_, mk) in zip(yy, rows):
    b.errorbar(est, y_, xerr=[[est - lo], [hi - est]], fmt=mk, ms=3.2, color=c_, capsize=2, lw=0.9)
b.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
b.set_yticks(yy); b.set_yticklabels([r[0] for r in rows])
b.set_xlabel("Change per recipient (pp)")
b.set_title("Model promise vs\ncausal estimate", fontsize=7)
b.xaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
tag(b, "B", x=-0.9)

# C: population MTP dose-response for SBP
c = axes[2]
d = pd.read_csv(os.path.join(RES, "minor_dose_response.csv"))
s = d[d.Exposure == "admissionSBP"].sort_values("Delta")
x = np.r_[0, s.Delta]
c.fill_between(x, np.r_[0, 100 * s.Boot_lo], np.r_[0, 100 * s.Boot_hi], color=BLUE, alpha=0.18, lw=0)
c.plot(x, np.r_[0, 100 * s.RD], marker="o", ms=3, lw=1.1, color=BLUE)
c.axhline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
c.set_xlabel("Reduction (mmHg)"); c.set_xlim(-2, 33); c.set_ylabel("Change in END risk (pp)")
c.set_title("Cohort: SBP −δ,\nfloor 140 mmHg", fontsize=7)
c.yaxis.grid(True, color=GRID, lw=0.5); c.set_axisbelow(True)
tag(c, "C", x=-0.55)

fig.subplots_adjust(left=0.105, right=0.985, bottom=0.2, top=0.84)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Figure_ce.{ext}"), bbox_inches=None, pad_inches=0)
from PIL import Image
print("saved; width", round(Image.open(os.path.join(OUT, "Figure_ce.png")).size[0] / 600, 2), "in (type area 4.88)")
