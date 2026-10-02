"""Print-size figures (12.4 cm wide) for the counterfactual paper.

usage: python make_figures.py [results_dir] [out_dir]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "outputs_counterfactual"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)
W_IN = 12.4 / 2.54
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"SBP_at_admission": BLUE, "Blood_glucose_at_admission": ORANGE, "Admission_to_groinpunture": AQUA}
MK = {"SBP_at_admission": "o", "Blood_glucose_at_admission": "s", "Admission_to_groinpunture": "D"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.0, "pdf.fonttype": 42})
def rd(f):
    d = pd.read_csv(os.path.join(RES, f))
    if "Boot_lo" in d:                      # bootstrap intervals are the primary inference
        d["RD_lo"] = d["Boot_lo"].fillna(d["RD_lo"]); d["RD_hi"] = d["Boot_hi"].fillna(d["RD_hi"])
    return d


def tag(ax, s, x=-0.2, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png")); fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)
    print("saved", name)


def fig_dose_response():
    d = rd("cf_dose_response.csv")
    exps = list(dict.fromkeys(d.Exposure))
    fig, axes = plt.subplots(1, len(exps), figsize=(W_IN, 1.85), sharey=True)
    for k, (ax, e) in enumerate(zip(axes, exps)):
        s = d[d.Exposure == e].sort_values("Delta")
        x = np.r_[0, s.Delta.values]
        y = np.r_[0, 100 * s.RD.values]
        lo = np.r_[0, 100 * s.RD_lo.values]; hi = np.r_[0, 100 * s.RD_hi.values]
        ax.fill_between(x, lo, hi, color=COL[e], alpha=0.18, lw=0)
        ax.plot(x, y, marker=MK[e], ms=3, lw=1.1, color=COL[e])
        ax.axhline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
        ax.set_xlabel(f"Reduction ({s.Unit.iloc[0]})")
        ax.set_title(f"{s.Label.iloc[0]}\n(floor {s.Floor.iloc[0]:g} {s.Unit.iloc[0]})", fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        tag(ax, "ABC"[k], x=-0.12 if k else -0.3, y=1.2)
    axes[0].set_ylabel("Change in sICH risk (pp)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "Figure_cf_dose_response")


def fig_policies_and_strata():
    m = rd("cf_main_policies.csv")
    st = rd("cf_risk_strata.csv")
    fig = plt.figure(figsize=(W_IN, 2.45))
    gs = fig.add_gridspec(1, 2, wspace=0.95, width_ratios=[1.15, 1], left=0.2, right=0.97, bottom=0.2, top=0.9)
    a = fig.add_subplot(gs[0, 0])
    rows = m[~m.Policy.isin(["treated vs untreated"])].reset_index(drop=True)
    short = {"SBP_at_admission": "SBP", "Blood_glucose_at_admission": "Glucose", "Admission_to_groinpunture": "Door-to-groin"}
    labs = []
    for r in rows.itertuples():
        if r.Exposure == "Thrombolysis":
            labs.append(f"Lysis: {r.Policy.replace(' treated', '')} treated")
        else:
            dlt, unit = r.Policy.split(",")[0].split(" ", 1)
            labs.append(f"{short[r.Exposure]} {dlt} {unit}\n(floor {r.Policy.split('below ')[1]})")
    yy = np.arange(len(rows))[::-1]
    for y_, r in zip(yy, rows.itertuples()):
        c = COL.get(r.Exposure, VIOLET); mk = MK.get(r.Exposure, "^")
        a.errorbar(100 * r.RD, y_, xerr=[[100 * (r.RD - r.RD_lo)], [100 * (r.RD_hi - r.RD)]], fmt=mk, ms=3.2, color=c, capsize=2, lw=0.9)
    a.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
    a.set_yticks(yy); a.set_yticklabels(labs)
    a.set_xlabel("Change vs natural course (pp)")
    a.xaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    tag(a, "A", x=-0.62)
    b = fig.add_subplot(gs[0, 1])
    exps = list(dict.fromkeys(st.Exposure))
    for k, e in enumerate(exps):
        s = st[st.Exposure == e]
        x = s.Tertile.values + (k - 1) * 0.18
        b.errorbar(x, 100 * s.DR_RD, yerr=[100 * (s.DR_RD - s.DR_lo), 100 * (s.DR_hi - s.DR_RD)], fmt=MK[e], ms=3,
                   color=COL[e], capsize=1.6, lw=0.8,
                   label={"SBP_at_admission": "SBP", "Blood_glucose_at_admission": "Glucose", "Admission_to_groinpunture": "Door-to-groin"}[e])
    b.axhline(0, color=INK2, ls=(0, (2, 2)), lw=0.6)
    b.set_xticks([1, 2, 3]); b.set_xticklabels(["low", "middle", "high"]); b.set_xlim(0.5, 3.5)
    lo_, hi_ = b.get_ylim(); b.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))
    b.set_xlabel("Baseline-risk tertile"); b.set_ylabel("DR change in sICH risk (pp)")
    b.yaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(loc="upper left", ncol=1, handletextpad=0.3, handlelength=1.0, borderaxespad=0.2)
    tag(b, "B", x=-0.42)
    save(fig, "Figure_cf_policies_strata")


if __name__ == "__main__":
    fig_dose_response()
    fig_policies_and_strata()
