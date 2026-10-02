import os
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from econml.dml import CausalForestDML, LinearDML

import sich_prediction_v2 as v2
import sich_learning_curve as lc
import sich_pipeline as sp
from sich_multitask_causal import load, prepare, discover2, Avg3, S

try:
    from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
    HAS_FR = True
except Exception:
    HAS_FR = False

FILE_PATH = os.environ.get("SICH_DATA", "dataset.xlsx")
OUT_DIR   = os.environ.get("SICH_OUT", "outputs_novel")
MODULES   = os.environ.get("SICH_MODULES", "5,6,7,9").split(",")
QUICK     = os.environ.get("SICH_QUICK", "0") == "1"
REPEATS   = int(os.environ.get("SICH_REPEATS", 1 if QUICK else 2))
SEED      = 42
DPI       = 300
PAL       = v2.PAL
os.makedirs(OUT_DIR, exist_ok=True)


def out(n):
    return os.path.join(OUT_DIR, n)


def div(t):
    v2.divider(t)


def design(X, cols):
    parts, names = [], []
    for c in cols:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype(float)
            miss = v.isna()
            v = v.fillna(v.median())
            parts.append(((v - v.mean()) / (v.std() + 1e-9)).values[:, None]); names.append(c)
            if miss.any():
                parts.append(miss.values.astype(float)[:, None]); names.append(c + "_missing")
        else:
            d = pd.get_dummies(s.astype(object).where(s.notna(), "missing"), prefix=c, drop_first=True, dtype=float)
            if d.shape[1]:
                parts.append(d.values); names += list(d.columns)
    W = np.hstack(parts) if parts else np.zeros((len(X), 0))
    return W, names


class AvgClf:
    def __init__(self):
        self.ms = [LogisticRegression(C=0.5, max_iter=5000),
                   RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=1, random_state=SEED)]

    def fit(self, W, y):
        self.ms = [clone(m).fit(W, y) for m in self.ms]
        self.classes_ = self.ms[0].classes_
        return self

    def predict_proba(self, W):
        return np.mean([m.predict_proba(W) for m in self.ms], axis=0)


def clf(kind):
    return LogisticRegression(C=0.5, max_iter=5000) if kind == "linear" else AvgClf()


def folds_for(T, Y, k=5, seed=SEED):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(Y)), np.asarray(T) * 2 + np.asarray(Y)))


def summarize(r1, r0, if1, if0):
    n = len(if1)
    rd = r1 - r0
    se = np.std(if1 - if0, ddof=1) / np.sqrt(n)
    lrr = np.log(max(r1, 1e-9) / max(r0, 1e-9))
    se_l = np.std(if1 / max(r1, 1e-9) - if0 / max(r0, 1e-9), ddof=1) / np.sqrt(n)
    rr, rr_lo, rr_hi = np.exp(lrr), np.exp(lrr - 1.96 * se_l), np.exp(lrr + 1.96 * se_l)
    return dict(Risk_T1=r1, Risk_T0=r0, RD=rd, RD_lo=rd - 1.96 * se, RD_hi=rd + 1.96 * se,
                p=2 * stats.norm.sf(abs(rd / se)) if se > 0 else np.nan,
                RR=rr, RR_lo=rr_lo, RR_hi=rr_hi, E_value=sp.evalue_rr(rr), E_value_CI=sp.evalue_ci(rr_lo, rr_hi))


def nuis_bin(W, T, Y, kind, folds):
    n = len(T)
    Wx = W if W.shape[1] else np.ones((n, 1))
    e, m0, m1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in folds:
        e[te] = clf(kind).fit(Wx[tr], T[tr]).predict_proba(Wx[te])[:, 1]
        for arm, arr in ((0, m0), (1, m1)):
            idx = tr[T[tr] == arm]
            yy = Y[idx]
            arr[te] = yy.mean() if yy.min() == yy.max() else clf(kind).fit(Wx[idx], yy).predict_proba(Wx[te])[:, 1]
    return np.clip(e, 0.02, 0.98), m0, m1


def est_bin(method, T, Y, e, m0, m1):
    if method == "crude":
        r1, r0 = Y[T == 1].mean(), Y[T == 0].mean()
        return summarize(r1, r0, T * (Y - r1) / T.mean(), (1 - T) * (Y - r0) / (1 - T).mean())
    if method == "gcomp":
        r1, r0 = m1.mean(), m0.mean()
        return summarize(r1, r0, m1 - r1, m0 - r0)
    if method == "ipw":
        w1, w0 = T / e, (1 - T) / (1 - e)
        r1, r0 = (w1 * Y).sum() / w1.sum(), (w0 * Y).sum() / w0.sum()
        return summarize(r1, r0, w1 * (Y - r1) / w1.mean(), w0 * (Y - r0) / w0.mean())
    psi1 = m1 + T * (Y - m1) / e
    psi0 = m0 + (1 - T) * (Y - m0) / (1 - e)
    return summarize(psi1.mean(), psi0.mean(), psi1 - psi1.mean(), psi0 - psi0.mean())


def pretreatment_cols(X, t_tier, exclude):
    cols = []
    for c in X.columns:
        if c in exclude or c in v2.OUTCOME_COLS or c == "SBP_more_than140_admission":
            continue
        if sp.get_tier(c) <= t_tier:
            cols.append(c)
    return cols


def causal_confounders(full, T_name, cols):
    g = full["g"]
    anc = sp.possible_ancestors(g, [T_name, S])
    desc = sp.descendants(g, T_name)
    return [c for c in cols if (c in anc or (c == "TICI" and "TICI_ordinal" in anc)) and c not in desc]


def forest(ax, labels, est, lo, hi, ref, xlabel, title, color=PAL[2]):
    y = np.arange(len(labels))[::-1]
    ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)], fmt="o", color=color, capsize=3)
    ax.axvline(ref, color="grey", ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title, fontweight="bold")


def module5(X, ys, full):
    div("#5 — WHO IS HARMED OR HELPED BY BRIDGING THROMBOLYSIS? (sICH risk, heterogeneous effects)")
    T = (X["Thrombolysis"].astype(str) == "1").astype(int).values
    Y = ys.values.astype(int)
    conf_full = pretreatment_cols(X, 2, {"Thrombolysis"})
    conf_min = [c for c in ["Age", "NIHSS_at_admission", "ASPECTS_Score", "Onset_to_admission",
                            "Blood_glucose_at_admission", "SBP_at_admission", "Anticoagulant_therapy",
                            "History_of_AF", "Platelet_count_at_admission", "INR_at_admission"] if c in X.columns]
    conf_causal = causal_confounders(full, "Thrombolysis", conf_full)
    onset = pd.to_numeric(X["Onset_to_admission"], errors="coerce").values
    eligible = onset <= 270
    print(f"  Thrombolysis: {T.sum()} treated / {len(T) - T.sum()} untreated; sICH {Y[T == 1].mean():.3f} vs {Y[T == 0].mean():.3f}")
    print(f"  Confounder sets: full pre-treatment ({len(conf_full)}), minimal clinical ({len(conf_min)}), "
          f"causal-graph selected ({len(conf_causal)}): {conf_causal}")
    fo = folds_for(T, Y)
    W_full, wn = design(X, conf_full)
    e_p, m0_p, m1_p = nuis_bin(W_full, T, Y, "flexible", fo)
    e_l, m0_l, m1_l = nuis_bin(W_full, T, Y, "linear", fo)
    W_min, _ = design(X, conf_min)
    e_m, m0_m, m1_m = nuis_bin(W_min, T, Y, "flexible", fo)
    W_c, _ = design(X, conf_causal)
    e_c, m0_c, m1_c = nuis_bin(W_c, T, Y, "flexible", fo)
    trim = (e_p >= 0.1) & (e_p <= 0.9)
    fe = folds_for(T[eligible], Y[eligible])
    e_e, m0_e, m1_e = nuis_bin(W_full[eligible], T[eligible], Y[eligible], "flexible", fe)
    rows = [
        ("PRIMARY: AIPW, full confounders, flexible nuisances", est_bin("aipw", T, Y, e_p, m0_p, m1_p), len(T)),
        ("− adjustment (crude)", est_bin("crude", T, Y, e_p, m0_p, m1_p), len(T)),
        ("− propensity (outcome regression only)", est_bin("gcomp", T, Y, e_p, m0_p, m1_p), len(T)),
        ("− outcome model (IPW only)", est_bin("ipw", T, Y, e_p, m0_p, m1_p), len(T)),
        ("linear instead of flexible nuisances", est_bin("aipw", T, Y, e_l, m0_l, m1_l), len(T)),
        ("minimal clinical confounders", est_bin("aipw", T, Y, e_m, m0_m, m1_m), len(T)),
        ("causal-graph confounders", est_bin("aipw", T, Y, e_c, m0_c, m1_c), len(T)),
        ("+ overlap trimming (0.1 ≤ e ≤ 0.9)", est_bin("aipw", T[trim], Y[trim], e_p[trim], m0_p[trim], m1_p[trim]), int(trim.sum())),
        ("+ eligibility restriction (onset ≤ 4.5 h)", est_bin("aipw", T[eligible], Y[eligible], e_e, m0_e, m1_e), int(eligible.sum())),
    ]
    ab = pd.DataFrame([{"Configuration": r[0], "N": r[2], **r[1]} for r in rows]).round(4)
    ab.to_csv(out("m5_ate_ablation.csv"), index=False)
    print("\n  (A) Average effect of thrombolysis on sICH risk — component ablation")
    print(ab[["Configuration", "N", "Risk_T1", "Risk_T0", "RD", "RD_lo", "RD_hi", "p", "RR", "E_value", "E_value_CI"]].to_string(index=False))

    psi1 = m1_p + T * (Y - m1_p) / e_p
    psi0 = m0_p + (1 - T) * (Y - m0_p) / (1 - e_p)
    psi = psi1 - psi0
    learners = ["S-learner", "T-learner", "DR-learner", "Causal forest"]
    taus = {}
    for name in learners:
        tau = np.zeros(len(T))
        for tr, te in fo:
            if name == "S-learner":
                m = RandomForestClassifier(300, min_samples_leaf=10, n_jobs=1, random_state=SEED).fit(np.column_stack([W_full[tr], T[tr]]), Y[tr])
                tau[te] = (m.predict_proba(np.column_stack([W_full[te], np.ones(len(te))]))[:, 1]
                           - m.predict_proba(np.column_stack([W_full[te], np.zeros(len(te))]))[:, 1])
            elif name == "T-learner":
                ms = [RandomForestClassifier(300, min_samples_leaf=10, n_jobs=1, random_state=SEED).fit(W_full[tr][T[tr] == a], Y[tr][T[tr] == a]) for a in (0, 1)]
                tau[te] = ms[1].predict_proba(W_full[te])[:, 1] - ms[0].predict_proba(W_full[te])[:, 1]
            elif name == "DR-learner":
                tau[te] = RandomForestRegressor(300, min_samples_leaf=20, n_jobs=1, random_state=SEED).fit(W_full[tr], psi[tr]).predict(W_full[te])
            else:
                cf = CausalForestDML(model_y=RandomForestRegressor(200, min_samples_leaf=10, random_state=SEED),
                                     model_t=RandomForestClassifier(200, min_samples_leaf=10, random_state=SEED),
                                     discrete_treatment=True, n_estimators=400, min_samples_leaf=10, cv=3, random_state=SEED)
                cf.fit(Y[tr], T[tr], X=W_full[tr])
                tau[te] = np.ravel(cf.effect(W_full[te]))
        taus[name] = tau

    het_rows = []
    for name, tau in taus.items():
        D = np.column_stack([np.ones(len(tau)), tau - tau.mean()])
        beta, *_ = np.linalg.lstsq(D, psi, rcond=None)
        res = psi - D @ beta
        XtXi = np.linalg.inv(D.T @ D)
        V = XtXi @ (D.T * res ** 2) @ D @ XtXi
        se2 = np.sqrt(V[1, 1])
        q1, q2 = np.quantile(tau, [1 / 3, 2 / 3])
        grp = np.digitize(tau, [q1, q2])
        g_m = [psi[grp == k].mean() for k in range(3)]
        g_se = [psi[grp == k].std(ddof=1) / np.sqrt((grp == k).sum()) for k in range(3)]
        diff, diff_se = g_m[2] - g_m[0], np.sqrt(g_se[2] ** 2 + g_se[0] ** 2)
        pi = (tau <= 0).astype(float)
        gain = (1 - pi) * (psi0 - psi1)
        het_rows.append(dict(Learner=name, BLP_slope=beta[1], BLP_p=2 * stats.norm.sf(abs(beta[1] / se2)),
                             GATES_low=g_m[0], GATES_mid=g_m[1], GATES_high=g_m[2],
                             High_minus_low=diff, HmL_p=2 * stats.norm.sf(abs(diff / diff_se)),
                             Pct_withheld=100 * (1 - pi.mean()), sICH_change_vs_treat_all=gain.mean(),
                             change_lo=gain.mean() - 1.96 * gain.std(ddof=1) / np.sqrt(len(gain)),
                             change_hi=gain.mean() + 1.96 * gain.std(ddof=1) / np.sqrt(len(gain)),
                             _g_se=g_se))
    het = pd.DataFrame(het_rows)
    for i_, nm in enumerate(["GATES_low_se", "GATES_mid_se", "GATES_high_se"]):
        het[nm] = het["_g_se"].apply(lambda v, i_=i_: v[i_])
    het.drop(columns="_g_se").round(4).to_csv(out("m5_hte_ablation.csv"), index=False)
    print("\n  (B) Heterogeneity — CATE-learner ablation (out-of-fold CATE; AIPW-validated)")
    print("      BLP slope > 0 with small p = learner captures real heterogeneity; GATES = AIPW effect in")
    print("      tertiles of predicted CATE; policy = withhold thrombolysis if predicted CATE > 0 (sICH only).")
    print(het.drop(columns="_g_se").round(4).to_string(index=False))
    best = het.sort_values("BLP_p").iloc[0]["Learner"]
    num_conf = [c for c in conf_full if pd.api.types.is_numeric_dtype(X[c])]
    Xr = X[num_conf].astype(float).fillna(X[num_conf].astype(float).median())
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(30, len(X) // 10), random_state=SEED).fit(Xr, taus[best])
    print(f"\n  (C) Interpretable summary of {best} CATE (depth-2 tree; exploratory):")
    print("      " + export_text(tree, feature_names=num_conf, decimals=3).replace("\n", "\n      "))
    print("  CAVEAT: only the sICH harm side is measured (no functional outcome in the data); a withholding")
    print("  policy must not be derived from this analysis alone.")

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    ax[0, 0].hist(e_p[T == 1], bins=25, alpha=0.6, color=PAL[1], label="Thrombolysis")
    ax[0, 0].hist(e_p[T == 0], bins=25, alpha=0.6, color=PAL[0], label="No thrombolysis")
    ax[0, 0].set_xlabel("Propensity score"); ax[0, 0].set_title("(A) Overlap", fontweight="bold"); ax[0, 0].legend()
    for i, r in het.iterrows():
        xs = np.arange(3) + (i - 1.5) * 0.18
        ax[0, 1].errorbar(xs, [r.GATES_low, r.GATES_mid, r.GATES_high], yerr=1.96 * np.array(r._g_se), fmt="o", capsize=3, color=PAL[i], label=r.Learner)
    ax[0, 1].axhline(0, color="grey", ls="--"); ax[0, 1].set_xticks(range(3)); ax[0, 1].set_xticklabels(["Low CATE", "Mid", "High CATE"])
    ax[0, 1].set_ylabel("AIPW effect on sICH risk"); ax[0, 1].set_title("(B) GATES by learner", fontweight="bold"); ax[0, 1].legend(fontsize=7)
    ax[1, 0].hist(taus[best], bins=35, color=PAL[2]); ax[1, 0].axvline(0, color="k", ls="--")
    ax[1, 0].set_xlabel("Predicted CATE (risk difference)"); ax[1, 0].set_title(f"(C) CATE distribution — {best}", fontweight="bold")
    forest(ax[1, 1], [r[0] for r in rows], ab.RD, ab.RD_lo, ab.RD_hi, 0, "Risk difference (thrombolysis − none)", "(D) ATE ablation")
    plt.tight_layout(); plt.savefig(out("m5_thrombolysis.png"), dpi=DPI); plt.close()
    return ab.iloc[0], best


def nuis_multi(W, T, Y, kind, folds, levels):
    n = len(T)
    Wx = W if W.shape[1] else np.ones((n, 1))
    e, mu = np.zeros((n, len(levels))), np.zeros((n, len(levels)))
    for tr, te in folds:
        e[te] = clf(kind).fit(Wx[tr], T[tr]).predict_proba(Wx[te])
        for k, lv in enumerate(levels):
            idx = tr[T[tr] == lv]
            yy = Y[idx]
            mu[te, k] = yy.mean() if yy.min() == yy.max() else clf(kind).fit(Wx[idx], yy).predict_proba(Wx[te])[:, 1]
    return np.clip(e, 0.02, 1.0), mu


def est_multi(method, T, Y, e, mu, levels):
    n = len(T)
    r, IF = np.zeros(len(levels)), np.zeros((n, len(levels)))
    for k, lv in enumerate(levels):
        I = (T == lv).astype(float)
        if method == "crude":
            r[k] = Y[I == 1].mean(); IF[:, k] = I * (Y - r[k]) / I.mean()
        elif method == "gcomp":
            r[k] = mu[:, k].mean(); IF[:, k] = mu[:, k] - r[k]
        elif method == "ipw":
            w = I / e[:, k]; r[k] = (w * Y).sum() / w.sum(); IF[:, k] = w * (Y - r[k]) / w.mean()
        else:
            p = mu[:, k] + I * (Y - mu[:, k]) / e[:, k]; r[k] = p.mean(); IF[:, k] = p - r[k]
    se = IF.std(axis=0, ddof=1) / np.sqrt(n)
    t = np.array(levels, float)
    c = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    slope = (c * r).sum()
    s_se = (IF @ c).std(ddof=1) / np.sqrt(n)
    return r, se, slope, s_se


def module6(X, Xf, ys, full):
    div("#6 — DOSE–RESPONSE OF THROMBECTOMY PASSES ON sICH (multi-valued AIPW)")
    passes = pd.to_numeric(X["Number_of_passes"], errors="coerce")
    keep = (passes >= 1).values
    Xk, Xfk = X[keep].reset_index(drop=True), Xf[keep].reset_index(drop=True)
    T = passes[keep].clip(upper=4).astype(int).values
    Y = ys.values[keep].astype(int)
    levels = [1, 2, 3, 4]
    excl = {"Number_of_passes", "TICI", "EVT_time", "Onset_to_successful_recanalization", "Angioplasty", "Stenting",
            "Onset_to_groinpunture"}
    conf_full = pretreatment_cols(Xk, 2, excl) + [c for c in ["Admission_to_groinpunture"] if c in Xk.columns]
    conf_min = [c for c in ["Occlusion_site", "ASPECTS_Score", "NIHSS_at_admission", "Age", "Thrombolysis"] if c in Xk.columns]
    conf_causal = causal_confounders(full, "Number_of_passes", conf_full)
    mediators = [c for c in ["EVT_time", "TICI_ordinal", "Onset_to_successful_recanalization", "Angioplasty", "Stenting"] if c in Xfk.columns]
    print(f"  Patients with ≥1 pass: {len(T)}; levels 1/2/3/≥4: {[int((T == l).sum()) for l in levels]}")
    print(f"  Crude sICH risk by level: {[round(float(Y[T == l].mean()), 3) for l in levels]}")
    print(f"  Confounders: full pre-procedural ({len(conf_full)}), minimal ({conf_min}), causal-graph ({conf_causal})")
    fo = folds_for(T, Y)
    Wf, _ = design(Xk, conf_full)
    Wm, _ = design(Xk, conf_min)
    Wc, _ = design(Xk, conf_causal)
    Wmed = np.hstack([Wf, design(Xfk, mediators)[0]])
    nf = {"flex": nuis_multi(Wf, T, Y, "flexible", fo, levels), "lin": nuis_multi(Wf, T, Y, "linear", fo, levels),
          "min": nuis_multi(Wm, T, Y, "flexible", fo, levels), "cau": nuis_multi(Wc, T, Y, "flexible", fo, levels),
          "med": nuis_multi(Wmed, T, Y, "flexible", fo, levels)}
    trim = nf["flex"][0].min(axis=1) >= 0.05
    print(f"  Positivity trimming removes {int((~trim).sum())} patient(s)")
    Tb = (T >= 2).astype(int)
    fb = folds_for(Tb, Y)
    nb = {"flex": nuis_bin(Wf, Tb, Y, "flexible", fb), "lin": nuis_bin(Wf, Tb, Y, "linear", fb),
          "min": nuis_bin(Wm, Tb, Y, "flexible", fb), "cau": nuis_bin(Wc, Tb, Y, "flexible", fb),
          "med": nuis_bin(Wmed, Tb, Y, "flexible", fb)}
    cfg = [("PRIMARY: AIPW, full pre-procedural confounders, flexible", "aipw", "flex", None),
           ("− adjustment (crude)", "crude", "flex", None),
           ("− propensity (outcome regression only)", "gcomp", "flex", None),
           ("− outcome model (IPW only)", "ipw", "flex", None),
           ("linear instead of flexible nuisances", "aipw", "lin", None),
           ("minimal confounders", "aipw", "min", None),
           ("causal-graph confounders", "aipw", "cau", None),
           ("+ mediators adjusted (INAPPROPRIATE, bias check)", "aipw", "med", None),
           ("+ positivity trimming (all e_t ≥ 0.05)", "aipw", "flex", trim)]
    rows, curves = [], {}
    for lab, meth, key, mask in cfg:
        e, mu = nf[key]
        eb, m0, m1 = nb[key]
        if mask is not None:
            r, se, sl, sse = est_multi(meth, T[mask], Y[mask], e[mask], mu[mask], levels)
            fp = est_bin(meth, Tb[mask], Y[mask], eb[mask], m0[mask], m1[mask])
            n = int(mask.sum())
        else:
            r, se, sl, sse = est_multi(meth, T, Y, e, mu, levels)
            fp = est_bin(meth, Tb, Y, eb, m0, m1)
            n = len(T)
        curves[lab] = (r, se)
        rows.append(dict(Configuration=lab, N=n, **{f"Risk_{l}{'+' if l == 4 else ''}": r[k] for k, l in enumerate(levels)},
                         RD_per_pass=sl, RDpp_lo=sl - 1.96 * sse, RDpp_hi=sl + 1.96 * sse,
                         RDpp_p=2 * stats.norm.sf(abs(sl / sse)) if sse > 0 else np.nan,
                         FirstPass_RR=fp["RR"], FP_RR_lo=fp["RR_lo"], FP_RR_hi=fp["RR_hi"], FP_E_value=fp["E_value"]))
    drows = []
    for cname, lab in [("crude", cfg[1][0]), ("primary", cfg[0][0])]:
        r_, se_ = curves[lab]
        for k_, l_ in enumerate(levels):
            drows.append(dict(Config=cname, Level=l_, Risk=r_[k_], SE=se_[k_]))
    pd.DataFrame(drows).to_csv(out("m6_dose_response.csv"), index=False)
    try:
        dml = LinearDML(model_y=RandomForestRegressor(200, min_samples_leaf=10, random_state=SEED),
                        model_t=RandomForestRegressor(200, min_samples_leaf=10, random_state=SEED),
                        discrete_treatment=False, cv=5, random_state=SEED)
        dml.fit(Y.astype(float), passes[keep].clip(upper=6).values.astype(float), X=None, W=Wf)
        eff = float(np.ravel(dml.effect(T0=0, T1=1))[0])
        lo, hi = (float(np.ravel(a)[0]) for a in dml.effect_interval(T0=0, T1=1, alpha=0.05))
        rows.append(dict(Configuration="continuous partially-linear DML (linear dose)", N=len(T),
                         RD_per_pass=eff, RDpp_lo=lo, RDpp_hi=hi))
    except Exception as ex:
        print(f"  [WARN] DML ablation failed: {ex}")
    ab = pd.DataFrame(rows).round(4)
    ab.to_csv(out("m6_passes_ablation.csv"), index=False)
    print("\n  Dose–response and first-pass effect — component ablation")
    print("  (RD_per_pass = linear trend in counterfactual sICH risk per extra pass; FirstPass_RR = risk if ≥2 passes vs 1)")
    print(ab.to_string(index=False))
    print("\n  CAVEAT: residual confounding by occlusion difficulty (clot composition, tortuosity) is likely;")
    print("  read E-values accordingly. A 'stop after k passes' rule needs pass-by-pass recanalisation data.")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for i, lab in enumerate([cfg[1][0], cfg[0][0]]):
        r, se = curves[lab]
        ax[0].errorbar(np.array(levels) + (i - 0.5) * 0.1, r, yerr=1.96 * se, fmt="o-", capsize=4, color=[PAL[7], PAL[0]][i],
                       label="Crude" if i == 0 else "AIPW (primary)")
    ax[0].set_xticks(levels); ax[0].set_xticklabels(["1", "2", "3", "≥4"]); ax[0].set_xlabel("Number of passes")
    ax[0].set_ylabel("Counterfactual sICH risk"); ax[0].set_title("(A) Dose–response", fontweight="bold"); ax[0].legend()
    forest(ax[1], ab.Configuration, ab.RD_per_pass, ab.RDpp_lo, ab.RDpp_hi, 0, "Risk difference per extra pass", "(B) Per-pass effect ablation")
    fp_ab = ab.dropna(subset=["FirstPass_RR"])
    forest(ax[2], fp_ab.Configuration, fp_ab.FirstPass_RR, fp_ab.FP_RR_lo, fp_ab.FP_RR_hi, 1, "RR (≥2 passes vs 1)", "(C) First-pass effect ablation", PAL[1])
    ax[2].set_xscale("log")
    plt.tight_layout(); plt.savefig(out("m6_passes.png"), dpi=DPI); plt.close()
    return ab.iloc[0]


class SingleModel:
    def __init__(self, feats, num_all):
        self.feats, self.num_all = list(feats), set(num_all)

    def fit(self, X, y):
        num = [f for f in self.feats if f in self.num_all]
        cat = [f for f in self.feats if f not in self.num_all]
        self.m = clone(lc.models(num, cat, 1.0)["ElasticNet-LR"]).fit(X[self.feats], y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X[self.feats])[:, 1]


def qhat(s, alpha):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(s)[k - 1]


def conf_sets(pc, yc, pt, alpha, mondrian):
    sc = np.where(yc == 1, 1 - pc, pc)
    s0, s1 = pt, 1 - pt
    if mondrian:
        q0, q1 = qhat(sc[yc == 0], alpha), qhat(sc[yc == 1], alpha)
    else:
        q0 = q1 = qhat(sc, alpha)
    return s0 <= q0, s1 <= q1


def pvals(pc, yc, pt):
    sc = np.where(yc == 1, 1 - pc, pc)
    c0, c1 = np.sort(sc[yc == 0]), np.sort(sc[yc == 1])
    p0 = (len(c0) - np.searchsorted(c0, pt, side="left") + 1) / (len(c0) + 1)
    p1 = (len(c1) - np.searchsorted(c1, 1 - pt, side="left") + 1) / (len(c1) + 1)
    return p0, p1


def module7(Xf, num_all, ys):
    div("#7 — CONFORMAL SELECTIVE PREDICTION (distribution-free per-patient guarantees)")
    y = ys.values.astype(int)
    feats = list(Xf.columns)
    bases = {"Ensemble (LR+LGBM+RF)": lambda: Avg3(feats, num_all), "Elastic-net LR only": lambda: SingleModel(feats, num_all)}
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED).split(Xf, y))
    store = []
    for j, (tr, te) in enumerate(folds):
        Xtr, ytr = Xf.iloc[tr].reset_index(drop=True), y[tr]
        for bname, make in bases.items():
            oof = np.zeros(len(tr))
            for itr, iva in StratifiedKFold(5, shuffle=True, random_state=SEED + j).split(Xtr, ytr):
                oof[iva] = make().fit(Xtr.iloc[itr], ytr[itr]).predict(Xtr.iloc[iva])
            p_cross = make().fit(Xtr, ytr).predict(Xf.iloc[te])
            a, b = train_test_split(np.arange(len(tr)), test_size=0.3, stratify=ytr, random_state=SEED + j)
            ms = make().fit(Xtr.iloc[a], ytr[a])
            store.append(dict(fold=j, rep=j // 5, base=bname, te=te, cross=(oof, ytr, p_cross),
                              split=(ms.predict(Xtr.iloc[b]), ytr[b], ms.predict(Xf.iloc[te]))))
        print(f"  fold {j + 1}/{len(folds)} done", flush=True)
    rows = []
    for bname in bases:
        for scheme in ["cross", "split"]:
            for mond in [True, False]:
                for alpha in [0.05, 0.10, 0.20]:
                    reps = []
                    for rep in range(REPEATS):
                        in0s, in1s, yy = [], [], []
                        for s in store:
                            if s["base"] != bname or s["rep"] != rep:
                                continue
                            pc, yc, pt = s[scheme]
                            i0, i1 = conf_sets(pc, yc, pt, alpha, mond)
                            in0s.append(i0); in1s.append(i1); yy.append(y[s["te"]])
                        i0, i1, yv = np.concatenate(in0s), np.concatenate(in1s), np.concatenate(yy)
                        single = i0 ^ i1
                        pred1 = single & i1
                        cov = np.where(yv == 1, i1, i0)
                        flag = i1
                        reps.append(dict(Coverage=cov.mean(), Cov_sICH=cov[yv == 1].mean(), Cov_nonsICH=cov[yv == 0].mean(),
                                         Singleton=single.mean(), Both=(i0 & i1).mean(), Empty=(~i0 & ~i1).mean(),
                                         Acc_singletons=(np.where(pred1, 1, 0)[single] == yv[single]).mean() if single.any() else np.nan,
                                         F1_singletons=f1_score(yv[single], pred1[single].astype(int), zero_division=0) if single.any() else np.nan,
                                         Flag_sens=flag[yv == 1].mean(), Flag_PPV=yv[flag].mean() if flag.any() else np.nan,
                                         Flagged=flag.mean()))
                    rows.append(dict(Base=bname, Calibration=scheme + "-conformal", Mondrian=mond, alpha=alpha,
                                     **pd.DataFrame(reps).mean().to_dict()))
    ab = pd.DataFrame(rows).round(3)
    ab.to_csv(out("m7_conformal_ablation.csv"), index=False)
    prim = ab[(ab.Base == "Ensemble (LR+LGBM+RF)") & (ab.Calibration == "cross-conformal") & (ab.Mondrian)]
    print("\n  (A) PRIMARY: ensemble, cross-conformal, Mondrian (class-conditional)")
    print(prim.to_string(index=False))
    print("\n  (B) Component ablation at alpha = 0.10 (each row changes one component)")
    sub = ab[ab.alpha == 0.10]
    order = [("Ensemble (LR+LGBM+RF)", "cross-conformal", True), ("Ensemble (LR+LGBM+RF)", "cross-conformal", False),
             ("Ensemble (LR+LGBM+RF)", "split-conformal", True), ("Elastic-net LR only", "cross-conformal", True)]
    names = ["PRIMARY", "− Mondrian (marginal)", "split instead of cross", "LR base model"]
    abl = pd.concat([sub[(sub.Base == b) & (sub.Calibration == c) & (sub.Mondrian == m)].assign(Ablation=n)
                     for (b, c, m), n in zip(order, names)])
    print(abl[["Ablation", "Coverage", "Cov_sICH", "Cov_nonsICH", "Singleton", "Both", "Empty", "Acc_singletons",
               "F1_singletons", "Flag_sens", "Flag_PPV", "Flagged"]].to_string(index=False))
    abl.to_csv(out("m7_conformal_components_alpha010.csv"), index=False)

    grid = np.linspace(0.2, 1.0, 17)
    curves = {}
    for bname in bases:
        conf_all, pred_all, y_all, p_all = [], [], [], []
        for s in store:
            if s["base"] != bname or s["rep"] != 0:
                continue
            pc, yc, pt = s["cross"]
            p0, p1 = pvals(pc, yc, pt)
            conf_all.append(1 - np.minimum(p0, p1)); pred_all.append((p1 > p0).astype(int)); y_all.append(y[s["te"]]); p_all.append(pt)
        cf, pr, yv, pp = map(np.concatenate, (conf_all, pred_all, y_all, p_all))
        order_i = np.argsort(-cf)
        res = []
        for g in grid:
            k = max(1, int(round(g * len(cf))))
            idx = order_i[:k]
            res.append(dict(Coverage=g, Accuracy=(pr[idx] == yv[idx]).mean(),
                            F1=f1_score(yv[idx], pr[idx], zero_division=0),
                            Sens=recall_score(yv[idx], pr[idx], zero_division=0),
                            PPV=precision_score(yv[idx], pr[idx], zero_division=0),
                            AUC=roc_auc_score(yv[idx], pp[idx]) if len(np.unique(yv[idx])) == 2 else np.nan,
                            n_sICH=int(yv[idx].sum())))
        curves[bname] = pd.DataFrame(res)
    rc = curves["Ensemble (LR+LGBM+RF)"]
    rc.round(3).to_csv(out("m7_risk_coverage.csv"), index=False)
    print("\n  (C) Selective prediction (primary): performance on the most-confident fraction of patients")
    print(rc.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    xs = np.arange(len(abl))
    ax[0].bar(xs - 0.2, abl.Cov_sICH, 0.4, color=PAL[0], label="sICH coverage")
    ax[0].bar(xs + 0.2, abl.Cov_nonsICH, 0.4, color=PAL[2], label="non-sICH coverage")
    ax[0].axhline(0.9, color="k", ls="--", label="Target 1−α = 0.90")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(abl.Ablation, rotation=15, fontsize=8); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(A) Class-conditional coverage (α = 0.10)", fontweight="bold"); ax[0].legend(fontsize=8)
    bottom = np.zeros(len(abl))
    for col, c in [("Singleton", PAL[1]), ("Both", PAL[3]), ("Empty", PAL[7])]:
        ax[1].bar(xs, abl[col], bottom=bottom, color=c, label=col); bottom += abl[col].values
    ax[1].set_xticks(xs); ax[1].set_xticklabels(abl.Ablation, rotation=15, fontsize=8)
    ax[1].set_title("(B) Prediction-set composition", fontweight="bold"); ax[1].legend(fontsize=8)
    for i, (bname, cdf) in enumerate(curves.items()):
        ax[2].plot(cdf.Coverage, cdf.F1, "o-", color=PAL[i], label=f"sICH F1 — {bname}")
        ax[2].plot(cdf.Coverage, cdf.Accuracy, "s--", color=PAL[i], alpha=0.6, label=f"Accuracy — {bname}")
    ax[2].axhline(0.6, color="k", ls=":", lw=1)
    ax[2].set_xlabel("Fraction of patients the model decides on (rest deferred)"); ax[2].invert_xaxis()
    ax[2].set_title("(C) Risk–coverage (selective prediction)", fontweight="bold"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out("m7_conformal.png"), dpi=DPI); plt.close()
    return prim, rc


def make_rules(Xtr, cols):
    rules = []
    for c in cols:
        s = Xtr[c]
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna().astype(float)
            if v.nunique() <= 1:
                continue
            if v.nunique() == 2:
                rules.append((c, ">=", float(v.max())))
                continue
            for q in sorted(set(np.round(v.quantile([0.25, 0.5, 0.75]).values, 2))):
                if q > v.min():
                    rules.append((c, ">=", float(q)))
        else:
            vc = s.dropna().astype(str).value_counts()
            lv = [l for l in vc.index if vc[l] >= 10]
            if len(vc) == 2:
                rules.append((c, "==", "1" if "1" in vc.index else lv[0]))
            else:
                rules += [(c, "==", l) for l in lv]
    return rules


def apply_rules(X, rules):
    cols, names = [], []
    for c, op, v in rules:
        if op == ">=":
            cols.append((pd.to_numeric(X[c], errors="coerce") >= v).fillna(False).values.astype(float)); names.append(f"{c} ≥ {v:g}")
        else:
            cols.append((X[c].astype(str) == v).values.astype(float)); names.append(f"{c} = {v}")
    return np.column_stack(cols), names


def _int_loss(B, y, ints):
    s = (B @ ints)[:, None]
    if np.ptp(s) == 0:
        return log_loss(y, np.full(len(y), y.mean())), None
    cal = LogisticRegression(C=1e6, max_iter=5000).fit(s, y)
    return log_loss(y, cal.predict_proba(s)[:, 1]), cal


def _int_optimise(B, y, w, k, lb=-5, ub=5):
    supp = np.argsort(-np.abs(w))[:k]
    supp = supp[np.abs(w[supp]) > 1e-8]
    best_ints, best_loss = None, np.inf
    for top in (3, 4, 5):
        ints = np.zeros(B.shape[1])
        ints[supp] = np.clip(np.round(w[supp] * top / max(np.abs(w).max(), 1e-9)), lb, ub)
        loss, _ = _int_loss(B, y, ints)
        for _ in range(30):
            improved = False
            for j in supp:
                for d in (-1, 1):
                    cand = ints.copy()
                    cand[j] = np.clip(cand[j] + d, lb, ub)
                    if cand[j] == ints[j]:
                        continue
                    l2, _ = _int_loss(B, y, cand)
                    if l2 < loss - 1e-9:
                        ints, loss, improved = cand, l2, True
            if not improved:
                break
        if loss < best_loss:
            best_ints, best_loss = ints, loss
    return best_ints


def fit_score(B, y, k, method):
    y = np.asarray(y)
    if method == "fasterrisk" and HAS_FR:
        opt = RiskScoreOptimizer(X=B, y=np.where(y == 1, 1, -1).astype(float), k=k, lb=-5, ub=5, parent_size=10)
        opt.optimize()
        mult, b0, betas = opt.get_models()
        clf_ = RiskScoreClassifier(multiplier=mult[0], intercept=b0[0], coefficients=betas[0], X_train=B)
        return dict(kind="int", coef=np.asarray(betas[0]), b0=float(b0[0]), mult=float(mult[0]),
                    predict=lambda Z, c=clf_: c.predict_prob(Z))
    Cs = np.logspace(-3, 1, 40)
    lr = None
    for C in Cs:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000).fit(B, y)
        if (np.abs(m.coef_) > 1e-8).sum() > k:
            break
        lr = m
    if lr is None:
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=Cs[0], max_iter=5000).fit(B, y)
    w = lr.coef_.ravel()
    if method == "continuous":
        return dict(kind="cont", coef=w, predict=lambda Z, m=lr: m.predict_proba(Z)[:, 1])
    if method == "fasterrisk":
        ints = _int_optimise(B, y, w, k)
    else:
        ints = np.round(w * 5 / max(np.abs(w).max(), 1e-9))
    score = B @ ints
    cal = LogisticRegression(C=1e6, max_iter=5000).fit(score[:, None], y)
    return dict(kind="int", coef=ints, b0=np.nan, mult=np.nan,
                predict=lambda Z, ii=ints, c=cal: c.predict_proba((Z @ ii)[:, None])[:, 1])


def sedan_like(X):
    g = pd.to_numeric(X["Blood_glucose_at_admission"], errors="coerce")
    pts = np.where(g > 12, 2, np.where(g > 8, 1, 0))
    pts = pts + (pd.to_numeric(X["ASPECTS_Score"], errors="coerce") < 10).astype(int)
    pts = pts + (pd.to_numeric(X["Age"], errors="coerce") > 75).astype(int)
    pts = pts + (pd.to_numeric(X["NIHSS_at_admission"], errors="coerce") >= 10).astype(int)
    return np.asarray(pts, float)


def hat_like(X):
    n = pd.to_numeric(X["NIHSS_at_admission"], errors="coerce")
    a = pd.to_numeric(X["ASPECTS_Score"], errors="coerce")
    g = pd.to_numeric(X["Blood_glucose_at_admission"], errors="coerce")
    pts = np.where(n >= 20, 2, np.where(n >= 15, 1, 0))
    pts = pts + np.where(a <= 7, 2, np.where(a <= 9, 1, 0))
    pts = pts + ((X["History_of_diabetes"].astype(str) == "1") | (g > 11.1)).astype(int)
    return np.asarray(pts, float)


def best_thr(y, p):
    return v2.best_threshold(np.asarray(y), np.asarray(p))[0]


def module9(X, Xf, Xb, num_all, num_b, cat_b, ys, yh):
    div("#9 — CAUSALLY-CONSTRAINED SPARSE INTEGER RISK SCORE (bedside points score)")
    print(f"  Integer optimiser: {'FasterRisk (Liu et al., NeurIPS 2022)' if HAS_FR else 'coordinate-descent integer optimisation of logistic loss (FasterRisk not installed)'}")
    y = ys.values.astype(int)
    base_cols = [c for c in X.columns if c not in v2.OUTCOME_COLS]
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED).split(X, y))
    cfgs = [("PRIMARY: integer k=7, causal pool, optimised", "causal", 7, "fasterrisk"),
            ("all-variable pool", "all", 7, "fasterrisk"),
            ("k = 3", "causal", 3, "fasterrisk"),
            ("k = 5", "causal", 5, "fasterrisk"),
            ("L1 + naive rounding instead of optimisation", "causal", 7, "round"),
            ("no integer constraint (sparse continuous LR)", "causal", 7, "continuous")]
    res = []
    for j, (tr, te) in enumerate(folds):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        disc = discover2(Xb.iloc[tr], ys.iloc[tr], yh.iloc[tr], num_b, cat_b)
        cvars = set(disc["causal"]) | set(disc["pa_s"]) | set(disc["pa_h"])
        if "TICI_ordinal" in cvars:
            cvars.add("TICI")
        pools = {"all": base_cols, "causal": [c for c in base_cols if c in cvars] or base_cols}
        cache = {}
        for lab, pool, k, meth in cfgs:
            if pool not in cache:
                rules = make_rules(Xtr, pools[pool])
                cache[pool] = (rules, apply_rules(Xtr, rules)[0], apply_rules(Xte, rules)[0])
            rules, Btr, Bte = cache[pool]
            sc = fit_score(Btr, ytr, k, meth)
            ptr, pte = sc["predict"](Btr), sc["predict"](Bte)
            res.append(dict(Config=lab, Fold=j, AUC=roc_auc_score(yte, pte), PR_AUC=average_precision_score(yte, pte),
                            F1=f1_score(yte, (pte >= best_thr(ytr, ptr)).astype(int), zero_division=0),
                            n_vars=int((np.abs(sc["coef"]) > 1e-8).sum())))
        for lab, fn in [("benchmark: SEDAN-like (no dense-artery sign)", sedan_like), ("benchmark: HAT-like (ASPECTS proxy)", hat_like)]:
            ptr, pte = fn(Xtr), fn(Xte)
            res.append(dict(Config=lab, Fold=j, AUC=roc_auc_score(yte, pte), PR_AUC=average_precision_score(yte, pte),
                            F1=f1_score(yte, (pte >= best_thr(ytr, ptr)).astype(int), zero_division=0), n_vars=np.nan))
        feats = list(Xf.columns)
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED + j)
        Xftr = Xf.iloc[tr].reset_index(drop=True)
        oof = np.zeros(len(tr))
        for itr, iva in inner.split(Xftr, ytr):
            oof[iva] = Avg3(feats, num_all).fit(Xftr.iloc[itr], ytr[itr]).predict(Xftr.iloc[iva])
        pte = Avg3(feats, num_all).fit(Xftr, ytr).predict(Xf.iloc[te])
        res.append(dict(Config="reference: full ML ensemble (61 features)", Fold=j, AUC=roc_auc_score(yte, pte),
                        PR_AUC=average_precision_score(yte, pte),
                        F1=f1_score(yte, (pte >= best_thr(ytr, oof)).astype(int), zero_division=0), n_vars=len(feats)))
        print(f"  fold {j + 1}/{len(folds)} done", flush=True)
    r = pd.DataFrame(res)
    r.to_csv(out("m9_per_fold.csv"), index=False)
    prim = r[r.Config == cfgs[0][0]].sort_values("Fold")
    rows = []
    for lab in r.Config.unique():
        s = r[r.Config == lab].sort_values("Fold")
        t = sp.corrected_ttest(s.AUC.values - prim.AUC.values, len(folds[0][0]), len(folds[0][1])) if lab != cfgs[0][0] else None
        rows.append(dict(Configuration=lab, AUC=s.AUC.mean(), AUC_SD=s.AUC.std(), PR_AUC=s.PR_AUC.mean(), F1=s.F1.mean(),
                         F1_SD=s.F1.std(), Vars=s.n_vars.mean(), dAUC_vs_primary=t["mean"] if t else np.nan,
                         p_vs_primary=t["p"] if t else np.nan))
    ab = pd.DataFrame(rows).round(4)
    ab.to_csv(out("m9_score_ablation.csv"), index=False)
    print("\n  (A) Component ablation (fold mean; Δ and p vs primary via corrected resampled t-test)")
    print(ab.to_string(index=False))

    full = discover2(Xb, ys, yh, num_b, cat_b)
    cvars = set(full["causal"]) | set(full["pa_s"]) | set(full["pa_h"])
    if "TICI_ordinal" in cvars:
        cvars.add("TICI")
    pool = [c for c in base_cols if c in cvars] or base_cols
    rules = make_rules(X, pool)
    B, names = apply_rules(X, rules)
    sc = fit_score(B, y, 7, "fasterrisk")
    coef = np.asarray(sc["coef"], float)
    shift = -coef[coef < 0].sum()
    items = []
    for nm, c in zip(names, coef):
        if c > 0:
            items.append((nm.replace(" = 1", " (yes)"), int(c)))
        elif c < 0:
            items.append((nm.replace(" ≥ ", " < ").replace(" = 1", " (no)").replace(" = ", " ≠ "), int(-c)))
    pts = pd.DataFrame(items, columns=["Item", "Points"])
    total = B @ coef + shift
    risk = sc["predict"](B)
    tab = (pd.DataFrame({"Total": total, "Pred": risk, "Obs": y}).groupby("Total")
           .agg(Patients=("Obs", "size"), Predicted_risk=("Pred", "mean"), Observed_risk=("Obs", "mean")).reset_index())
    pts.to_csv(out("m9_final_score_points.csv"), index=False)
    tab.round(3).to_csv(out("m9_final_score_risk_table.csv"), index=False)
    print(f"\n  (B) Final bedside score (fit on all patients; causal pool of {len(pool)} variables)")
    print(pts.to_string(index=False))
    print("\n  Risk by total points:")
    print(tab.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    abp = ab.iloc[::-1]
    ax[0].barh(abp.Configuration, abp.AUC, xerr=abp.AUC_SD, capsize=3,
               color=[PAL[0] if c.startswith("PRIMARY") else PAL[7] if c.startswith(("benchmark", "reference")) else PAL[2] for c in abp.Configuration])
    ax[0].axvline(0.5, color="grey", ls=":"); ax[0].set_xlim(0.4, 0.9)
    ax[0].set_xlabel("AUC (fold mean ± SD)"); ax[0].set_title("(A) Sparse-score ablation", fontweight="bold")
    ax[1].plot(tab.Predicted_risk, tab.Observed_risk, "o-", color=PAL[0])
    for _, r_ in tab.iterrows():
        ax[1].annotate(f"{int(r_.Total)} pts (n={int(r_.Patients)})", (r_.Predicted_risk, r_.Observed_risk), fontsize=7)
    m = max(tab.Predicted_risk.max(), tab.Observed_risk.max()) * 1.1
    ax[1].plot([0, m], [0, m], "--", color="grey"); ax[1].set_xlabel("Predicted sICH risk"); ax[1].set_ylabel("Observed")
    ax[1].set_title("(B) Final score calibration by total points", fontweight="bold")
    plt.tight_layout(); plt.savefig(out("m9_score.png"), dpi=DPI); plt.close()
    return ab.iloc[0]


def main():
    X, ys, yh = load()
    Xf, num, cat, new, Xb, num_b, cat_b = prepare(X)
    print(f"  Patients {len(ys)}, sICH {int(ys.sum())}, HT {int(yh.sum())}; modules {MODULES}; repeats {REPEATS}")
    full = discover2(Xb, ys, yh, num_b, cat_b) if ("5" in MODULES or "6" in MODULES) else None
    summary = []
    for mod, fn in [("5", lambda: module5(X, ys, full)), ("6", lambda: module6(X, Xf, ys, full)),
                    ("7", lambda: module7(Xf, set(num), ys)),
                    ("9", lambda: module9(X, Xf, Xb, set(num), num_b, cat_b, ys, yh))]:
        if mod not in MODULES:
            continue
        try:
            fn()
            summary.append(f"#{mod}: completed")
        except Exception:
            print(f"\n  [ERROR] module #{mod} failed:\n{traceback.format_exc()}")
            summary.append(f"#{mod}: FAILED")
    div("RUN SUMMARY")
    print("  " + " | ".join(summary))
    print(f"  Outputs in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
