"""Causal counterfactual analysis of modifiable peri-procedural factors for sICH after thrombectomy.

Replaces the model-based DiCE counterfactuals of sich_pipeline.py with population and individual
causal counterfactuals:

  * modified treatment policies (MTPs) for continuous exposures, e.g. "lower admission systolic BP by
    20 mmHg, but not below 140", estimated with a cross-fitted doubly robust (one-step) estimator whose
    density ratio is learned by classification (stack observed and shifted exposures);
  * static interventions for bridging thrombolysis (everyone / no one treated) with AIPW;
  * counterfactual dose-response over the size of each shift;
  * individual counterfactual risk changes and doubly robust effects within baseline-risk tertiles;
  * positivity diagnostics (density-ratio range, effective sample size);
  * one-component-at-a-time ablations: estimator, nuisance learner, adjustment set, ratio truncation,
    and a mediator-adjusted negative control.

Adjustment sets follow the temporal tiers (variables that precede or share the exposure tier, minus
graph descendants and known mediators). Causal discovery (PC + GES under tier constraints, from
sich_multitask_causal.discover2) is run inside every training fold to report how often each exposure
is a possible ancestor of sICH, and once on the full cohort for the graph-based adjustment ablation.

Run:  SICH_DATA=dataset.xlsx python sich_counterfactual.py      (SICH_QUICK=1 for a smoke test)
"""
import os
import json
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("SICH_OUT", "outputs_counterfactual")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from joblib import Parallel, delayed

import sich_pipeline as sp
import sich_multitask_causal as smc

OUT = os.environ["SICH_OUT"]
QUICK = os.environ.get("SICH_QUICK", "0") == "1"
REPEATS = int(os.environ.get("SICH_REPEATS", 1 if QUICK else 2))
FOLDS = 5
SEED = 42
RATIO_Q = 0.99           # upper truncation quantile for the density ratio
NORMALIZE = False        # rescaling the ratio to mean 1 did not improve bias or coverage in simulation
B_BOOT = int(os.environ.get("SICH_BOOT", 30 if QUICK else 200))   # bootstrap resamples per policy
os.makedirs(OUT, exist_ok=True)
S = smc.S

# ---------------------------------------------------------------- exposures and policies
PROC_MEDIATORS = ["EVT_time", "TICI", "Number_of_passes", "Angioplasty", "Stenting", "Onset_to_successful_recanalization"]
EXPOSURES = {
    "SBP_at_admission": dict(
        label="Admission systolic BP", unit="mmHg", floor=140.0, grid=[5, 10, 15, 20, 30], primary=20,
        exclude={"SBP_more_than140_admission", "DBP_at_admission", "Thrombolysis"},   # coupled / lysis is decided on BP
        minimal=["Age", "History_of_hypertension", "NIHSS_at_admission", "ASPECTS_Score", "Onset_to_admission",
                 "Occlusion_site", "GCS_at_admission"],
        negative_control=None),
    "Blood_glucose_at_admission": dict(
        label="Admission glucose", unit="mmol/L", floor=7.8, grid=[1, 2, 3, 4], primary=2,
        exclude={"Thrombolysis"},                                                    # glucose is a lysis criterion
        minimal=["Age", "History_of_diabetes", "NIHSS_at_admission", "ASPECTS_Score", "Onset_to_admission",
                 "Occlusion_site"],
        negative_control=None),
    "Admission_to_groinpunture": dict(
        label="Door-to-groin time", unit="min", floor=60.0, grid=[15, 30, 45, 60], primary=30,
        exclude={"Onset_to_groinpunture"} | set(PROC_MEDIATORS),                      # composite and post-puncture mediators
        minimal=["Age", "NIHSS_at_admission", "ASPECTS_Score", "Onset_to_admission", "Thrombolysis",
                 "Occlusion_site", "Wakeupstroke"],
        negative_control=["EVT_time", "TICI", "Number_of_passes"]),
}


def shift(a, delta, floor):
    """MTP: lower values above the floor by delta, never below the floor; values at or below are untouched."""
    a = np.asarray(a, float)
    return np.where(a > floor, np.maximum(a - delta, floor), a)


# ---------------------------------------------------------------- design matrices and learners
def design(X, cols):
    parts = []
    for c in cols:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype(float)
            miss = v.isna()
            v = v.fillna(v.median())
            parts.append(((v - v.mean()) / (v.std() + 1e-9)).values[:, None])
            if miss.any():
                parts.append(miss.values.astype(float)[:, None])
        else:
            d = pd.get_dummies(s.astype(object).where(s.notna(), "missing"), prefix=c, drop_first=True, dtype=float)
            if d.shape[1]:
                parts.append(d.values)
    return np.hstack(parts) if parts else np.zeros((len(X), 0))


class AvgClf:
    """Average of an l2-logistic regression and a random forest (the nuisance learner used throughout)."""

    def __init__(self, kind="flexible", n_jobs=-1):
        self.kind, self.n_jobs = kind, n_jobs

    def fit(self, Z, y):
        base = [LogisticRegression(C=0.5, max_iter=5000)]
        if self.kind == "flexible":
            base.append(RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=self.n_jobs, random_state=SEED))
        self.ms = [clone(m).fit(Z, y) for m in base]
        return self

    def predict(self, Z):
        return np.clip(np.mean([m.predict_proba(Z)[:, 1] for m in self.ms], axis=0), 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------- doubly robust MTP estimator
def mtp_crossfit(A, W, Y, Ad, folds, kind="flexible", ratio_q=RATIO_Q, normalize=NORMALIZE, rf_jobs=-1):
    """Cross-fitted nuisances for one MTP. Returns out-of-fold m(A,W), m(d(A),W) and density ratio r(A,W)."""
    n = len(Y)
    a_mu, a_sd = A.mean(), A.std() + 1e-9
    za, zd = ((A - a_mu) / a_sd)[:, None], ((Ad - a_mu) / a_sd)[:, None]
    m_obs, m_shift, ratio = np.zeros(n), np.zeros(n), np.ones(n)
    changed = np.abs(Ad - A) > 1e-12
    for tr, te in folds:
        mo = AvgClf(kind, rf_jobs).fit(np.hstack([za[tr], W[tr]]), Y[tr])
        m_obs[te] = mo.predict(np.hstack([za[te], W[te]]))
        m_shift[te] = mo.predict(np.hstack([zd[te], W[te]]))
        if changed[tr].any():
            # density-ratio trick: label 1 = exposure drawn from the post-intervention distribution
            Zs = np.vstack([np.hstack([za[tr], W[tr]]), np.hstack([zd[tr], W[tr]])])
            lab = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
            rc = AvgClf(kind, rf_jobs).fit(Zs, lab)
            p = rc.predict(np.hstack([za[te], W[te]]))
            ratio[te] = p / (1 - p)
            if normalize:                                   # E[r] = 1 under the true ratio (Hajek-type stabilisation)
                ratio[te] = ratio[te] / ratio[te].mean()
    if ratio_q is not None:
        ratio = np.minimum(ratio, np.quantile(ratio, ratio_q))
    return m_obs, m_shift, ratio


def mtp_estimates(Y, m_obs, m_shift, ratio):
    """Per-patient influence-function contributions for the DR, g-computation and IPW estimators."""
    phi_dr = ratio * (Y - m_obs) + m_shift
    phi_g = m_shift
    w = ratio / ratio.mean()
    phi_ipw = w * Y
    return dict(dr=phi_dr, gcomp=phi_g, ipw=phi_ipw)


def contrast(phi, Y, nat_vec=None):
    """Policy risk vs natural course: risk difference, risk ratio, prevented fraction, with IF-based CIs.
    nat_vec replaces the observed outcome as the natural-course term (plug-in m(A,W) for g-computation)."""
    n = len(Y)
    Yn = Y if nat_vec is None else nat_vec
    psi, nat = phi.mean(), Yn.mean()
    d = phi - Yn
    se = d.std(ddof=1) / np.sqrt(n)
    rd = psi - nat
    se_l = (phi / max(psi, 1e-9) - Yn / max(nat, 1e-9)).std(ddof=1) / np.sqrt(n)
    rr = psi / max(nat, 1e-9)
    return dict(Risk_policy=psi, Risk_natural=nat, RD=rd, RD_lo=rd - 1.96 * se, RD_hi=rd + 1.96 * se, RD_se=se,
                p=2 * stats.norm.sf(abs(rd / se)) if se > 0 else np.nan,
                RR=rr, RR_lo=np.exp(np.log(rr) - 1.96 * se_l), RR_hi=np.exp(np.log(rr) + 1.96 * se_l),
                Prevented_fraction=-rd / max(nat, 1e-9))


def combine_repeats(rows):
    """Median-of-splits aggregation for repeated cross-fitting (Chernozhukov et al. 2018)."""
    df = pd.DataFrame(rows)
    out = {}
    for k in ["Risk_policy", "Risk_natural", "RD", "RR", "Prevented_fraction", "ESS", "Ratio_max", "Pct_shifted"]:
        if k in df:
            out[k] = float(df[k].median())
    se = np.sqrt(np.median(df.RD_se ** 2 + (df.RD - out["RD"]) ** 2))
    out.update(RD_se=se, RD_lo=out["RD"] - 1.96 * se, RD_hi=out["RD"] + 1.96 * se,
               p=2 * stats.norm.sf(abs(out["RD"] / se)) if se > 0 else np.nan,
               RR_lo=float(df.RR_lo.median()), RR_hi=float(df.RR_hi.median()))
    out["E_value"] = sp.evalue_rr(out["RR"])
    return out


# ---------------------------------------------------------------- nonparametric bootstrap
def _group_folds(Y, groups, seed):
    return list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(Y)), Y, groups))


def _boot_mtp_one(b, A, W, Y, Ad, kind, ratio_q):
    rng = np.random.default_rng(SEED + 1000 + b)
    idx = rng.integers(0, len(Y), len(Y))
    folds = _group_folds(Y[idx], idx, SEED + b)          # resampled copies of a patient stay in one fold
    mo, ms, ra = mtp_crossfit(A[idx], W[idx], Y[idx], Ad[idx], folds, kind, ratio_q, NORMALIZE, rf_jobs=1)
    return float((ra * (Y[idx] - mo) + ms).mean() - Y[idx].mean())


def _boot_lysis_one(b, T, W, Y, kind):
    rng = np.random.default_rng(SEED + 2000 + b)
    idx = rng.integers(0, len(Y), len(Y))
    folds = _group_folds(Y[idx] + 2 * T[idx], idx, SEED + b)
    psi1, psi0 = aipw_once(T[idx], W[idx], Y[idx], folds, kind, rf_jobs=1)
    yb = Y[idx]
    return float(psi1.mean() - yb.mean()), float(psi0.mean() - yb.mean()), float(psi1.mean() - psi0.mean())


def boot_summary(draws, est):
    d = np.asarray(draws, float)
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return dict(Boot_lo=float(lo), Boot_hi=float(hi), Boot_se=float(d.std(ddof=1)), Boot_p=float(p), Boot_B=len(d))


# ---------------------------------------------------------------- adjustment sets
def tier_adjustment(X, T, spec, graph):
    t_tier = sp.get_tier(T)
    desc = sp.descendants(graph, T) if T in graph["names"] else set()
    desc = desc | ({"TICI"} if "TICI_ordinal" in desc else set())
    cols = []
    for c in X.columns:
        if c == T or c in spec["exclude"] or c in desc:
            continue
        if sp.get_tier(c) <= t_tier:
            cols.append(c)
    return cols


def graph_adjustment(X, T, cols, graph):
    if T not in graph["names"]:
        return cols
    anc = sp.possible_ancestors(graph, [T, S])
    desc = sp.descendants(graph, T)
    return [c for c in cols if (c in anc or (c == "TICI" and "TICI_ordinal" in anc)) and c not in desc]


# ---------------------------------------------------------------- analyses
def run_mtp(X, Y, T, spec, adj, folds_by_rep, delta, kind="flexible", estimator="dr", ratio_q=RATIO_Q, boot=False):
    A_all = pd.to_numeric(X[T], errors="coerce").values
    keep = ~np.isnan(A_all)
    A, Yk = A_all[keep], Y[keep]
    W = design(X.loc[keep].reset_index(drop=True), adj)
    Ad = shift(A, delta, spec["floor"])
    idx = np.where(keep)[0]
    pos = {g: i for i, g in enumerate(idx)}
    rows, per_patient = [], []
    for folds in folds_by_rep:
        f2 = [(np.array([pos[i] for i in tr if i in pos]), np.array([pos[i] for i in te if i in pos])) for tr, te in folds]
        m_obs, m_shift, ratio = mtp_crossfit(A, W, Yk, Ad, f2, kind, ratio_q)
        phi = mtp_estimates(Yk, m_obs, m_shift, ratio)[estimator]
        c = contrast(phi, Yk, m_obs if estimator == "gcomp" else None)
        c.update(ESS=ratio.sum() ** 2 / (ratio ** 2).sum(), Ratio_max=float(ratio.max()),
                 Pct_shifted=float((np.abs(Ad - A) > 1e-12).mean()))
        rows.append(c)
        per_patient.append(dict(m_obs=m_obs, m_shift=m_shift, phi_dr=mtp_estimates(Yk, m_obs, m_shift, ratio)["dr"]))
    res = combine_repeats(rows)
    res["N"] = int(keep.sum())
    if boot:
        draws = Parallel(n_jobs=-1)(delayed(_boot_mtp_one)(b, A, W, Yk, Ad, kind, ratio_q) for b in range(B_BOOT))
        res.update(boot_summary(draws, res["RD"]))
    return res, per_patient, idx, A, Ad


def aipw_once(T, W, Y, folds, kind="flexible", rf_jobs=-1):
    e, m0, m1 = np.zeros(len(Y)), np.zeros(len(Y)), np.zeros(len(Y))
    for tr, te in folds:
        e[te] = AvgClf(kind, rf_jobs).fit(W[tr], T[tr]).predict(W[te])
        for arm, arr in ((0, m0), (1, m1)):
            it = tr[T[tr] == arm]
            arr[te] = AvgClf(kind, rf_jobs).fit(W[it], Y[it]).predict(W[te])
    e = np.clip(e, 0.02, 0.98)
    return m1 + T * (Y - m1) / e, m0 + (1 - T) * (Y - m0) / (1 - e)


def run_lysis(X, Y, adj, folds_by_rep, kind="flexible", boot=False):
    T = (X["Thrombolysis"].astype(str) == "1").astype(int).values
    W = design(X, adj)
    out = {"all treated": [], "none treated": [], "treated vs untreated": []}
    for folds in folds_by_rep:
        psi1, psi0 = aipw_once(T, W, Y, folds, kind)
        for name, phi in (("all treated", psi1), ("none treated", psi0)):
            c = contrast(phi, Y); c.update(ESS=np.nan, Ratio_max=np.nan, Pct_shifted=np.nan)
            out[name].append(c)
        n = len(Y); d = psi1 - psi0; se = d.std(ddof=1) / np.sqrt(n)
        rr = psi1.mean() / psi0.mean()
        out["treated vs untreated"].append(dict(Risk_policy=psi1.mean(), Risk_natural=psi0.mean(), RD=d.mean(), RD_se=se,
                                                RR=rr, RR_lo=np.nan, RR_hi=np.nan, Prevented_fraction=np.nan))
    res = {k: combine_repeats(v) for k, v in out.items()}
    if boot:
        draws = np.array(Parallel(n_jobs=-1)(delayed(_boot_lysis_one)(b, T, W, Y, kind) for b in range(B_BOOT)))
        for j, k in enumerate(["all treated", "none treated", "treated vs untreated"]):
            res[k].update(boot_summary(draws[:, j], res[k]["RD"]))
    return res


def risk_strata(Y, per_patient, idx, nbins=3):
    """DR policy effect within tertiles of out-of-fold baseline predicted risk; individual predicted changes."""
    m_obs = np.mean([p["m_obs"] for p in per_patient], axis=0)
    m_shift = np.mean([p["m_shift"] for p in per_patient], axis=0)
    phi = np.mean([p["phi_dr"] for p in per_patient], axis=0)
    Yk = Y[idx]
    q = pd.qcut(m_obs, nbins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(set(q)):
        s = q == b
        d = phi[s] - Yk[s]
        rows.append(dict(Tertile=int(b) + 1, N=int(s.sum()), Baseline_risk=float(m_obs[s].mean()),
                         Observed_risk=float(Yk[s].mean()), DR_RD=float(d.mean()),
                         DR_lo=float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(s.sum())),
                         DR_hi=float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(s.sum())),
                         Predicted_change=float((m_shift[s] - m_obs[s]).mean())))
    indiv = pd.DataFrame(dict(patient_row=idx, baseline_risk=m_obs, counterfactual_risk=m_shift, change=m_shift - m_obs))
    return pd.DataFrame(rows), indiv


def structural_support(Xb, ys, yh, num_b, cat_b, folds, cache):
    """Fold-wise discovery: share of training folds in which each exposure is a possible ancestor / neighbour of sICH."""
    names = list(EXPOSURES) + ["Thrombolysis"]
    counts = {e: dict(ancestor=0, adjacent=0, present=0) for e in names}
    for j, (tr, _) in enumerate(folds):
        key = j
        if key not in cache:
            cache[key] = smc.discover2(Xb.iloc[tr].reset_index(drop=True), ys.iloc[tr].reset_index(drop=True),
                                       yh.iloc[tr].reset_index(drop=True), num_b, cat_b)
        g = cache[key]["g"]
        anc = sp.possible_ancestors(g, [S])
        for e in names:
            if e in cache[key]["names"]:
                counts[e]["present"] += 1
                counts[e]["ancestor"] += int(e in anc)
                counts[e]["adjacent"] += int(any(set(x) == {e, S} for x in list(g["dir"]) + list(g["und"])))
        print(f"  discovery fold {j + 1}/{len(folds)} done", flush=True)
    m = len(folds)
    return pd.DataFrame([dict(Exposure=e, Ancestor_of_sICH=c["ancestor"] / m, Adjacent_to_sICH=c["adjacent"] / m,
                              Retained_after_screening=c["present"] / m) for e, c in counts.items()])


def main():
    t0 = time.time()
    X, ys, yh = smc.load()
    Xf, num, cat, new, Xb, num_b, cat_b = smc.prepare(X)
    Y = ys.values.astype(int)
    folds = list(RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
                 .split(np.zeros(len(Y)), (ys + yh).values))
    folds_by_rep = [folds[r * FOLDS:(r + 1) * FOLDS] for r in range(REPEATS)]
    pd.DataFrame([dict(patient_row=int(i), repeat=j // FOLDS, fold=j % FOLDS) for j, (_, te) in enumerate(folds) for i in te]) \
        .to_csv(os.path.join(OUT, "fold_assignment.csv"), index=False)

    sp.divider("CAUSAL DISCOVERY (tier-constrained; reperfusion grade in the procedure tier)")
    full = smc.discover2(Xb, ys, yh, num_b, cat_b)
    graph = dict(dir=[list(e) for e in full["g"]["dir"]], und=[list(e) for e in full["g"]["und"]], names=full["names"],
                 pa_s=full["pa_s"], pa_h=full["pa_h"], causal=full["causal"],
                 tiers={n: sp.get_tier(n) for n in full["names"]})
    json.dump(graph, open(os.path.join(OUT, "full_graph.json"), "w"), indent=1)
    g_full = dict(dir=[tuple(e) for e in graph["dir"]], und=[tuple(e) for e in graph["und"]], names=graph["names"])
    print(f"  parents of sICH: {full['pa_s']}\n  parents of HT  : {full['pa_h']}")
    cache = {}
    support = structural_support(Xb, ys, yh, num_b, cat_b, folds, cache)
    support.round(3).to_csv(os.path.join(OUT, "cf_structural_support.csv"), index=False)
    print(support.round(2).to_string(index=False))

    main_rows, dose_rows, abl_rows, strata_all, indiv_all, adj_log = [], [], [], [], [], {}
    for T, spec in EXPOSURES.items():
        sp.divider(f"MODIFIED TREATMENT POLICY — {spec['label']} ({T})")
        adj = tier_adjustment(X, T, spec, g_full)
        adj_graph = graph_adjustment(X, T, adj, g_full)
        adj_min = [c for c in spec["minimal"] if c in X.columns]
        adj_log[T] = dict(tier=adj, graph=adj_graph, minimal=adj_min)
        print(f"  adjustment: tier-based {len(adj)}, graph-based {len(adj_graph)}, minimal {len(adj_min)}")
        for delta in spec["grid"]:
            res, pp, idx, A, Ad = run_mtp(X, Y, T, spec, adj, folds_by_rep, delta, boot=True)
            dose_rows.append(dict(Exposure=T, Label=spec["label"], Delta=delta, Floor=spec["floor"], Unit=spec["unit"], **res))
            print(f"  shift −{delta} {spec['unit']} (floor {spec['floor']:g}): RD {100 * res['RD']:+.2f} pp "
                  f"IF [{100 * res['RD_lo']:+.2f}, {100 * res['RD_hi']:+.2f}] | bootstrap [{100 * res['Boot_lo']:+.2f}, {100 * res['Boot_hi']:+.2f}], "
                  f"shifted {res['Pct_shifted']:.0%}, ESS {res['ESS']:.0f}", flush=True)
            if delta == spec["primary"]:
                main_rows.append(dict(Exposure=T, Label=spec["label"],
                                      Policy=f"−{delta} {spec['unit']}, not below {spec['floor']:g}", **res))
                st, ind = risk_strata(Y, pp, idx)
                st.insert(0, "Exposure", T); strata_all.append(st)
                ind.insert(0, "Exposure", T); ind["observed"] = A; ind["shifted"] = Ad; indiv_all.append(ind)
        d = spec["primary"]
        configs = [("PRIMARY: DR, tier-based set, flexible, ratio truncated", dict(adj=adj)),
                   ("− outcome model (IPW only)", dict(adj=adj, estimator="ipw")),
                   ("− density ratio (g-computation only)", dict(adj=adj, estimator="gcomp")),
                   ("linear instead of flexible nuisances", dict(adj=adj, kind="linear")),
                   ("graph-based adjustment set", dict(adj=adj_graph)),
                   ("minimal clinical adjustment set", dict(adj=adj_min)),
                   ("no ratio truncation", dict(adj=adj, ratio_q=None))]
        if spec["negative_control"]:
            configs.append(("+ mediators adjusted (INAPPROPRIATE, negative control)",
                            dict(adj=adj + [c for c in spec["negative_control"] if c in X.columns])))
        for lab, kw in configs:
            res, *_ = run_mtp(X, Y, T, spec, kw.pop("adj"), folds_by_rep, d, **kw)
            abl_rows.append(dict(Exposure=T, Configuration=lab, **res))

    sp.divider("STATIC INTERVENTIONS — bridging thrombolysis (AIPW)")
    adj_lys = [c for c in X.columns if c != "Thrombolysis" and sp.get_tier(c) <= sp.TIER_ADMISSION
               and c not in {"SBP_more_than140_admission"}]
    adj_log["Thrombolysis"] = dict(tier=adj_lys)
    lys = run_lysis(X, Y, adj_lys, folds_by_rep, boot=True)
    for k, v in lys.items():
        main_rows.append(dict(Exposure="Thrombolysis", Label="Bridging thrombolysis", Policy=k, N=len(Y), **v))
        print(f"  {k:22s}: RD {100 * v['RD']:+.2f} pp IF [{100 * v['RD_lo']:+.2f}, {100 * v['RD_hi']:+.2f}] | "
              f"bootstrap [{100 * v['Boot_lo']:+.2f}, {100 * v['Boot_hi']:+.2f}]")
    lys_lin = run_lysis(X, Y, adj_lys, folds_by_rep, kind="linear")
    abl_rows.append(dict(Exposure="Thrombolysis", Configuration="PRIMARY: AIPW, flexible (treated vs untreated)",
                         **lys["treated vs untreated"]))
    abl_rows.append(dict(Exposure="Thrombolysis", Configuration="linear instead of flexible nuisances",
                         **lys_lin["treated vs untreated"]))

    pd.DataFrame(main_rows).to_csv(os.path.join(OUT, "cf_main_policies.csv"), index=False)
    pd.DataFrame(dose_rows).to_csv(os.path.join(OUT, "cf_dose_response.csv"), index=False)
    pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "cf_ablation.csv"), index=False)
    pd.concat(strata_all).to_csv(os.path.join(OUT, "cf_risk_strata.csv"), index=False)
    pd.concat(indiv_all).to_csv(os.path.join(OUT, "cf_individual_changes.csv"), index=False)
    json.dump(adj_log, open(os.path.join(OUT, "cf_adjustment_sets.json"), "w"), indent=1)

    sp.divider("SUMMARY")
    lines = [f"Design: {REPEATS}x{FOLDS}-fold repeated cross-fitting, seed {SEED}; n = {len(Y)}, sICH = {Y.sum()}; "
             f"{B_BOOT} bootstrap resamples (primary inference); IF-based intervals under-covered in simulation"]
    for r in main_rows:
        lines.append(f"{r['Label']}: {r['Policy']} -> risk {100 * r['Risk_policy']:.1f}% vs {100 * r['Risk_natural']:.1f}%, "
                     f"RD {100 * r['RD']:+.2f} pp, bootstrap 95% CI [{100 * r['Boot_lo']:+.2f}, {100 * r['Boot_hi']:+.2f}], "
                     f"bootstrap p={r['Boot_p']:.3f} (IF-based [{100 * r['RD_lo']:+.2f}, {100 * r['RD_hi']:+.2f}])")
    open(os.path.join(OUT, "cf_key_numbers.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
