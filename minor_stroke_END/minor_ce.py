"""Machine-learning counterfactual explanations for END after minor stroke, with a causal check.

1. Risk model: END from baseline and admission variables (no in-hospital treatment, nothing after END).
   L2-logistic regression and LightGBM, repeated stratified 5-fold CV (AUC, AUPRC, Brier, calibration).
2. Counterfactual explanations (Wachter et al. 2017) for every out-of-fold high-risk patient
   (predicted risk >= Youden threshold): the smallest MAD-weighted L1 decrease of the actionable
   admission values (systolic BP not below 140 mmHg, glucose not below 7.8 mmol/L) that brings the
   model's risk below the threshold, all other variables held fixed. The action space is 2-D, so the
   optimum is found exactly by exhaustive search (1 mmHg x 0.1 mmol/L grid) instead of a stochastic
   search (DiCE's genetic search took > 5 min per patient with 93 features).
3. Causal check: the explanations define a patient-specific treatment policy ("give each flagged patient
   the change the model asks for"). Its effect on END risk is estimated with the same cross-fitted
   doubly robust MTP estimator as minor_stroke_counterfactual.py (2-D exposure: SBP and glucose),
   with a grouped bootstrap of the estimator (explanations held fixed), and compared with the risk
   reduction the model itself promises.

Run:  python minor_ce.py            (SICH_BOOT=30 for a quick run)
"""
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import minor_stroke_counterfactual as ms   # noqa: E402  (sets up the sich_counterfactual import path)

cf = ms.cf
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
from joblib import Parallel, delayed         # noqa: E402
from scipy.special import logit              # noqa: E402
from sklearn.compose import ColumnTransformer                    # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve   # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold   # noqa: E402
from sklearn.pipeline import Pipeline                            # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from lightgbm import LGBMClassifier                              # noqa: E402

OUT = os.path.join(HERE, "outputs_ce")
os.makedirs(OUT, exist_ok=True)
SBP, GLU = "admissionSBP", "Blood glucose on admission"
ACTION = {SBP: 140.0, GLU: 7.8}            # actionable features and their clinical floors
STEP = {SBP: 1.0, GLU: 0.1}                # search resolution
SEED = cf.SEED


# ---------------------------------------------------------------- data and models
def prep_fit(Xtr, num, cat):
    """Training-fold imputation: numeric median, categorical 'missing' level (DiCE needs complete rows)."""
    med = Xtr[num].median()
    def f(Z):
        Z = Z[num + cat].copy()
        Z[num] = Z[num].astype(float).fillna(med)
        for c in cat:
            Z[c] = Z[c].astype(object).where(Z[c].notna(), "missing").astype(str)
        return Z
    return f


def make_model(kind, num, cat):
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
    if kind == "logistic":
        ct = ColumnTransformer([("num", StandardScaler(), num), ("cat", oh, cat)])
        clf = LogisticRegression(C=0.05, max_iter=5000)
    else:
        ct = ColumnTransformer([("num", "passthrough", num), ("cat", oh, cat)])
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=25, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, random_state=SEED, verbose=-1, n_jobs=4)
    return Pipeline([("ct", ct), ("clf", clf)])


def calib(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    slope = float(lr.coef_[0, 0])
    citl = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))
    return slope, citl


# ---------------------------------------------------------------- 2-D doubly robust MTP (patient-specific shift)
def mtp_crossfit_2d(A, W, Y, Ad, folds, ratio_q=cf.RATIO_Q, rf_jobs=-1):
    n = len(Y)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    za, zd = (A - mu) / sd, (Ad - mu) / sd
    m_obs, m_shift, ratio = np.zeros(n), np.zeros(n), np.ones(n)
    for tr, te in folds:
        mo = cf.AvgClf("flexible", rf_jobs).fit(np.hstack([za[tr], W[tr]]), Y[tr])
        m_obs[te] = mo.predict(np.hstack([za[te], W[te]]))
        m_shift[te] = mo.predict(np.hstack([zd[te], W[te]]))
        Zs = np.vstack([np.hstack([za[tr], W[tr]]), np.hstack([zd[tr], W[tr]])])
        lab = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
        p = cf.AvgClf("flexible", rf_jobs).fit(Zs, lab).predict(np.hstack([za[te], W[te]]))
        ratio[te] = p / (1 - p)
    if ratio_q is not None:
        ratio = np.minimum(ratio, np.quantile(ratio, ratio_q))
    return m_obs, m_shift, ratio


def boot_2d(b, A, W, Y, Ad):
    rng = np.random.default_rng(SEED + 3000 + b)
    idx = rng.integers(0, len(Y), len(Y))
    folds = cf._group_folds(Y[idx], idx, SEED + b)
    mo, msh, ra = mtp_crossfit_2d(A[idx], W[idx], Y[idx], Ad[idx], folds, rf_jobs=1)
    return float((ra * (Y[idx] - mo) + msh).mean() - Y[idx].mean())


def causal_policy(A, W, Y, Ad, folds_by_rep, label):
    rows, ind = [], []
    for folds in folds_by_rep:
        mo, msh, ra = mtp_crossfit_2d(A, W, Y, Ad, folds)
        c = cf.contrast(ra * (Y - mo) + msh, Y)
        g = cf.contrast(msh, Y, mo)
        c.update(ESS=ra.sum() ** 2 / (ra ** 2).sum(), Ratio_max=float(ra.max()),
                 Pct_shifted=float((np.abs(Ad - A).sum(1) > 1e-9).mean()))
        c["Gcomp_RD"] = g["RD"]
        rows.append(c); ind.append(msh - mo)
    res = cf.combine_repeats(rows)
    res["Gcomp_RD"] = float(np.median([r["Gcomp_RD"] for r in rows]))
    draws = Parallel(n_jobs=-1)(delayed(boot_2d)(b, A, W, Y, Ad) for b in range(cf.B_BOOT))
    res.update(cf.boot_summary(draws, res["RD"]))
    res["Policy"] = label
    print(f"  {label:28s}: DR {100 * res['RD']:+.2f} pp, bootstrap [{100 * res['Boot_lo']:+.2f}, {100 * res['Boot_hi']:+.2f}]"
          f" | g-comp {100 * res['Gcomp_RD']:+.2f} | ESS {res['ESS']:.0f} | max ratio {res['Ratio_max']:.1f}")
    return res, np.mean(ind, axis=0)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    X, Y = ms.load()
    feats = [c for c in X.columns if ms.tier(c) <= 2 and c != "admissionDBP"]   # DBP moves with SBP: left out
    cat = [c for c in feats if c in ms.CATEGORICAL]
    num = [c for c in feats if c not in ms.CATEGORICAL]
    print(f"n = {len(Y)}, END = {Y.sum()}; model features = {len(feats)} ({len(num)} numeric, {len(cat)} categorical)")

    # 1 -------- risk model, repeated CV
    rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cf.REPEATS, random_state=SEED).split(X, Y))
    perf, oof = [], {}
    for kind in ["logistic", "lightgbm"]:
        oof[kind] = np.zeros((cf.REPEATS, len(Y)))
        for k, (tr, te) in enumerate(rskf):
            f = prep_fit(X.iloc[tr], num, cat)
            m = make_model(kind, num, cat).fit(f(X.iloc[tr]), Y[tr])
            oof[kind][k // 5, te] = m.predict_proba(f(X.iloc[te]))[:, 1]
        for r in range(cf.REPEATS):
            p = oof[kind][r]
            s, citl = calib(Y, p)
            perf.append(dict(Model=kind, Repeat=r, AUC=roc_auc_score(Y, p), AUPRC=average_precision_score(Y, p),
                             Brier=brier_score_loss(Y, p), Cal_slope=s, Cal_in_large=citl))
    perf = pd.DataFrame(perf)
    summ = perf.groupby("Model")[["AUC", "AUPRC", "Brier", "Cal_slope", "Cal_in_large"]].mean()
    summ.to_csv(os.path.join(OUT, "ce_model_performance.csv"))
    print(summ.round(3).to_string())
    best = summ.AUC.idxmax()
    p_oof = oof[best][0]
    fpr, tpr, thr = roc_curve(Y, p_oof)
    t_star = float(thr[np.argmax(tpr - fpr)])
    print(f"CE model: {best}; Youden threshold {t_star:.3f}")

    # 2 -------- counterfactual explanations, out-of-fold
    mad = {a: float(np.median(np.abs(X[a] - X[a].median()))) for a in ACTION}
    ce_rows = []
    folds5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, Y))
    for k, (tr, te) in enumerate(folds5):
        f = prep_fit(X.iloc[tr], num, cat)
        Xtr, Xte = f(X.iloc[tr]), f(X.iloc[te])
        pipe = make_model(best, num, cat).fit(Xtr, Y[tr])
        p_te = pipe.predict_proba(Xte)[:, 1]
        for j in np.where(p_te >= t_star)[0]:
            i = te[j]
            row = dict(idx=int(i), fold=k, END=int(Y[i]), p_orig=float(p_te[j]), SBP=float(X.at[i, SBP]), GLU=float(X.at[i, GLU]),
                       SBP_cf=float(X.at[i, SBP]), GLU_cf=float(X.at[i, GLU]), p_cf=float(p_te[j]), status="")
            if not any(X.at[i, a] > ACTION[a] + 1e-9 for a in ACTION):
                row["status"] = "not actionable"; ce_rows.append(row); continue
            # exhaustive search over the 2-D actionable space (decreases only, never below the floors)
            g = {a: (np.arange(ACTION[a], X.at[i, a], STEP[a]).tolist() + [float(X.at[i, a])]) if X.at[i, a] > ACTION[a] else [float(X.at[i, a])]
                 for a in ACTION}
            gs, gg = np.meshgrid(g[SBP], g[GLU], indexing="ij")
            cand = pd.concat([Xte.iloc[[j]]] * gs.size, ignore_index=True)
            cand[SBP], cand[GLU] = gs.ravel(), gg.ravel()
            pc = pipe.predict_proba(cand)[:, 1]
            cost = np.abs(gs.ravel() - X.at[i, SBP]) / mad[SBP] + np.abs(gg.ravel() - X.at[i, GLU]) / mad[GLU]
            ok = pc < t_star
            if ok.any():
                b = np.lexsort((pc, cost))  # minimum cost, then lowest risk
                b = b[ok[b]][0]
                row.update(SBP_cf=float(gs.ravel()[b]), GLU_cf=float(gg.ravel()[b]), p_cf=float(pc[b]), status="valid")
            else:
                row.update(p_cf=float(pc.min()), status="no counterfactual")
            ce_rows.append(row)
        print(f"  fold {k + 1}: flagged {int((p_te >= t_star).sum())}, cumulative valid "
              f"{sum(r['status'] == 'valid' for r in ce_rows)} ({time.time() - t0:.0f}s)")
    ce = pd.DataFrame(ce_rows)
    ce["dSBP"], ce["dGLU"] = ce.SBP_cf - ce.SBP, ce.GLU_cf - ce.GLU
    ce["model_change"] = ce.p_cf - ce.p_orig
    ce.to_csv(os.path.join(OUT, "ce_individual.csv"), index=False)       # patient-level: do not publish

    # 3 -------- causal check of the explanation policy
    A = X[[SBP, GLU]].astype(float).values
    Ad = A.copy()
    v = ce[ce.status == "valid"]
    Ad[v.idx.values, 0], Ad[v.idx.values, 1] = v.SBP_cf.values, v.GLU_cf.values
    adj = [c for c in ms.pre_exposure(X, SBP, exclude={"admissionDBP", GLU})]
    W = cf.design(X, adj)
    folds = list(RepeatedStratifiedKFold(n_splits=cf.FOLDS, n_repeats=cf.REPEATS, random_state=SEED).split(np.zeros(len(Y)), Y))
    fbr = [folds[r * cf.FOLDS:(r + 1) * cf.FOLDS] for r in range(cf.REPEATS)]
    print("\nCAUSAL CHECK of the counterfactual-explanation policy (whole cohort, n = %d)" % len(Y))
    Ad_s, Ad_g = A.copy(), A.copy()
    Ad_s[:, 0], Ad_g[:, 1] = Ad[:, 0], Ad[:, 1]
    pol = []
    ind_full = None
    for lab, AD in [("explanation policy (both)", Ad), ("SBP part only", Ad_s), ("glucose part only", Ad_g)]:
        res, ind = causal_policy(A, W, Y, AD, fbr, lab)
        pol.append(res)
        if ind_full is None:
            ind_full = ind
    pol = pd.DataFrame(pol)
    n, nv = len(Y), len(v)
    model_pop = float(v.model_change.sum() / n)
    pol["Model_implied_RD"] = [model_pop, np.nan, np.nan]
    pol.to_csv(os.path.join(OUT, "ce_causal_check.csv"), index=False)
    ce["causal_change"] = np.nan
    ce.loc[ce.status == "valid", "causal_change"] = ind_full[v.idx.values]
    ce.to_csv(os.path.join(OUT, "ce_individual.csv"), index=False)

    # summary (aggregate only)
    st = ce.status.value_counts().to_dict()
    vv = ce[ce.status == "valid"]
    S = dict(model=best, threshold=t_star, n=int(n), n_flagged=int(len(ce)), flagged_END=int(ce.END.sum()), status=st,
             n_valid=int(nv), pct_valid=float(nv / len(ce)),
             sbp_changed=int((vv.dSBP < -1e-6).sum()), glu_changed=int((vv.dGLU < -1e-6).sum()),
             both_changed=int(((vv.dSBP < -1e-6) & (vv.dGLU < -1e-6)).sum()),
             dSBP_median=float(vv.loc[vv.dSBP < -1e-6, "dSBP"].median()), dSBP_iqr=[float(x) for x in vv.loc[vv.dSBP < -1e-6, "dSBP"].quantile([.25, .75])],
             dGLU_median=float(vv.loc[vv.dGLU < -1e-6, "dGLU"].median()), dGLU_iqr=[float(x) for x in vv.loc[vv.dGLU < -1e-6, "dGLU"].quantile([.25, .75])],
             p_orig_mean=float(vv.p_orig.mean()), p_cf_mean=float(vv.p_cf.mean()),
             model_change_recipients=float(vv.model_change.mean()), causal_change_recipients=float(vv.causal_change.mean()),
             model_implied_pop=model_pop,
             corr_model_causal=float(np.corrcoef(vv.model_change, vv.causal_change)[0, 1]) if nv > 2 else None,
             share_causal_reduction=float((vv.causal_change < 0).mean()))
    for r in pol.itertuples():
        S[f"DR_{r.Policy}"] = dict(RD=r.RD, Boot_lo=r.Boot_lo, Boot_hi=r.Boot_hi, IF_lo=r.RD_lo, IF_hi=r.RD_hi, Gcomp=r.Gcomp_RD,
                                    ESS=r.ESS, Ratio_max=r.Ratio_max, Pct_shifted=r.Pct_shifted, Per_recipient=r.RD * n / max(nv, 1))
    json.dump(S, open(os.path.join(OUT, "ce_summary.json"), "w"), indent=1, default=float)
    # aggregate distribution for the figure (binned, no individual rows)
    print(json.dumps(S, indent=1, default=float))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
