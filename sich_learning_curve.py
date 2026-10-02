import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed

import sich_prediction_v2 as v2

FILE_PATH   = os.environ.get("SICH_DATA", "dataset.xlsx")
OUT_DIR     = os.environ.get("SICH_OUT", "outputs_lc")
QUICK       = os.environ.get("SICH_QUICK", "0") == "1"
ENDPOINTS   = [("sICH", "1", "2"), ("Hemorrhagic_transformation", "1", "2")]
FRACTIONS   = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
REPEATS     = 1 if QUICK else 3
FOLDS       = 5
SEED        = 42
N_JOBS      = -1
PROJECT_N   = [470, 750, 1000, 1500, 2000]
os.makedirs(OUT_DIR, exist_ok=True)


def models(num, cat, spw):
    lr_pre = v2.make_pre(num, cat, True)
    lr_pre.set_params(num__sc=QuantileTransformer(output_distribution="normal", n_quantiles=100))
    return {
        "ElasticNet-LR": Pipeline([("pre", lr_pre),
                                   ("clf", v2.enet_lr(C=0.1, class_weight="balanced", random_state=SEED))]),
        "LightGBM": Pipeline([("pre", v2.make_pre(num, cat, False)),
                              ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=4,
                                                     min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                     colsample_bytree=0.5, reg_lambda=5.0,
                                                     scale_pos_weight=float(np.sqrt(spw)), n_jobs=1,
                                                     verbose=-1, random_state=SEED))]),
        "RandomForest": Pipeline([("pre", v2.make_pre(num, cat, False)),
                                  ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                                 max_features=0.35, class_weight="balanced_subsample",
                                                                 n_jobs=1, random_state=SEED))]),
    }


def one_task(X, y, tr, te, frac, fold, num, cat, spw):
    rng = np.random.default_rng(SEED + fold * 100 + int(frac * 100))
    ytr = y.iloc[tr].values
    pos, neg = tr[ytr == 1], tr[ytr == 0]
    sub = np.concatenate([rng.choice(pos, max(4, int(round(len(pos) * frac))), replace=False),
                          rng.choice(neg, max(4, int(round(len(neg) * frac))), replace=False)])
    Xs, ys = X.iloc[sub], y.iloc[sub]
    Xt, yt = X.iloc[te], y.iloc[te].values
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED + fold)
    rows, oofs, ps, trains = [], [], [], []
    for name, pipe in models(num, cat, spw).items():
        est = clone(pipe)
        oof = cross_val_predict(est, Xs, ys, cv=inner, method="predict_proba")[:, 1]
        thr, _ = v2.best_threshold(ys.values, oof)
        est.fit(Xs, ys)
        p = est.predict_proba(Xt)[:, 1]
        ptr = est.predict_proba(Xs)[:, 1]
        oofs.append(oof); ps.append(p); trains.append(ptr)
        rows.append(dict(Model=name, Fraction=frac, N_train=len(sub), Fold=fold,
                         AUC=roc_auc_score(yt, p), PR_AUC=average_precision_score(yt, p),
                         F1=f1_score(yt, (p >= thr).astype(int), zero_division=0),
                         Train_AUC=roc_auc_score(ys, ptr)))
    oof_a, p_a, tr_a = np.mean(oofs, 0), np.mean(ps, 0), np.mean(trains, 0)
    thr, _ = v2.best_threshold(ys.values, oof_a)
    rows.append(dict(Model="Average(3)", Fraction=frac, N_train=len(sub), Fold=fold,
                     AUC=roc_auc_score(yt, p_a), PR_AUC=average_precision_score(yt, p_a),
                     F1=f1_score(yt, (p_a >= thr).astype(int), zero_division=0),
                     Train_AUC=roc_auc_score(ys, tr_a)))
    return rows


def power_law(n, a, b, c):
    return a - b * np.power(n, -c)


def project(ns, vals, upper):
    try:
        popt, _ = curve_fit(power_law, ns, vals, p0=[min(upper, vals[-1] + 0.05), 1.0, 0.5],
                            bounds=([0, 0, 0.05], [upper, 100, 2.0]), maxfev=20000)
        return popt
    except Exception:
        return None


def run_endpoint(outcome, pos, neg):
    v2.OUTCOME, v2.POS_CODE, v2.NEG_CODE = outcome, pos, neg
    df = v2.load_data(FILE_PATH)
    y = df["_y"].astype(int).reset_index(drop=True)
    X = df.drop(columns=["_y"] + [c for c in v2.OUTCOME_COLS if c in df.columns]).reset_index(drop=True)
    X, _ = v2.add_clinical_features(X)
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    spw = float((y == 0).sum() / max(y.sum(), 1))
    folds = list(RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED).split(X, y))
    res = Parallel(n_jobs=N_JOBS)(delayed(one_task)(X, y, tr, te, f, j, num, cat, spw)
                                  for j, (tr, te) in enumerate(folds) for f in FRACTIONS)
    d = pd.DataFrame([r for rr in res for r in rr])
    d["Endpoint"] = outcome
    return d, y.mean(), len(y)


def main():
    allres, info = [], {}
    for outcome, pos, neg in ENDPOINTS:
        print(f"\nLearning curve: {outcome}")
        d, prev, n = run_endpoint(outcome, pos, neg)
        allres.append(d)
        info[outcome] = (prev, n)
    res = pd.concat(allres)
    res.to_csv(os.path.join(OUT_DIR, "lc_raw.csv"), index=False)
    agg = (res.groupby(["Endpoint", "Model", "Fraction"])
              .agg(N_train=("N_train", "mean"), AUC=("AUC", "mean"), AUC_SD=("AUC", "std"),
                   PR_AUC=("PR_AUC", "mean"), F1=("F1", "mean"), F1_SD=("F1", "std"),
                   Train_AUC=("Train_AUC", "mean")).reset_index())
    agg.to_csv(os.path.join(OUT_DIR, "lc_summary.csv"), index=False)

    proj_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for r, (outcome, _, _) in enumerate(ENDPOINTS):
        prev, n_total = info[outcome]
        a = agg[agg.Endpoint == outcome]
        full_n = n_total
        for k, (metric, ax) in enumerate(zip(["AUC", "F1", "PR_AUC"], axes[r])):
            for i, m in enumerate(a.Model.unique()):
                s = a[a.Model == m]
                ax.plot(s.N_train, s[metric], "o-", color=v2.PAL[i], lw=2.5 if m == "Average(3)" else 1.2, label=m)
                if metric in ("AUC", "F1"):
                    sd = s[f"{metric}_SD"] / np.sqrt(FOLDS * REPEATS)
                    ax.fill_between(s.N_train, s[metric] - 1.96 * sd, s[metric] + 1.96 * sd,
                                    color=v2.PAL[i], alpha=0.08)
            if metric == "AUC":
                s = a[a.Model == "Average(3)"]
                ax.plot(s.N_train, s["Train_AUC"], "--", color="grey", label="Average(3) train AUC")
            s = a[a.Model == "Average(3)"]
            if metric in ("AUC", "F1"):
                popt = project(s.N_train.values, s[metric].values, 1.0)
                if popt is not None:
                    grid = np.linspace(s.N_train.min(), 2000 * 0.8, 200)
                    ax.plot(grid, power_law(grid, *popt), ":", color=v2.PAL[3], lw=2, label="Power-law fit")
                    for N in PROJECT_N:
                        proj_rows.append(dict(Endpoint=outcome, Metric=metric, Total_N=N, Train_N=int(N * 0.8),
                                              Projected=float(power_law(N * 0.8, *popt)), Asymptote=float(popt[0])))
            if metric == "F1":
                ax.axhline(2 * prev / (1 + prev), color="k", ls=":", lw=1, label="Flag-everyone F1")
                ax.axhline(0.6, color=v2.PAL[0], ls="--", lw=1, label="Target 0.6")
            if metric == "PR_AUC":
                ax.axhline(prev, color="k", ls=":", lw=1, label="Prevalence")
            ax.set_xlabel("Training patients"); ax.set_ylabel(metric.replace("_", "-"))
            ax.set_title(f"{'sICH' if outcome == 'sICH' else 'Any HT'} — {metric.replace('_', '-')} "
                         f"(prevalence {prev * 100:.1f}%)", fontweight="bold")
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_learning_curves.png"), dpi=300)
    plt.close()
    proj = pd.DataFrame(proj_rows)
    proj.to_csv(os.path.join(OUT_DIR, "lc_projection.csv"), index=False)

    v2.divider("LEARNING CURVES — mean over folds (test metrics; train AUC shows overfitting gap)")
    show = agg[agg.Model == "Average(3)"][["Endpoint", "N_train", "AUC", "AUC_SD", "F1", "F1_SD", "PR_AUC", "Train_AUC"]]
    print(show.round(3).to_string(index=False))
    v2.divider("PROJECTION (inverse power-law fit of Average(3); extrapolation — indicative only)")
    if len(proj):
        print(proj.pivot_table(index=["Endpoint", "Metric"], columns="Total_N", values="Projected").round(3).to_string())
        print("\n  Fitted asymptotes: " + ", ".join(
            f"{e} {m}={v:.3f}" for (e, m), v in proj.groupby(["Endpoint", "Metric"])["Asymptote"].first().items()))


if __name__ == "__main__":
    main()
