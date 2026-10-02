import os
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from scipy.stats import loguniform, randint, uniform

import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, QuantileTransformer,
                                   SplineTransformer, FunctionTransformer)
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     RandomizedSearchCV, cross_val_predict)
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

FILE_PATH    = os.environ.get("SICH_DATA", "dataset.xlsx")
OUT_DIR      = os.environ.get("SICH_OUT", "outputs_v2")
OUTCOME      = os.environ.get("SICH_OUTCOME", "sICH")
POS_CODE     = os.environ.get("SICH_POS_CODE", "1")
NEG_CODE     = os.environ.get("SICH_NEG_CODE", "2")
QUICK        = os.environ.get("SICH_QUICK", "0") == "1"
RANDOM_STATE = 42
OUTER_FOLDS  = 5
OUTER_REPEATS = int(os.environ.get("SICH_REPEATS", 1 if QUICK else 3))
INNER_FOLDS  = int(os.environ.get("SICH_INNER", 3 if QUICK else 5))
N_ITER       = int(os.environ.get("SICH_NITER", 3 if QUICK else 25))
N_JOBS       = -1
N_BOOT       = 200 if QUICK else 2000
F1_TARGET    = 0.60
DPI          = 300

OUTCOME_COLS = ["sICH", "Hemorrhagic_transformation", "NIHSS_24h", "Glasgow_24h"]
NULL_TOKENS  = ["#NULL!", "#null!", "#Null!", "NULL", "null", "NaN", "nan", "NA", "N/A"]
TRUE_CATEGORICAL = {
    "Gender", "Wakeupstroke", "SBP_more_than140_admission",
    "History_of_hypertension", "History_of_diabetes", "History_of_AF",
    "History_of_CKD", "History_of_Heartfailure", "History_of_Coronarydiseases",
    "Historyof_Dislipidemia", "Anticoagulant_therapy", "Thrombolysis",
    "Occlusion_side", "Occlusion_site", "TICI", "Angioplasty", "Stenting",
    "sICH", "Hemorrhagic_transformation", "TOAST_classification",
}
YES_CODE = "1"
TICI_ORDER = {"0": 0, "1": 1, "2a": 2, "2b": 3, "2c": 4, "3": 5, "2": 2.5}
ID_PATTERN = re.compile(r"(^|_)(id|mrn|stt|code|record|patient)(_|$)", re.I)
PAL = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A", "#F4A261",
       "#264653", "#A8DADC", "#6D6875", "#B5838D", "#FFCAD4"]

os.makedirs(OUT_DIR, exist_ok=True)


def out(name):
    return os.path.join(OUT_DIR, name)


def divider(title):
    print("\n" + "═" * 78 + f"\n  {title}\n" + "═" * 78)


def normalise_codes(s):
    s = s.astype(object).where(s.notna(), np.nan)
    nonnull = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    if nonnull.sum() and num[nonnull].notna().all():
        return num.map(lambda v: np.nan if pd.isna(v) else f"{float(v):g}").astype(object)
    return s.map(lambda v: np.nan if pd.isna(v) else str(v).strip().lower().rstrip(".")).astype(object)


def load_data(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower() == "name"])
    df = df.replace(r"^\s*$", np.nan, regex=True).replace(NULL_TOKENS, np.nan)
    drop = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    drop += [c for c in df.columns if ID_PATTERN.search(c) and c not in TRUE_CATEGORICAL
             and df[c].notna().sum() > 0
             and df[c].nunique(dropna=True) >= 0.95 * df[c].notna().sum()]
    if drop:
        print(f"  [WARN] Dropping ID/datetime columns: {drop}")
        df = df.drop(columns=drop)
    for col in df.columns:
        if col in TRUE_CATEGORICAL:
            df[col] = normalise_codes(df[col])
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n0, n1 = df[col].notna().sum(), coerced.notna().sum()
        if n0 and n1 / n0 >= 0.95:
            df[col] = coerced
        elif n0:
            df[col] = normalise_codes(df[col])
    if OUTCOME not in df.columns:
        raise KeyError(f"Outcome column '{OUTCOME}' not found")
    observed = sorted(df[OUTCOME].dropna().unique().tolist())
    bad = df[~df[OUTCOME].isin([POS_CODE, NEG_CODE])]
    if len(bad):
        print(f"  [WARN] {len(bad)} row(s) with unexpected {OUTCOME} code(s) {sorted(bad[OUTCOME].astype(str).unique())} excluded; check the source file")
    df = df[df[OUTCOME].isin([POS_CODE, NEG_CODE])].copy()
    if df.empty:
        raise ValueError(f"No rows with {OUTCOME} in {{{POS_CODE},{NEG_CODE}}}; observed: {observed}")
    df["_y"] = (df[OUTCOME] == POS_CODE).astype(int)
    return df


def _yes(s):
    return (s.astype(object) == YES_CODE).astype(float).where(s.notna(), np.nan)


def add_clinical_features(df):
    d = df.copy()
    new = []
    def has(*c):
        return all(x in d.columns for x in c)
    if "TICI" in d:
        d["TICI_ordinal"] = d["TICI"].map(lambda v: TICI_ORDER.get(str(v).lower(), np.nan) if pd.notna(v) else np.nan)
        d["TICI_ge_2b"] = (d["TICI_ordinal"] >= 3).astype(float).where(d["TICI_ordinal"].notna())
        new += ["TICI_ordinal", "TICI_ge_2b"]
    if "Number_of_passes" in d:
        d["Passes_ge3"] = (d["Number_of_passes"] >= 3).astype(float).where(d["Number_of_passes"].notna())
        new.append("Passes_ge3")
    if has("Thrombolysis", "Number_of_passes"):
        d["Lysis_x_passes"] = _yes(d["Thrombolysis"]) * d["Number_of_passes"]
        new.append("Lysis_x_passes")
    if has("Thrombolysis", "Blood_glucose_at_admission"):
        d["Lysis_x_glucose"] = _yes(d["Thrombolysis"]) * d["Blood_glucose_at_admission"]
        new.append("Lysis_x_glucose")
    if has("NIHSS_at_admission", "ASPECTS_Score"):
        d["NIHSS_x_lowASPECTS"] = d["NIHSS_at_admission"] * (10 - d["ASPECTS_Score"])
        d["ASPECTS_le7"] = (d["ASPECTS_Score"] <= 7).astype(float).where(d["ASPECTS_Score"].notna())
        new += ["NIHSS_x_lowASPECTS", "ASPECTS_le7"]
    if has("Age", "NIHSS_at_admission"):
        d["Age_x_NIHSS"] = d["Age"] * d["NIHSS_at_admission"]
        new.append("Age_x_NIHSS")
    sedan = []
    if "Age" in d:
        sedan.append((d["Age"] > 75).astype(float).where(d["Age"].notna()))
    if "NIHSS_at_admission" in d:
        sedan.append((d["NIHSS_at_admission"] >= 10).astype(float).where(d["NIHSS_at_admission"].notna()))
    if "ASPECTS_Score" in d:
        sedan.append((d["ASPECTS_Score"] <= 7).astype(float).where(d["ASPECTS_Score"].notna()))
    if "Blood_glucose_at_admission" in d:
        g = d["Blood_glucose_at_admission"]
        sedan.append((g > g.median() * 1.5).astype(float).where(g.notna()))
    if len(sedan) >= 3:
        d["SEDAN_like"] = pd.concat(sedan, axis=1).sum(axis=1, min_count=1)
        new.append("SEDAN_like")
    if has("History_of_AF", "Anticoagulant_therapy"):
        d["AF_or_anticoag"] = ((_yes(d["History_of_AF"]) == 1) | (_yes(d["Anticoagulant_therapy"]) == 1)).astype(float)
        new.append("AF_or_anticoag")
    if has("Onset_to_successful_recanalization"):
        d["Log_onset_to_recan"] = np.log1p(d["Onset_to_successful_recanalization"].clip(lower=0))
        new.append("Log_onset_to_recan")
    if has("Platelet_count_at_admission", "INR_at_admission"):
        d["Platelet_over_INR"] = d["Platelet_count_at_admission"] / d["INR_at_admission"].where(d["INR_at_admission"] > 0)
        new.append("Platelet_over_INR")
    return d, new


def _to_object_nan(d):
    d = pd.DataFrame(d).astype(object)
    return d.where(pd.notna(d), np.nan)


def cat_pipe():
    return Pipeline([("obj", FunctionTransformer(_to_object_nan, validate=False)),
                     ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5))])


def make_pre(num, cat, linear):
    num_steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if linear:
        num_steps.append(("spl", "passthrough"))
    tr = [("num", Pipeline(num_steps), num),
          ("miss", MissingIndicator(features="missing-only", error_on_new=False), num)]
    if cat:
        tr.append(("cat", cat_pipe(), cat))
    return ColumnTransformer(tr)


def enet_lr(**kw):
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8):
        return LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000, **kw)
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000, **kw)


def candidates(num, cat, spw):
    smote = ["passthrough", SMOTE(k_neighbors=3, random_state=RANDOM_STATE)]
    c = {}
    c["ElasticNet-LR (splines)"] = (
        Pipeline([("pre", make_pre(num, cat, True)), ("smote", "passthrough"),
                  ("clf", enet_lr(random_state=RANDOM_STATE))]),
        {"pre__num__sc": [StandardScaler(), QuantileTransformer(output_distribution="normal", n_quantiles=100)],
         "pre__num__spl": ["passthrough", SplineTransformer(n_knots=4, degree=3)],
         "clf__C": loguniform(1e-3, 10), "clf__l1_ratio": uniform(0, 1),
         "clf__class_weight": [None, "balanced"], "smote": smote})
    c["SVM-RBF"] = (
        Pipeline([("pre", make_pre(num, cat, False)), ("smote", "passthrough"),
                  ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
        {"clf__C": loguniform(0.05, 100), "clf__gamma": loguniform(1e-4, 1e-1),
         "clf__class_weight": [None, "balanced"], "smote": smote})
    c["RandomForest"] = (
        Pipeline([("pre", make_pre(num, cat, False)), ("smote", "passthrough"),
                  ("clf", RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=RANDOM_STATE))]),
        {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10, 20],
         "clf__max_features": ["sqrt", 0.2, 0.35, 0.5],
         "clf__class_weight": [None, "balanced", "balanced_subsample"], "smote": smote})
    c["ExtraTrees"] = (
        Pipeline([("pre", make_pre(num, cat, False)), ("smote", "passthrough"),
                  ("clf", ExtraTreesClassifier(n_estimators=500, n_jobs=1, random_state=RANDOM_STATE))]),
        {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10, 20],
         "clf__max_features": ["sqrt", 0.2, 0.35, 0.5],
         "clf__class_weight": [None, "balanced", "balanced_subsample"], "smote": smote})
    c["XGBoost"] = (
        Pipeline([("pre", make_pre(num, cat, False)), ("smote", "passthrough"),
                  ("clf", XGBClassifier(eval_metric="logloss", n_jobs=1, verbosity=0,
                                        random_state=RANDOM_STATE))]),
        {"clf__n_estimators": randint(100, 700), "clf__learning_rate": loguniform(0.01, 0.2),
         "clf__max_depth": [2, 3, 4], "clf__min_child_weight": [1, 3, 5, 10],
         "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.3, 0.7),
         "clf__reg_lambda": loguniform(0.1, 20), "clf__gamma": [0, 0.5, 1, 2],
         "clf__scale_pos_weight": [1.0, float(np.sqrt(spw)), spw], "smote": smote})
    c["LightGBM"] = (
        Pipeline([("pre", make_pre(num, cat, False)), ("smote", "passthrough"),
                  ("clf", LGBMClassifier(n_jobs=1, verbose=-1, random_state=RANDOM_STATE))]),
        {"clf__n_estimators": randint(100, 700), "clf__learning_rate": loguniform(0.01, 0.2),
         "clf__num_leaves": [3, 4, 7, 15], "clf__min_child_samples": [5, 10, 20, 40],
         "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1],
         "clf__colsample_bytree": uniform(0.3, 0.7), "clf__reg_lambda": loguniform(0.1, 20),
         "clf__scale_pos_weight": [1.0, float(np.sqrt(spw)), spw], "smote": smote})
    c["CatBoost"] = (
        Pipeline([("pre", make_pre(num, cat, False)), ("smote", "passthrough"),
                  ("clf", CatBoostClassifier(verbose=0, thread_count=1, allow_writing_files=False,
                                             random_seed=RANDOM_STATE))]),
        {"clf__iterations": [200, 400, 800], "clf__learning_rate": loguniform(0.01, 0.2),
         "clf__depth": [3, 4, 5, 6], "clf__l2_leaf_reg": loguniform(1, 30),
         "clf__auto_class_weights": [None, "Balanced", "SqrtBalanced"], "smote": smote})
    return c


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, 0.0
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.argmax(f1s))
    return float(thr[i]), float(f1s[i])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_outer_fold(X_tr, y_tr, X_te, num, cat, spw, seed):
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    res = {}
    for name, (pipe, dist) in candidates(num, cat, spw).items():
        est = clone(pipe)
        params = {}
        if dist:
            search = RandomizedSearchCV(est, dist, n_iter=N_ITER, scoring="average_precision",
                                        cv=inner, random_state=seed, n_jobs=N_JOBS,
                                        refit=False, error_score=np.nan)
            search.fit(X_tr, y_tr)
            params = search.best_params_
            est.set_params(**params)
        oof = cross_val_predict(est, X_tr, y_tr, cv=inner, method="predict_proba", n_jobs=N_JOBS)[:, 1]
        thr, f1_in = best_threshold(y_tr, oof)
        est.fit(X_tr, y_tr)
        res[name] = dict(oof=oof, thr=thr, inner_f1=f1_in,
                         inner_ap=average_precision_score(y_tr, oof),
                         p=est.predict_proba(X_te)[:, 1], params=params)

    base = list(res)
    top3 = sorted(base, key=lambda n: -res[n]["inner_ap"])[:3]
    oof_avg = np.mean([res[n]["oof"] for n in top3], axis=0)
    thr, f1_in = best_threshold(y_tr, oof_avg)
    res["Average(top3)"] = dict(oof=oof_avg, thr=thr, inner_f1=f1_in,
                                inner_ap=average_precision_score(y_tr, oof_avg),
                                p=np.mean([res[n]["p"] for n in top3], axis=0), params={"members": top3})

    Z_tr = np.column_stack([logit(res[n]["oof"]) for n in base])
    Z_te = np.column_stack([logit(res[n]["p"]) for n in base])
    meta = LogisticRegression(C=0.5, max_iter=5000)
    meta_oof = cross_val_predict(meta, Z_tr, y_tr, cv=inner, method="predict_proba")[:, 1]
    thr, f1_in = best_threshold(y_tr, meta_oof)
    meta.fit(Z_tr, y_tr)
    res["Stacking(LR)"] = dict(oof=meta_oof, thr=thr, inner_f1=f1_in,
                               inner_ap=average_precision_score(y_tr, meta_oof),
                               p=meta.predict_proba(Z_te)[:, 1], params={})

    auto = max(res, key=lambda n: (res[n]["inner_f1"], res[n]["inner_ap"]))
    res["AutoSelect (nested)"] = dict(res[auto], params={"chosen": auto})
    return res


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(F1=f1_score(y, pred, zero_division=0),
                Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0),
                AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                Brier=brier_score_loss(y, p))


def boot_f1_ci(y, pred, n, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(f1_score(y[i], pred[i], zero_division=0))
    return np.percentile(vals, [2.5, 97.5])


def net_benefit(y, p, thresholds):
    y = np.asarray(y)
    n = len(y)
    nb = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb.append(tp / n - fp / n * t / (1 - t))
    return np.array(nb)


def main():
    df = load_data(FILE_PATH)
    y = df["_y"].astype(int).reset_index(drop=True)
    X = df.drop(columns=["_y"] + [c for c in OUTCOME_COLS if c in df.columns]).reset_index(drop=True)
    X, new_feats = add_clinical_features(X)
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    prev = y.mean()
    spw = float((y == 0).sum() / max(y.sum(), 1))

    divider("PREDICTION v2 — setup")
    print(f"  Outcome              : {OUTCOME} (positive code '{POS_CODE}')")
    print(f"  Patients / events    : {len(y)} / {int(y.sum())} ({prev * 100:.1f}%)")
    print(f"  Features             : {X.shape[1]} ({len(num)} numeric, {len(cat)} categorical)")
    print(f"  Clinical features added ({len(new_feats)}): {new_feats}")
    print(f"  Excluded outcome/post-outcome columns: {[c for c in OUTCOME_COLS if c in df.columns]}")
    print(f"  Evaluation           : {OUTER_REPEATS}x repeated {OUTER_FOLDS}-fold outer CV on ALL patients;")
    print(f"                         tuning ({N_ITER} random configs, PR-AUC), threshold and model")
    print(f"                         selection all inside {INNER_FOLDS}-fold inner CV")
    print(f"  Reference F1 (flag everyone) : {2 * prev / (1 + prev):.3f}")

    rskf = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
    folds = list(rskf.split(X, y))
    per_fold = []
    pooled = {}
    chosen = []
    for j, (tr, te) in enumerate(folds):
        rep = j // OUTER_FOLDS
        print(f"  Outer fold {j + 1}/{len(folds)} ...", flush=True)
        res = run_outer_fold(X.iloc[tr], y.iloc[tr], X.iloc[te], num, cat, spw, RANDOM_STATE + j)
        chosen.append(res["AutoSelect (nested)"]["params"]["chosen"])
        for name, r in res.items():
            m = metrics(y.iloc[te].values, r["p"], r["thr"])
            per_fold.append({"Approach": name, "Fold": j, "Repeat": rep, "Threshold": r["thr"], **m})
            d = pooled.setdefault((name, rep), {"idx": [], "p": [], "pred": []})
            d["idx"].extend(te.tolist())
            d["p"].extend(r["p"].tolist())
            d["pred"].extend((r["p"] >= r["thr"]).astype(int).tolist())

    pf = pd.DataFrame(per_fold)
    pf.to_csv(out("v2_per_fold_metrics.csv"), index=False)
    rows = []
    curves = {}
    for name in pf["Approach"].unique():
        pooled_scores = []
        for rep in range(OUTER_REPEATS):
            d = pooled[(name, rep)]
            yy = y.values[d["idx"]]
            pp, pr = np.array(d["p"]), np.array(d["pred"])
            pooled_scores.append(dict(F1=f1_score(yy, pr, zero_division=0),
                                      Precision=precision_score(yy, pr, zero_division=0),
                                      Recall=recall_score(yy, pr, zero_division=0),
                                      AUC=roc_auc_score(yy, pp), PR_AUC=average_precision_score(yy, pp),
                                      Brier=brier_score_loss(yy, pp)))
            if rep == 0:
                curves[name] = (yy, pp, pr)
        ps = pd.DataFrame(pooled_scores).mean()
        lo, hi = boot_f1_ci(*curves[name][0::2], N_BOOT)
        sub = pf[pf.Approach == name]
        rows.append({"Approach": name, "F1 (pooled OOF)": ps["F1"], "F1 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "F1 fold mean": sub["F1"].mean(), "F1 fold SD": sub["F1"].std(),
                     "Precision": ps["Precision"], "Recall": ps["Recall"], "AUC": ps["AUC"],
                     "PR-AUC": ps["PR_AUC"], "Brier": ps["Brier"], "Mean threshold": sub["Threshold"].mean()})
    summary = pd.DataFrame(rows).sort_values("F1 (pooled OOF)", ascending=False).round(4)
    summary.to_csv(out("v2_summary.csv"), index=False)

    divider("RESULTS — repeated nested CV on all patients (every patient is a test case)")
    print(summary.to_string(index=False))
    auto = summary[summary.Approach == "AutoSelect (nested)"].iloc[0]
    best = summary[summary.Approach != "AutoSelect (nested)"].iloc[0]
    print(f"\n  Models picked by AutoSelect across folds: { {k: int(v) for k, v in pd.Series(chosen).value_counts().items()} }")
    print(f"\n  Unbiased estimate of the whole procedure (AutoSelect): F1 = {auto['F1 (pooled OOF)']:.3f} "
          f"{auto['F1 95% CI']}, AUC = {auto['AUC']:.3f}")
    print(f"  Best single approach in hindsight: {best['Approach']} F1 = {best['F1 (pooled OOF)']:.3f} "
          f"(slightly optimistic: chosen after seeing these results)")
    flag_all = 2 * prev / (1 + prev)
    print(f"  Flag-everyone baseline F1: {flag_all:.3f}  →  lift of AutoSelect: {auto['F1 (pooled OOF)'] - flag_all:+.3f}")
    reached = auto["F1 (pooled OOF)"] >= F1_TARGET
    print(f"\n  F1 target {F1_TARGET:.2f}: {'REACHED' if reached else 'NOT REACHED'} by the unbiased estimate.")
    if not reached:
        print("  Remaining legitimate levers: more events (pooled/multicentre data), stronger")
        print("  predictors (post-procedural BP, collateral grade, final infarct volume, contrast")
        print("  extravasation on post-EVT CT), or a different pre-registered endpoint (e.g. any")
        print("  haemorrhagic transformation: SICH_OUTCOME=Hemorrhagic_transformation).")

    if os.environ.get("SICH_SKIP_FINAL", "0") == "1":
        print("\n  [INFO] Final refit skipped (SICH_SKIP_FINAL=1)")
    else:
        refit_final(X, y, num, cat, spw)
    make_figures(summary, curves, auto, best, prev)


def refit_final(X, y, num, cat, spw):
    print("\nRefitting the full procedure on all patients for the deployable model...")
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    final_res = run_outer_fold(X, y, X.iloc[:5], num, cat, spw, RANDOM_STATE)
    fin_name = final_res["AutoSelect (nested)"]["params"]["chosen"]
    base_names = [n for n in final_res if n not in ("Average(top3)", "Stacking(LR)", "AutoSelect (nested)")]
    cands = candidates(num, cat, spw)
    fitted = {}
    for n in base_names:
        est = clone(cands[n][0]).set_params(**final_res[n]["params"])
        fitted[n] = est.fit(X, y)
    bundle = dict(chosen=fin_name, threshold=final_res[fin_name]["thr"], models=fitted,
                  members=final_res["Average(top3)"]["params"]["members"],
                  params={n: final_res[n]["params"] for n in base_names},
                  features=X.columns.tolist(), outcome=OUTCOME)
    joblib.dump(bundle, out("v2_final_model.joblib"))
    print(f"  Final approach: {fin_name}; threshold {bundle['threshold']:.3f}; saved v2_final_model.joblib")
    print("  Report the CV estimate above, not performance on the training data.")


def make_figures(summary, curves, auto, best, prev):

    show = [auto["Approach"], best["Approach"]] + [n for n in summary.Approach if n not in (auto["Approach"], best["Approach"])][:3]
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    s = summary.iloc[::-1]
    cis = [tuple(float(v) for v in c.strip("[]").split(",")) for c in s["F1 95% CI"]]
    ax[0, 0].barh(s["Approach"], s["F1 (pooled OOF)"],
                  xerr=[[f - c[0] for f, c in zip(s["F1 (pooled OOF)"], cis)],
                        [c[1] - f for f, c in zip(s["F1 (pooled OOF)"], cis)]],
                  color=[PAL[0] if a == "AutoSelect (nested)" else PAL[1] for a in s["Approach"]], capsize=3)
    ax[0, 0].axvline(F1_TARGET, color="k", ls="--", lw=1, label=f"Target {F1_TARGET}")
    ax[0, 0].axvline(2 * prev / (1 + prev), color="grey", ls=":", lw=1, label="Flag-everyone F1")
    ax[0, 0].set_xlabel("F1 (pooled out-of-fold, 95% CI)"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("(A) F1 by approach", fontweight="bold")
    for i, n in enumerate(show):
        yy, pp, _ = curves[n]
        fpr, tpr, _ = roc_curve(yy, pp)
        ax[0, 1].plot(fpr, tpr, color=PAL[i], lw=2 if i < 2 else 1, label=f"{n} ({roc_auc_score(yy, pp):.3f})")
        pr, rc, _ = precision_recall_curve(yy, pp)
        ax[1, 0].plot(rc, pr, color=PAL[i], lw=2 if i < 2 else 1, label=f"{n} ({average_precision_score(yy, pp):.3f})")
        ptrue, ppred = calibration_curve(yy, pp, n_bins=8, strategy="quantile")
        ax[1, 1].plot(ppred, ptrue, "o-", color=PAL[i], lw=2 if i < 2 else 1, ms=3, label=n)
    ax[0, 1].plot([0, 1], [0, 1], "--", color="grey"); ax[0, 1].set_title("(B) ROC", fontweight="bold")
    ax[0, 1].set_xlabel("FPR"); ax[0, 1].set_ylabel("TPR"); ax[0, 1].legend(fontsize=7)
    ax[1, 0].axhline(prev, ls="--", color="grey"); ax[1, 0].set_title("(C) Precision–recall", fontweight="bold")
    ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot([0, 1], [0, 1], "--", color="grey"); ax[1, 1].set_title("(D) Calibration", fontweight="bold")
    ax[1, 1].set_xlabel("Predicted"); ax[1, 1].set_ylabel("Observed"); ax[1, 1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out("v2_performance.png"), dpi=DPI); plt.close()

    ts = np.linspace(0.01, min(0.8, max(0.3, prev * 4)), 80)
    fig, a = plt.subplots(figsize=(8, 5))
    yy0 = curves[show[0]][0]
    a.plot(ts, net_benefit(yy0, np.ones(len(yy0)), ts), color="grey", ls="--", label="Treat all")
    a.axhline(0, color="k", lw=0.8, label="Treat none")
    for i, n in enumerate(show[:3]):
        yy, pp, _ = curves[n]
        a.plot(ts, net_benefit(yy, pp, ts), color=PAL[i], lw=2, label=n)
    a.set_ylim(-0.05, prev * 1.1); a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit")
    a.set_title("Decision curve analysis", fontweight="bold"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out("v2_decision_curve.png"), dpi=DPI); plt.close()
    print(f"\n  Outputs in '{OUT_DIR}/': v2_summary.csv, v2_per_fold_metrics.csv, "
          f"v2_performance.png, v2_decision_curve.png, v2_final_model.joblib")


if __name__ == "__main__":
    main()
