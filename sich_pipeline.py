import os
import re
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import networkx as nx
import networkx.algorithms as nxa
from collections import Counter, defaultdict
from joblib import Parallel, delayed
from scipy import stats

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler,
                                   FunctionTransformer)
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                             roc_curve, brier_score_loss, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestRegressor,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from causallearn.search.ConstraintBased.PC import pc as pc_algo
from causallearn.search.ScoreBased.GES import ges as ges_algo
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from econml.dml import CausalForestDML
from dowhy import CausalModel
import dice_ml
from dice_ml import Dice

if not hasattr(nxa, "d_separated"):
    from networkx.algorithms.d_separation import is_d_separator
    nxa.d_separated = is_d_separator

for _lg in ("dowhy", "econml", "causallearn"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

FILE_PATH       = os.environ.get("SICH_DATA", "dataset.xlsx")
OUT_DIR         = os.environ.get("SICH_OUT", "outputs")
QUICK           = os.environ.get("SICH_QUICK", "0") == "1"

RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5
CV_REPEATS      = 1 if QUICK else 2
PC_ALPHA        = 0.05
INDEP_TEST      = "fisherz"
MB_STRATEGY     = "union"
IMBALANCE       = "class_weight"
SELECTION_METRIC = "AUC"
VIF_MAX         = 1000.0
N_STABILITY     = 20 if QUICK else 100
STABILITY_FRAC  = 0.80
N_BOOT_EFFECT   = 100 if QUICK else 1000
N_BOOT_AUC      = 200 if QUICK else 1000
N_REFUTE_SIMS   = 20 if QUICK else 100
N_DICE_PATIENTS = 10
TREATMENT_OVERRIDE = None
N_JOBS          = -1
DPI             = 600
TARGET          = "sICH_bin"

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
BG   = "#FFFFFF"
GRAY = "#AAAAAA"
PAL  = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A", "#F4A261",
        "#264653", "#A8DADC", "#6D6875", "#B5838D", "#FFCAD4"]


def divider(title):
    print("\n" + "═" * 72 + f"\n  {title}\n" + "═" * 72)


def section(title):
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def out(name):
    return os.path.join(OUT_DIR, name)


def save_table(frame, name):
    frame.to_csv(out(name), index=False)


LEAKAGE_COLS = ["NIHSS_24h", "Hemorrhagic_transformation", "Glasgow_24h"]
NULL_TOKENS  = ["#NULL!", "#null!", "#Null!", "NULL", "null", "NaN", "nan", "NA", "N/A"]

TRUE_CATEGORICAL = {
    "Gender", "Wakeupstroke", "SBP_more_than140_admission",
    "History_of_hypertension", "History_of_diabetes", "History_of_AF",
    "History_of_CKD", "History_of_Heartfailure", "History_of_Coronarydiseases",
    "Historyof_Dislipidemia", "Anticoagulant_therapy", "Thrombolysis",
    "Occlusion_side", "Occlusion_site", "TICI", "Angioplasty", "Stenting",
    "sICH", "TOAST_classification",
}

def _safe_ratio(num, den, eps=1e-6):
    return num / den.mask(den.abs() < eps)

ENGINEERED_SPEC = {
    "NPR_at_admission": (["Neutrophil_at_admission", "Platelet_count_at_admission"],
                         lambda d: _safe_ratio(d["Neutrophil_at_admission"], d["Platelet_count_at_admission"])),
    "Neutrophil_WBC_fraction_at_admission": (["Neutrophil_at_admission", "White_bloodcell_at_admission"],
                         lambda d: _safe_ratio(d["Neutrophil_at_admission"], d["White_bloodcell_at_admission"])),
    "Pulse_pressure_at_admission": (["SBP_at_admission", "DBP_at_admission"],
                         lambda d: d["SBP_at_admission"] - d["DBP_at_admission"]),
    "BUN_Creatinine_ratio_at_admission": (["Ure_at_admission", "Creatinin_at_admission"],
                         lambda d: _safe_ratio(d["Ure_at_admission"], d["Creatinin_at_admission"])),
    "Cholesterol_HDL_ratio": (["Cholesterol", "HDL_C"],
                         lambda d: _safe_ratio(d["Cholesterol"], d["HDL_C"])),
}
DETERMINISTIC_DERIVED = {"SBP_more_than140_admission": ["SBP_at_admission"]}
KNOWN_COMPOSITES = [
    ("Onset_to_groinpunture", ["Onset_to_admission", "Admission_to_groinpunture"]),
    ("Onset_to_successful_recanalization", ["Onset_to_groinpunture", "EVT_time"]),
]

NON_MODIFIABLE = {"Age", "Gender", "Wakeupstroke", "Occlusion_side", "Occlusion_site",
                  "TOAST_classification", "History_of_hypertension", "History_of_diabetes",
                  "History_of_AF", "History_of_CKD", "History_of_Heartfailure",
                  "History_of_Coronarydiseases", "Historyof_Dislipidemia"}

DICE_EXCLUDE = NON_MODIFIABLE | {
    "Onset_to_admission", "Onset_to_successful_recanalization",
    "Onset_to_groinpunture", "Admission_to_groinpunture",
    "EVT_time", "Number_of_passes", "Thrombolysis", "Anticoagulant_therapy",
} | set(ENGINEERED_SPEC)

CLINICAL_DOMAIN = {
    "Age": "Demographics", "Gender": "Demographics",
    "History_of_hypertension": "Comorbidity", "History_of_diabetes": "Comorbidity",
    "History_of_AF": "Comorbidity", "History_of_CKD": "Comorbidity",
    "History_of_Heartfailure": "Comorbidity", "History_of_Coronarydiseases": "Comorbidity",
    "Historyof_Dislipidemia": "Comorbidity",
    "Wakeupstroke": "Onset/Timing", "Onset_to_admission": "Onset/Timing",
    "GCS_at_admission": "Neurological", "NIHSS_at_admission": "Neurological",
    "ASPECTS_Score": "Neuroimaging", "Occlusion_side": "Neuroimaging",
    "Occlusion_site": "Neuroimaging", "TOAST_classification": "Neuroimaging",
    "Pulse_at_admission": "Hemodynamic", "SBP_at_admission": "Hemodynamic",
    "SBP_more_than140_admission": "Hemodynamic", "DBP_at_admission": "Hemodynamic",
    "Pulse_pressure_at_admission": "Hemodynamic",
    "Blood_glucose_at_admission": "Metabolic", "spO2_at_admission": "Respiratory",
    "Cholesterol": "Metabolic/Lipid", "Triglycerid": "Metabolic/Lipid",
    "LDL_C": "Metabolic/Lipid", "HDL_C": "Metabolic/Lipid",
    "Cholesterol_HDL_ratio": "Metabolic/Lipid",
    "Hb_at_admission": "Hematologic", "Platelet_count_at_admission": "Hematologic",
    "White_bloodcell_at_admission": "Hematologic/Inflammatory",
    "Neutrophil_at_admission": "Hematologic/Inflammatory",
    "NPR_at_admission": "Hematologic/Inflammatory",
    "Neutrophil_WBC_fraction_at_admission": "Hematologic/Inflammatory",
    "INR_at_admission": "Coagulation", "APTT_at_admission": "Coagulation",
    "APTTbc_at_admission": "Coagulation", "Fibrinogen_at_admission": "Coagulation",
    "Anticoagulant_therapy": "Coagulation", "Thrombolysis": "Coagulation",
    "Ure_at_admission": "Renal", "Creatinin_at_admission": "Renal",
    "BUN_Creatinine_ratio_at_admission": "Renal",
    "AST_at_admission": "Hepatic", "ALT_at_admission": "Hepatic",
    "TroponinThs_at_admission": "Cardiac",
    "TICI": "Procedural", "Number_of_passes": "Procedural",
    "Angioplasty": "Procedural", "Stenting": "Procedural",
    "Admission_to_groinpunture": "Procedural", "Onset_to_groinpunture": "Procedural",
    "Onset_to_successful_recanalization": "Procedural", "EVT_time": "Procedural",
}

TIER_BASELINE, TIER_ONSET, TIER_ADMISSION, TIER_PROCEDURE, TIER_OUTCOME = 0, 1, 2, 3, 4
TIER_MAP = {
    **{v: TIER_BASELINE for v in ["Age", "Gender", "History_of_hypertension",
        "History_of_diabetes", "History_of_AF", "History_of_CKD",
        "History_of_Heartfailure", "History_of_Coronarydiseases", "Historyof_Dislipidemia"]},
    **{v: TIER_ONSET for v in ["Wakeupstroke", "Onset_to_admission"]},
    **{v: TIER_PROCEDURE for v in ["Admission_to_groinpunture", "Onset_to_groinpunture",
        "Onset_to_successful_recanalization", "EVT_time", "TICI", "TICI_ordinal", "Number_of_passes",
        "Angioplasty", "Stenting"]},
    TARGET: TIER_OUTCOME,
}


def get_tier(v):
    return TIER_MAP.get(v, TIER_ADMISSION)


def get_clinical_domain(v):
    return CLINICAL_DOMAIN.get(v, "Other")


def normalise_codes(s):
    s = s.astype(object).where(s.notna(), np.nan)
    nonnull = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    if nonnull.sum() and num[nonnull].notna().all():
        return num.map(lambda v: np.nan if pd.isna(v) else f"{float(v):g}").astype(object)
    return s.map(lambda v: np.nan if pd.isna(v) else str(v).strip().rstrip(".")).astype(object)


ID_PATTERN = re.compile(r"(^|_)(id|mrn|stt|code|record|patient)(_|$)", re.I)


def load_data(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower() == "name"])
    df = df.replace(r"^\s*$", np.nan, regex=True).replace(NULL_TOKENS, np.nan)

    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    id_cols = [c for c in df.columns if ID_PATTERN.search(c) and c not in TRUE_CATEGORICAL
               and df[c].notna().sum() > 0
               and df[c].nunique(dropna=True) >= 0.95 * df[c].notna().sum()]
    if dt_cols or id_cols:
        print(f"  [WARN] Dropping datetime columns {dt_cols} and ID-like columns {id_cols}")
        df = df.drop(columns=dt_cols + id_cols)

    for col in df.columns:
        if col in TRUE_CATEGORICAL:
            df[col] = normalise_codes(df[col])
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n0, n1 = df[col].notna().sum(), coerced.notna().sum()
        if n0 == 0:
            continue
        if n1 / n0 >= 0.95:
            if n1 < n0:
                print(f"  [WARN] {col}: {n0 - n1} non-numeric entries set to missing")
            df[col] = coerced
        else:
            print(f"  [WARN] {col}: not numeric and not declared categorical → treated as categorical")
            df[col] = normalise_codes(df[col])

    if "sICH" not in df.columns:
        raise KeyError("Column 'sICH' not found")
    observed = sorted(df["sICH"].dropna().unique().tolist())
    df = df[df["sICH"].isin(["1", "2"])].copy()
    if df.empty:
        raise ValueError(f"No rows with sICH in {{'1','2'}}; observed codes: {observed}")
    df[TARGET] = df["sICH"].map({"1": 1, "2": 0}).astype(int)

    for name, (comps, fn) in ENGINEERED_SPEC.items():
        if set(comps).issubset(df.columns):
            df[name] = fn(df)
    return df


def _to_object_nan(d):
    d = pd.DataFrame(d).astype(object)
    return d.where(pd.notna(d), np.nan)


def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(feats, clf, num_all, imbalance, smote_k):
    num = [f for f in feats if f in num_all]
    cat = [f for f in feats if f not in num_all]
    cat_prep = [("obj", FunctionTransformer(_to_object_nan, validate=False)),
                ("imp", SimpleImputer(strategy="constant", fill_value="Missing"))]
    if imbalance != "smotenc":
        tr = []
        if num:
            tr.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), num))
        if cat:
            tr.append(("cat", Pipeline(cat_prep + [("oh", _ohe())]), cat))
        return Pipeline([("pre", ColumnTransformer(tr)), ("clf", clf)])

    tr1 = []
    if num:
        tr1.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        tr1.append(("cat", Pipeline(cat_prep + [("ord", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1))]), cat))
    n_num, n_cat = len(num), len(cat)
    cat_idx = list(range(n_num, n_num + n_cat))
    if n_cat and n_num:
        sampler = SMOTENC(categorical_features=cat_idx, k_neighbors=smote_k, random_state=RANDOM_STATE)
    elif n_cat:
        sampler = SMOTEN(k_neighbors=smote_k, random_state=RANDOM_STATE)
    else:
        sampler = SMOTE(k_neighbors=smote_k, random_state=RANDOM_STATE)
    tr2 = []
    if n_num:
        tr2.append(("num", StandardScaler(), list(range(n_num))))
    if n_cat:
        tr2.append(("cat", _ohe(), cat_idx))
    return Pipeline([("pre1", ColumnTransformer(tr1)), ("smote", sampler),
                     ("pre2", ColumnTransformer(tr2)), ("clf", clf)])


def build_models(imbalance, scale_pos_weight):
    cw = "balanced" if imbalance == "class_weight" else None
    cb_kw = dict(iterations=300, learning_rate=0.05, depth=6, random_seed=RANDOM_STATE,
                 verbose=0, thread_count=1, allow_writing_files=False)
    if cw:
        cb_kw["auto_class_weights"] = "Balanced"
    return {
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight=cw, random_state=RANDOM_STATE),
        "KNeighbors":         KNeighborsClassifier(n_neighbors=7),
        "DecisionTree":       DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
        "RandomForest":       RandomForestClassifier(n_estimators=300, class_weight=cw,
                                                     random_state=RANDOM_STATE, n_jobs=1),
        "ExtraTrees":         ExtraTreesClassifier(n_estimators=300, class_weight=cw,
                                                   random_state=RANDOM_STATE, n_jobs=1),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM_RBF":            SVC(kernel="rbf", probability=True, class_weight=cw, random_state=RANDOM_STATE),
        "GaussianNB":         GaussianNB(),
        "XGBoost":            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                            scale_pos_weight=scale_pos_weight if cw else 1.0,
                                            random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
        "CatBoost":           CatBoostClassifier(**cb_kw),
    }


def make_ensemble(models, names):
    return VotingClassifier([(n, clone(models[n])) for n in names], voting="soft")


def score_block(y_true, proba, thr=0.5):
    y_true = np.asarray(y_true)
    pred = (proba >= thr).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "AUC":       roc_auc_score(y_true, proba) if both else np.nan,
        "PR_AUC":    average_precision_score(y_true, proba) if both else np.nan,
        "Brier":     brier_score_loss(y_true, proba),
        "F1":        f1_score(y_true, pred, zero_division=0),
        "Recall":    recall_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
    }


def tune_threshold(pipe, X_tr, y_tr):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    if len(thr) == 0:
        return 0.5
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.argmax(f1s[:-1]))])


def bootstrap_auc_ci(y_true, proba, n_boot, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def corrected_ttest(diffs, n_train, n_test):
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    J = len(diffs)
    m = diffs.mean() if J else np.nan
    if J < 2:
        return dict(mean=m, lo=np.nan, hi=np.nan, t=np.nan, p=np.nan)
    v = diffs.var(ddof=1)
    if v == 0:
        return dict(mean=m, lo=m, hi=m, t=np.nan, p=1.0 if m == 0 else 0.0)
    se = np.sqrt((1.0 / J + n_test / n_train) * v)
    t = m / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    tc = stats.t.ppf(0.975, J - 1)
    return dict(mean=m, lo=m - tc * se, hi=m + tc * se, t=t, p=p)


def composite_holds(frame, target, parts, frac=0.95):
    cols = [target] + parts
    if not set(cols).issubset(frame.columns):
        return False
    sub = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10:
        return False
    resid = (sub[target] - sub[parts].sum(axis=1)).abs()
    tol = 0.02 * sub[target].abs().median() + 1e-6
    return (resid <= tol).mean() >= frac


def discovery_spec(X_train, numeric_features, categorical_features):
    excluded = {}
    num = []
    for c in numeric_features:
        if c in ENGINEERED_SPEC:
            excluded[c] = "engineered ratio/difference"
        elif c in DETERMINISTIC_DERIVED:
            excluded[c] = "deterministic function of another variable"
        else:
            num.append(c)
    for tgt, parts in KNOWN_COMPOSITES:
        if tgt in num and composite_holds(X_train, tgt, parts):
            num.remove(tgt)
            excluded[tgt] = f"additive composite of {parts}"
    bin_cols, bin_maps = [], {}
    for c in categorical_features:
        levels = sorted(X_train[c].dropna().unique().tolist())
        if c in DETERMINISTIC_DERIVED:
            excluded[c] = "deterministic function of another variable"
        elif len(levels) == 2:
            bin_cols.append(c)
            bin_maps[c] = {levels[0]: 0.0, levels[1]: 1.0}
        else:
            excluded[c] = f"nominal with {len(levels)} levels (not Fisher-z compatible)"
    return (num, bin_cols, bin_maps), excluded


def encode_for_discovery(X_sub, y_sub, spec):
    num, bin_cols, bin_maps = spec
    parts = {c: pd.to_numeric(X_sub[c], errors="coerce") for c in num}
    for c in bin_cols:
        parts[c] = X_sub[c].map(bin_maps[c]).astype(float)
    Z = pd.DataFrame(parts, index=X_sub.index)
    Z = Z.loc[:, Z.notna().any()]
    Z = Z.fillna(Z.median())
    Z[TARGET] = np.asarray(y_sub, float)
    return Z


def vif_prune(Zs, protected=(TARGET,)):
    cols, dropped = list(Zs.columns), []
    A = Zs.values
    while True:
        best, best_c = 0.0, None
        idx = {c: i for i, c in enumerate(Zs.columns)}
        for c in cols:
            if c in protected:
                continue
            others = [idx[o] for o in cols if o != c]
            yv = A[:, idx[c]]
            Xo = np.column_stack([np.ones(len(yv)), A[:, others]])
            beta, *_ = np.linalg.lstsq(Xo, yv, rcond=None)
            r2 = 1 - np.sum((yv - Xo @ beta) ** 2) / max(np.sum((yv - yv.mean()) ** 2), 1e-12)
            vif = 1.0 / max(1.0 - r2, 1e-12)
            if vif > best:
                best, best_c = vif, c
        if best_c is None or best < VIF_MAX:
            return cols, dropped
        cols.remove(best_c)
        dropped.append(best_c)


def build_bk(names):
    bk = BackgroundKnowledge()
    for n in names:
        bk.add_node_to_tier(GraphNode(n), get_tier(n))
    return bk


def extract_edges(G, names):
    d, u = set(), set()
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                d.add((names[i], names[j]))
            elif a == 1 and b == -1:
                d.add((names[j], names[i]))
            else:
                u.add(tuple(sorted((names[i], names[j]))))
    return d, u


def orient_by_tiers(d, u):
    nd, nu, n_rev = set(), set(), 0
    for a, b in d:
        if get_tier(a) > get_tier(b):
            nd.add((b, a)); n_rev += 1
        else:
            nd.add((a, b))
    for a, b in u:
        ta, tb = get_tier(a), get_tier(b)
        if ta < tb:
            nd.add((a, b))
        elif ta > tb:
            nd.add((b, a))
        else:
            nu.add(tuple(sorted((a, b))))
    return {"dir": nd, "und": nu}, n_rev


def adj_pairs(g):
    return {tuple(sorted(e)) for e in g["dir"]} | set(g["und"])


def edge_mark(g, a, b):
    if (a, b) in g["dir"]:
        return ">"
    if (b, a) in g["dir"]:
        return "<"
    if (a, b) in g["und"]:
        return "-"
    return None


def combine_graphs(g1, g2, mode):
    pairs = adj_pairs(g1) | adj_pairs(g2) if mode == "union" else adj_pairs(g1) & adj_pairs(g2)
    out = {"dir": set(), "und": set()}
    for a, b in pairs:
        marks = {m for m in (edge_mark(g1, a, b), edge_mark(g2, a, b)) if m is not None}
        directed = marks - {"-"}
        if len(directed) == 1:
            m = directed.pop()
            out["dir"].add((a, b) if m == ">" else (b, a))
        else:
            out["und"].add((a, b))
    return out


def shd(g1, g2):
    return sum(1 for a, b in adj_pairs(g1) | adj_pairs(g2)
               if edge_mark(g1, a, b) != edge_mark(g2, a, b))


def derive_mb(g, allowed, target=TARGET):
    par = {a for a, b in g["dir"] if b == target}
    chi = {b for a, b in g["dir"] if a == target}
    nbr = {x for e in g["und"] if target in e for x in e} - {target}
    cop = {a for a, b in g["dir"] if b in chi and a != target}
    return sorted(f for f in (par | chi | nbr | cop) - {target} if f in allowed)


def descendants(g, node):
    res, stack = set(), [node]
    while stack:
        x = stack.pop()
        for a, b in g["dir"]:
            if a == x and b not in res:
                res.add(b); stack.append(b)
    return res


def possible_ancestors(g, targets):
    res, stack = set(), list(targets)
    while stack:
        x = stack.pop()
        for a, b in g["dir"]:
            if b == x and a not in res:
                res.add(a); stack.append(a)
        for a, b in g["und"]:
            if x in (a, b):
                o = b if a == x else a
                if o not in res:
                    res.add(o); stack.append(o)
    return res


def run_pc(data, names):
    bk = build_bk(names)
    try:
        cg = pc_algo(data=data, alpha=PC_ALPHA, indep_test=INDEP_TEST, stable=True,
                     uc_rule=0, uc_priority=2, background_knowledge=bk,
                     show_progress=False, node_names=names)
        return cg.G.graph, True
    except Exception:
        cg = pc_algo(data=data, alpha=PC_ALPHA, indep_test=INDEP_TEST, stable=True,
                     uc_rule=0, uc_priority=2, show_progress=False)
        return cg.G.graph, False


def run_ges(data, names):
    try:
        rec = ges_algo(data, score_func="local_score_BIC", node_names=names)
    except TypeError:
        rec = ges_algo(data, score_func="local_score_BIC")
    return rec["G"].graph


def discover(X_sub, y_sub, spec):
    Z = encode_for_discovery(X_sub, y_sub, spec)
    Z = Z.loc[:, Z.std(ddof=0) > 0]
    if TARGET not in Z.columns:
        raise ValueError("Outcome is constant in this subset; cannot run discovery")
    Zs = (Z - Z.mean()) / Z.std(ddof=0)
    keep, vif_dropped = vif_prune(Zs)
    Zs = Zs[keep]
    names = list(Zs.columns)
    data = Zs.values.astype(float)

    G_pc, bk_used = run_pc(data, names)
    g_pc, rev_pc = orient_by_tiers(*extract_edges(G_pc, names))
    ges_ok, ges_err, rev_ges = True, None, 0
    try:
        g_ges, rev_ges = orient_by_tiers(*extract_edges(run_ges(data, names), names))
    except Exception as e:
        ges_ok, ges_err = False, f"{type(e).__name__}: {e}"
        g_ges = {"dir": set(), "und": set()}

    graphs = {"pc_only": g_pc, "ges_only": g_ges,
              "union": combine_graphs(g_pc, g_ges, "union") if ges_ok else g_pc,
              "consensus": combine_graphs(g_pc, g_ges, "intersection") if ges_ok else g_pc}
    allowed = set(names) - {TARGET}
    return dict(names=names, graphs=graphs,
                mb={k: derive_mb(g, allowed) for k, g in graphs.items()},
                bk_used=bk_used, ges_ok=ges_ok, ges_err=ges_err,
                vif_dropped=vif_dropped, reoriented=(rev_pc, rev_ges))


def fallback_feature(X_tr, y_tr, candidates):
    best, best_r = candidates[0], -1
    for c in sorted(candidates):
        v = pd.to_numeric(X_tr[c], errors="coerce")
        if v.notna().sum() > 2 and v.std() > 0:
            r = abs(np.corrcoef(v.fillna(v.median()), y_tr)[0, 1])
            if r > best_r:
                best, best_r = c, r
    return [best]


def cv_task(key, clf, feats, X_tr, y_tr, X_va, y_va, num_all, imbalance, smote_k):
    pipe = make_pipeline(feats, clone(clf), num_all, imbalance, smote_k)
    pipe.fit(X_tr[feats], y_tr)
    p = pipe.predict_proba(X_va[feats])[:, 1]
    return key, score_block(y_va, p, 0.5)


def effect_point(t, y, z):
    D = np.column_stack([np.ones(len(t)), t, z])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    zs = (z - z.mean(0)) / (z.std(0) + 1e-12) if z.shape[1] else z
    F = np.column_stack([t, zs])
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(F, y)
    F1, F0 = F.copy(), F.copy()
    F1[:, 0], F0[:, 0] = 1, 0
    return beta[1], lr.predict_proba(F1)[:, 1].mean(), lr.predict_proba(F0)[:, 1].mean()


def effect_boot(seed, t, y, z):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), len(y))
    tb, yb, zb = t[idx], y[idx], z[idx]
    if tb.min() == tb.max() or yb.min() == yb.max():
        return (np.nan, np.nan, np.nan)
    try:
        return effect_point(tb, yb, zb)
    except Exception:
        return (np.nan, np.nan, np.nan)


def evalue_rr(rr):
    if not np.isfinite(rr) or rr <= 0:
        return np.nan
    rr = 1.0 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))


def evalue_ci(lo, hi):
    if lo <= 1 <= hi:
        return 1.0
    return evalue_rr(lo if lo > 1 else hi)


class ThresholdShiftedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, pipe, thr):
        self.pipe, self.thr = pipe, thr
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = self.pipe.predict_proba(X)[:, 1]
        t = self.thr
        q = np.where(p < t, 0.5 * p / max(t, 1e-9), 0.5 + 0.5 * (p - t) / max(1 - t, 1e-9))
        q = np.clip(q, 0, 1)
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.pipe.predict_proba(X)[:, 1] >= self.thr).astype(int)


def short_label(n):
    s = (str(n).replace("History_of_", "Hx_").replace("Historyof_", "Hx_")
         .replace("_at_admission", "").replace("successful_recanalization", "recan")
         .replace("sICH_bin", "sICH").replace("Onset_to_", "On_")
         .replace("Admission_to_", "Adm_").replace("ASPECTS_Score", "ASPECTS")
         .replace("groinpunture", "groin").replace("_count", ""))
    return s[:10]


def unique_labels(nodes):
    seen, labs = Counter(), {}
    for n in nodes:
        s = short_label(n)
        seen[s] += 1
        labs[n] = s if seen[s] == 1 else f"{s[:8]}{seen[s]}"
    return labs


def main():
    df = load_data(FILE_PATH)
    engineered = [c for c in ENGINEERED_SPEC if c in df.columns]
    print(f"  Clinical ratio features engineered ({len(engineered)}): {engineered}")
    print("  NOTE: Neutrophil_WBC_fraction_at_admission is a proxy for inflammatory")
    print("  signal; a true NLR needs a lymphocyte count, which is not in the dataset.")

    y = df[TARGET]
    X = df.drop(columns=["sICH", TARGET] + LEAKAGE_COLS, errors="ignore")
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    num_all = set(numeric_features)

    unknown_tier = sorted(c for c in X.columns if c not in TIER_MAP and c not in CLINICAL_DOMAIN
                          and c not in ENGINEERED_SPEC)
    if unknown_tier:
        print(f"  [WARN] No tier/domain mapping for {unknown_tier}; assumed admission tier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    n_pos_tr = int(y_train.sum())
    smote_k = max(1, min(5, int(n_pos_tr * (1 - 1 / CV_FOLDS) ** 2) - 1))
    spw = float((y_train == 0).sum() / max(n_pos_tr, 1))
    MODELS = build_models(IMBALANCE, spw)

    divider("TABLE 1 — Dataset")
    print(f"  Total patients       : {len(df)}")
    print(f"  sICH positive (n, %) : {y.sum()} ({y.mean() * 100:.1f}%)")
    print(f"  sICH negative (n, %) : {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f"  Features             : {X.shape[1]} ({len(numeric_features)} numeric, "
          f"{len(categorical_features)} categorical)")
    print(f"  Train / Test         : {len(X_train)} / {len(X_test)}")
    print(f"  CV                   : {CV_REPEATS}x repeated stratified {CV_FOLDS}-fold on TRAIN only")
    print(f"  Imbalance handling   : {IMBALANCE}")
    print(f"  Excluded (leakage)   : {[c for c in LEAKAGE_COLS if c in df.columns]}")
    print("  NOTE: procedural variables (TICI, passes, EVT times) are predictors, so the")
    print("  model predicts sICH at the END of thrombectomy, not at admission.")

    spec, disc_excluded = discovery_spec(X_train, numeric_features, categorical_features)
    train_disc = discover(X_train, y_train, spec)
    g_pc, g_ges = train_disc["graphs"]["pc_only"], train_disc["graphs"]["ges_only"]
    g_union, g_cons = train_disc["graphs"]["union"], train_disc["graphs"]["consensus"]
    mb_features = train_disc["mb"][MB_STRATEGY]
    mb_consensus = train_disc["mb"]["consensus"]
    if not mb_features:
        mb_features = fallback_feature(X_train, y_train, train_disc["names"][:-1])
        print(f"  [WARN] Empty MB; falling back to most correlated feature {mb_features}")
    SHD = shd(g_pc, g_ges) if train_disc["ges_ok"] else np.nan

    Z_tr = encode_for_discovery(X_train, y_train, spec)
    def rank_abs_r(cands):
        r = {c: abs(np.corrcoef(Z_tr[c], Z_tr[TARGET])[0, 1]) for c in cands if Z_tr[c].std() > 0}
        return sorted(r, key=lambda c: (-r[c], c))
    treatment_source = None
    if TREATMENT_OVERRIDE:
        TREATMENT, treatment_source = TREATMENT_OVERRIDE, "user override"
    else:
        TREATMENT = None
        for label, pool, modifiable_only in [
                ("consensus neighbour (modifiable)", mb_consensus, True),
                (f"{MB_STRATEGY} MB (modifiable)", mb_features, True),
                ("consensus neighbour", mb_consensus, False),
                (f"{MB_STRATEGY} MB", mb_features, False)]:
            cands = [c for c in pool if c in Z_tr.columns and (not modifiable_only or c not in NON_MODIFIABLE)]
            if cands:
                TREATMENT, treatment_source = rank_abs_r(cands)[0], label
                break
        if TREATMENT is None:
            TREATMENT = rank_abs_r([c for c in Z_tr.columns if c != TARGET])[0]
            treatment_source = "highest |r| (no graph neighbours)"
    if TREATMENT in NON_MODIFIABLE:
        print(f"  [WARN] Treatment {TREATMENT} is not modifiable; interpret the 'effect' accordingly.")

    divider("TABLE 3 — Causal Discovery Summary (training data only)")
    print(f"  Variables in discovery            : {len(train_disc['names']) - 1}")
    for c, why in sorted(disc_excluded.items()):
        print(f"    excluded {c:<38} {why}")
    if train_disc["vif_dropped"]:
        print(f"    VIF-pruned (VIF>{VIF_MAX:g})              : {train_disc['vif_dropped']}")
    print(f"  Temporal knowledge                : "
          f"{'PC BackgroundKnowledge + tier re-orientation' if train_disc['bk_used'] else 'tier re-orientation only (BK unsupported)'}")
    print(f"  Cross-tier edges re-oriented       : PC={train_disc['reoriented'][0]}, GES={train_disc['reoriented'][1]}")
    print(f"  PC  (alpha={PC_ALPHA}, {INDEP_TEST})   : {len(g_pc['dir'])} directed, {len(g_pc['und'])} undirected")
    if train_disc["ges_ok"]:
        print(f"  GES (BIC)                         : {len(g_ges['dir'])} directed, {len(g_ges['und'])} undirected")
    else:
        print(f"  GES                               : FAILED ({train_disc['ges_err']}) — PC used alone")
    print(f"  SHD (PC vs GES, CPDAG marks)      : {SHD}")
    print(f"  Consensus adjacencies (PC ∩ GES)  : {len(adj_pairs(g_cons))}")
    print(f"  Union adjacencies (PC ∪ GES)      : {len(adj_pairs(g_union))}")
    print(f"  Consensus neighbours of sICH      : {mb_consensus}")
    print(f"  MB strategy                       : {MB_STRATEGY}")
    print(f"  Markov blanket ({len(mb_features)})               : {mb_features}")
    print(f"  Treatment                         : {TREATMENT}  [{treatment_source}]")
    print("  NOTE: sICH is in the final tier, so it can have no children; the MB")
    print("  equals the set of variables adjacent to sICH.")

    divider("TABLE 3b — Markov Blanket by Clinical Domain")
    doms = defaultdict(list)
    for f in mb_features:
        doms[get_clinical_domain(f)].append(f)
    for d, fs in sorted(doms.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"  {d:<28} ({len(fs)}): {fs}")
    print(f"\n  {len(mb_features)} MB features span {len(doms)} clinical domain(s).")

    rskf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    folds = list(rskf.split(X_train, y_train))
    J = len(folds)
    n_tr_fold, n_va_fold = len(folds[0][0]), len(folds[0][1])
    print(f"\nRunning causal discovery inside {J} training folds (nested)...")
    fold_disc = Parallel(n_jobs=N_JOBS)(
        delayed(discover)(X_train.iloc[tr], y_train.iloc[tr], spec) for tr, _ in folds)

    strategies = ["consensus", "union", "pc_only", "ges_only"]

    def fold_feats(mode, j):
        if mode == "full":
            return X.columns.tolist()
        f = fold_disc[j]["mb"][mode.split(":")[1]]
        if f:
            return f
        tr = folds[j][0]
        return fallback_feature(X_train.iloc[tr], y_train.iloc[tr], fold_disc[j]["names"][:-1])

    modes = ["full"] + [f"mb:{s}" for s in strategies]
    tasks = []
    for mode in modes:
        for name, clf in MODELS.items():
            for j, (tr, va) in enumerate(folds):
                tasks.append(((mode, name, j), clf, fold_feats(mode, j), tr, va))
    for j, (tr, va) in enumerate(folds):
        tasks.append((("full", "Baseline", j), DummyClassifier(strategy="prior"),
                      X.columns.tolist(), tr, va))

    print(f"Fitting {len(tasks)} CV fits ({len(MODELS)} models x {len(modes)} feature sets x {J} folds)...")

    def run_tasks(task_list):
        res = Parallel(n_jobs=N_JOBS)(
            delayed(cv_task)(key, clf, feats, X_train.iloc[tr], y_train.iloc[tr],
                             X_train.iloc[va], y_train.iloc[va], num_all, IMBALANCE, smote_k)
            for key, clf, feats, tr, va in task_list)
        store = defaultdict(lambda: defaultdict(lambda: [None] * J))
        for (mode, name, j), sc in res:
            store[mode][name][j] = sc
        return store

    cv = run_tasks(tasks)
    proposed_mode = f"mb:{MB_STRATEGY}"

    def cv_summary(mode, names):
        rows = []
        for n in names:
            sc = pd.DataFrame(cv[mode][n])
            rows.append({"Model": n,
                         "CV AUC": sc["AUC"].mean(), "AUC SD": sc["AUC"].std(),
                         "CV PR-AUC": sc["PR_AUC"].mean(), "CV Brier": sc["Brier"].mean(),
                         "CV F1@0.5": sc["F1"].mean(), "F1 SD": sc["F1"].std(),
                         "CV Recall@0.5": sc["Recall"].mean()})
        return pd.DataFrame(rows).round(4)

    sel_col = "CV AUC" if SELECTION_METRIC == "AUC" else "CV PR-AUC"
    df_cv_mb_ind = cv_summary(proposed_mode, list(MODELS))
    top3 = df_cv_mb_ind.sort_values([sel_col, "Model"], ascending=[False, True]).head(3)["Model"].tolist()
    print(f"Soft-voting ensemble members (top 3 by {sel_col}, MB, train CV): {top3}")

    ens_tasks = []
    for mode in ["full", proposed_mode]:
        for j, (tr, va) in enumerate(folds):
            ens_tasks.append(((mode, "Ensemble(Top3)", j), make_ensemble(MODELS, top3),
                              fold_feats(mode, j), tr, va))
    for mode, d in run_tasks(ens_tasks).items():
        cv[mode]["Ensemble(Top3)"] = d["Ensemble(Top3)"]

    all_names = list(MODELS) + ["Ensemble(Top3)"]
    df_cv_full = cv_summary("full", all_names)
    df_cv_mb = cv_summary(proposed_mode, all_names)
    best_full = df_cv_full[df_cv_full.Model.isin(MODELS)].sort_values(
        [sel_col, "Model"], ascending=[False, True]).iloc[0]["Model"]
    PROPOSED = df_cv_mb[df_cv_mb.Model.isin(MODELS)].sort_values(
        [sel_col, "Model"], ascending=[False, True]).iloc[0]["Model"]

    print("\nFitting final models on TRAIN and evaluating once on TEST...")
    results_full, results_mb, fitted_mb = [], [], {}
    for name in all_names:
        clf = make_ensemble(MODELS, top3) if name == "Ensemble(Top3)" else MODELS[name]
        for feats, store in [(X.columns.tolist(), results_full), (mb_features, results_mb)]:
            pipe = make_pipeline(feats, clone(clf), num_all, IMBALANCE, smote_k)
            thr = tune_threshold(pipe, X_train[feats], y_train)
            pipe.fit(X_train[feats], y_train)
            p = pipe.predict_proba(X_test[feats])[:, 1]
            sc = score_block(y_test, p, thr)
            lo, hi = bootstrap_auc_ci(y_test.values, p, N_BOOT_AUC)
            store.append({"Model": name, **{k: round(v, 4) for k, v in sc.items()},
                          "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "Threshold": round(thr, 4),
                          "proba": p})
            if store is results_mb:
                fitted_mb[name] = (pipe, thr)

    df_te_full = pd.DataFrame([{k: v for k, v in r.items() if k != "proba"} for r in results_full])
    df_te_mb = pd.DataFrame([{k: v for k, v in r.items() if k != "proba"} for r in results_mb])

    divider(f"TABLE 2 — {len(all_names)}-Model Comparison: Full vs MB (train CV, nested discovery)")
    section(f"Full features (N={X.shape[1]}) — best by {sel_col}: {best_full}")
    print(df_cv_full.to_string(index=False))
    section(f"MB features ({MB_STRATEGY}; re-derived in each fold) — proposed: {PROPOSED}")
    print(df_cv_mb.to_string(index=False))
    print("  NOTE: the ensemble's CV score is optimistic (members were picked on these folds);")
    print("  its TEST score below is the unbiased one.")
    cols = ["Model", "AUC", "AUC 95% CI", "PR_AUC", "Brier", "F1", "Recall", "Precision", "Threshold"]
    print("\n  TEST set (threshold = F1-optimal on out-of-fold TRAIN predictions):")
    print("\n  Full features:")
    print(df_te_full[cols].to_string(index=False))
    print(f"\n  MB features ({mb_features}):")
    print(df_te_mb[cols].to_string(index=False))
    save_table(df_cv_full, "table2_cv_full.csv"); save_table(df_cv_mb, "table2_cv_mb.csv")
    save_table(df_te_full[cols], "table2_test_full.csv"); save_table(df_te_mb[cols], "table2_test_mb.csv")

    Z_all = encode_for_discovery(X, y, spec)
    t_raw_obs = X[TREATMENT].map(spec[2][TREATMENT]) if TREATMENT in spec[2] else pd.to_numeric(X[TREATMENT], errors="coerce")
    keep_rows = t_raw_obs.notna().values
    n_drop_t = int((~keep_rows).sum())
    Z_eff = Z_all.loc[keep_rows].reset_index(drop=True)
    t_raw = t_raw_obs[keep_rows].reset_index(drop=True).astype(float)

    if t_raw.nunique() <= 2:
        T = (t_raw == t_raw.max()).astype(int)
        if TREATMENT in spec[2]:
            inv = {v: k for k, v in spec[2][TREATMENT].items()}
            t_label = f"{TREATMENT} = '{inv[t_raw.max()]}' vs '{inv[t_raw.min()]}'"
        else:
            t_label = f"{TREATMENT} = {t_raw.max():g} vs {t_raw.min():g}"
        q_used = None
    else:
        for q_used in [0.33, 0.50, 0.25, 0.75]:
            cut = t_raw.quantile(q_used)
            T = (t_raw > cut).astype(int)
            if T.nunique() == 2 and min(T.mean(), 1 - T.mean()) >= 0.10:
                break
        t_label = f"{TREATMENT} > {cut:.4g} (above {q_used:.0%} percentile) vs ≤"

    graph = g_union
    desc_T = descendants(graph, TREATMENT)
    anc = possible_ancestors(graph, [TREATMENT, TARGET])
    adjust = sorted(v for v in anc if v not in desc_T and v not in (TREATMENT, TARGET)
                    and get_tier(v) <= get_tier(TREATMENT) and v in Z_eff.columns)
    same_tier_und = sorted(v for v in adjust if get_tier(v) == get_tier(TREATMENT))

    df_eff = Z_eff[adjust].copy()
    df_eff[TREATMENT] = T.values
    df_eff[TARGET] = Z_eff[TARGET].astype(int).values

    cm = CausalModel(data=df_eff, treatment=TREATMENT, outcome=TARGET,
                     common_causes=adjust if adjust else None)
    estimand = cm.identify_effect(proceed_when_unidentifiable=True)
    est_lr = cm.estimate_effect(estimand, method_name="backdoor.linear_regression", target_units="ate")
    try:
        est_ipw = cm.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting",
                                     target_units="ate", method_params={"weighting_scheme": "ips_weight"})
        ipw_val = float(est_ipw.value)
    except Exception as e:
        ipw_val = np.nan
        print(f"  [WARN] IPW estimator failed: {type(e).__name__}: {e}")

    refs = {}
    for label, kw in [("Random common cause", dict(method_name="random_common_cause",
                                                    num_simulations=N_REFUTE_SIMS, random_seed=RANDOM_STATE)),
                      ("Placebo treatment", dict(method_name="placebo_treatment_refuter", placebo_type="permute",
                                                 num_simulations=N_REFUTE_SIMS, random_seed=RANDOM_STATE)),
                      ("Data subset (80%)", dict(method_name="data_subset_refuter", subset_fraction=0.8,
                                                 num_simulations=N_REFUTE_SIMS, random_seed=RANDOM_STATE))]:
        try:
            r = cm.refute_estimate(estimand, est_lr, **kw)
            rr = getattr(r, "refutation_result", None)
            pval = rr.get("p_value") if isinstance(rr, dict) else np.nan
            refs[label] = (float(r.new_effect), float(pval) if pval is not None else np.nan)
        except Exception as e:
            refs[label] = (np.nan, np.nan)
            print(f"  [WARN] {label} refuter failed: {type(e).__name__}: {e}")

    tv, yv, zv = T.values.astype(float), df_eff[TARGET].values.astype(float), df_eff[adjust].values.astype(float)
    ate_lpm, r1, r0 = effect_point(tv, yv, zv)
    boots = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(effect_boot)(RANDOM_STATE + b, tv, yv, zv) for b in range(N_BOOT_EFFECT)), float)
    boots = boots[~np.isnan(boots).any(axis=1)]
    ci_ate = np.percentile(boots[:, 0], [2.5, 97.5])
    rr_point = r1 / r0 if r0 > 0 else np.nan
    rr_boot = boots[:, 1] / np.where(boots[:, 2] > 0, boots[:, 2], np.nan)
    rr_boot = rr_boot[np.isfinite(rr_boot)]
    ci_rr = np.percentile(rr_boot, [2.5, 97.5])
    ev_point, ev_ci = evalue_rr(rr_point), evalue_ci(*ci_rr)

    ate_dowhy = float(est_lr.value)
    direction = "lower" if ate_dowhy < 0 else "higher"
    divider("TABLE 4 — Average Treatment Effect + Refutation")
    print(f"  Treatment contrast   : {t_label}")
    print(f"  Outcome              : sICH (1 = yes)")
    print(f"  Rows used            : {len(df_eff)} ({n_drop_t} dropped for missing treatment)")
    print(f"  Adjustment set ({len(adjust)}) : {adjust}")
    print("  (disjunctive-cause criterion on the TRAIN union graph: possible ancestors of")
    print("   T or Y, not in a later tier than T, not descendants of T)")
    if same_tier_und:
        print(f"  NOTE: {same_tier_und} share T's tier; their order relative to T is unknown.")
    print(f"\n  ATE, linear probability (DoWhy) : {ate_dowhy:+.4f}  "
          f"(bootstrap 95% CI [{ci_ate[0]:+.4f}, {ci_ate[1]:+.4f}], N={len(boots)})")
    print(f"  ATE, IPW (secondary)            : {ipw_val:+.4f}")
    print(f"  Interpretation                  : the treated group has {abs(ate_dowhy) * 100:.1f} percentage "
          f"points {direction} adjusted sICH risk")
    print("\n  Refutations            new effect    p-value   check")
    rel = lambda v: abs(v - ate_dowhy) / max(abs(ate_dowhy), 1e-9)
    near0 = lambda v: abs(v) < max(0.10 * abs(ate_dowhy), 0.005)
    checks = {
        "Random common cause": lambda v, p: p >= 0.05 if np.isfinite(p) else rel(v) < 0.10,
        "Placebo treatment":   lambda v, p: near0(v),
        "Data subset (80%)":   lambda v, p: p >= 0.05 if np.isfinite(p) else rel(v) < 0.10,
    }
    passes = []
    for k, (v, p) in refs.items():
        ok = bool(checks[k](v, p)) if np.isfinite(v) else False
        passes.append(ok)
        print(f"    {k:<20} {v:>+10.4f}   {p:>8.3f}   {'pass' if ok else 'CHECK'}")
    print("  (pass = estimate stable under random confounder / subsetting, and")
    print("   placebo effect < 10% of |ATE|; placebo p-value shown for reference only)")
    ci_excl0 = not (ci_ate[0] <= 0 <= ci_ate[1])
    print(f"\n  Verdict: {'ROBUST' if all(passes) and ci_excl0 else 'NOT ROBUST / CHECK'} "
          f"(all refutations pass: {all(passes)}; CI excludes 0: {ci_excl0})")

    cate_cov = sorted(v for v in Z_eff.columns if v not in desc_T and v not in (TREATMENT, TARGET)
                      and get_tier(v) <= get_tier(TREATMENT))
    if not cate_cov:
        cate_cov = adjust
    Xc = Z_eff[cate_cov].values.astype(float)
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE),
        model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE),
        discrete_treatment=True, n_estimators=500, min_samples_leaf=5,
        cv=CV_FOLDS, random_state=RANDOM_STATE)
    cf.fit(yv, tv.astype(int), X=Xc)
    cate_vals = np.ravel(cf.effect(Xc))
    cate_lb, cate_ub = (np.ravel(a) for a in cf.effect_interval(Xc, alpha=0.05))
    try:
        cf_ate_lo, cf_ate_hi = (float(np.ravel(a)[0]) for a in cf.ate_interval(Xc, alpha=0.05))
    except Exception:
        cf_ate_lo = cf_ate_hi = np.nan

    divider("TABLE 5 — Conditional Average Treatment Effect (CausalForestDML)")
    print(f"  Treatment contrast : {t_label}")
    print(f"  Effect modifiers   : {len(cate_cov)} pre-treatment variables")
    print(f"  N                  : {len(cate_vals)}")
    for stat, val in [("Mean CATE", cate_vals.mean()), ("SD CATE", cate_vals.std()),
                      ("Min CATE", cate_vals.min()), ("Median CATE", np.median(cate_vals)),
                      ("Max CATE", cate_vals.max())]:
        print(f"  {stat:<20} {val:>+10.4f}")
    print(f"  Forest ATE 95% CI  : [{cf_ate_lo:+.4f}, {cf_ate_hi:+.4f}]")
    print(f"  CATE < 0 (point)                 : {(cate_vals < 0).mean() * 100:.1f}%")
    print(f"  CI entirely below 0 (risk lower) : {(cate_ub < 0).mean() * 100:.1f}%")
    print(f"  CI entirely above 0 (risk higher): {(cate_lb > 0).mean() * 100:.1f}%")
    print("  Negative CATE = lower sICH risk in the treated (T=1) group for that patient.")

    dice_pipe, dice_thr = fitted_mb[PROPOSED]
    mb_num = [f for f in mb_features if f in num_all]
    mb_cat = [f for f in mb_features if f not in num_all]
    vary = [f for f in mb_num if f not in DICE_EXCLUDE]
    cf_changes, n_attempted, n_success, dice_errors = [], 0, 0, Counter()
    if not vary:
        print("\n  [INFO] No actionable numeric MB features; DiCE skipped (non-actionable")
        print("  features are deliberately NOT varied).")
    else:
        med = X_train[mb_num].median()
        mode_cat = {c: (X_train[c].mode().iloc[0] if X_train[c].notna().any() else "Missing") for c in mb_cat}
        def impute_for_dice(frame):
            f = frame[mb_features].copy()
            f[mb_num] = f[mb_num].astype(float).fillna(med)
            for c in mb_cat:
                f[c] = f[c].astype(object).where(f[c].notna(), mode_cat[c]).astype(str)
            return f
        d_train = impute_for_dice(X_train)
        d_train[TARGET] = y_train.values
        permitted = {f: [float(X_train[f].quantile(0.01)), float(X_train[f].quantile(0.99))] for f in vary}
        d_obj = dice_ml.Data(dataframe=d_train, continuous_features=mb_num, outcome_name=TARGET)
        m_obj = dice_ml.Model(model=ThresholdShiftedModel(dice_pipe, dice_thr), backend="sklearn",
                              model_type="classifier")
        explainer = Dice(d_obj, m_obj, method="random")

        q_all = impute_for_dice(X_test)
        flagged = dice_pipe.predict_proba(q_all[mb_features])[:, 1] >= dice_thr
        pos_idx = [i for i, (yy, fl) in zip(X_test.index, zip(y_test.values, flagged)) if yy == 1 and fl]
        for idx in pos_idx[:N_DICE_PATIENTS]:
            n_attempted += 1
            query = q_all.loc[[idx]]
            try:
                res = explainer.generate_counterfactuals(
                    query, total_CFs=3, desired_class=0, features_to_vary=vary,
                    permitted_range=permitted, random_seed=RANDOM_STATE, verbose=False)
                cfs = res.cf_examples_list[0].final_cfs_df
                if cfs is None or len(cfs) == 0:
                    dice_errors["no counterfactual found"] += 1
                    continue
                cfs = cfs[mb_features].copy()
                cfs[mb_num] = cfs[mb_num].astype(float)
                valid = dice_pipe.predict_proba(cfs)[:, 1] < dice_thr
                if not valid.any():
                    dice_errors["counterfactuals failed re-validation"] += 1
                    continue
                n_success += 1
                orig = query.iloc[0]
                raw_orig = X_test.loc[idx]
                for k, row in cfs[valid].reset_index(drop=True).iterrows():
                    for f in vary:
                        if pd.isna(raw_orig[f]):
                            continue
                        delta = float(row[f]) - float(orig[f])
                        if abs(delta) > 1e-3:
                            cf_changes.append({"feature": f, "delta": delta, "orig": float(orig[f]),
                                               "cf": float(row[f]), "patient": idx, "cf_idx": k})
            except Exception as e:
                dice_errors[f"{type(e).__name__}: {str(e)[:80]}"] += 1

    feat_pat, feat_d = defaultdict(set), defaultdict(list)
    for c in cf_changes:
        feat_pat[c["feature"]].add(c["patient"])
        feat_d[c["feature"]].append(c["delta"])
    freq = {f: len(p) for f, p in feat_pat.items()}
    top_feats = sorted(freq, key=lambda f: (-freq[f], f))[:10]
    feat_delta = {f: float(np.mean(feat_d[f])) for f in top_feats}
    sd_tr = X_train[vary].std() if vary else pd.Series(dtype=float)

    divider("TABLE 7 — Counterfactual Explanations (DiCE, proposed MB model)")
    print(f"  Model / threshold     : {PROPOSED} / {dice_thr:.3f}")
    print(f"  Features allowed to vary: {vary}")
    print(f"  True positives flagged by the model and attempted: {n_attempted}; "
          f"with ≥1 valid counterfactual: {n_success}")
    for msg, n in dice_errors.items():
        print(f"    failed ({n}): {msg}")
    if top_feats:
        print(f"\n  {'Feature':<35} {'Pts':>4} {'%':>5} {'Mean Δ':>10} {'Mean Δ (SD)':>12}  Direction")
        rows7 = []
        for f in top_feats:
            d = feat_delta[f]
            dsd = d / sd_tr[f] if sd_tr.get(f, 0) else np.nan
            print(f"  {f:<35} {freq[f]:>4} {freq[f] / max(n_success, 1) * 100:>4.0f}% {d:>10.3f} {dsd:>12.2f}  "
                  f"{'Decrease' if d < 0 else 'Increase'}")
            rows7.append({"Feature": f, "Patients": freq[f], "MeanDelta": d, "MeanDeltaSD": dsd})
        save_table(pd.DataFrame(rows7), "table7_dice.csv")
    print("  NOTE: counterfactuals show what changes the MODEL's prediction; they are not")
    print("  causal intervention effects.")

    divider("TABLE 8 — E-value Sensitivity Analysis")
    print(f"  Adjusted risk (g-computation) : T=1 {r1:.4f} | T=0 {r0:.4f} | RD {r1 - r0:+.4f}")
    print(f"  Risk ratio                    : {rr_point:.3f}  (bootstrap 95% CI [{ci_rr[0]:.3f}, {ci_rr[1]:.3f}])")
    print(f"  E-value (point)               : {ev_point:.3f}")
    print(f"  E-value (CI limit nearest 1)  : {ev_ci:.3f}"
          + ("  ← CI includes RR=1" if ci_rr[0] <= 1 <= ci_rr[1] else ""))
    print(f"  Unmeasured confounding would need RR ≥ {ev_point:.2f} with both T and sICH to explain")
    print(f"  away the point estimate, and RR ≥ {ev_ci:.2f} to move the CI to include the null.")

    def fold_metric(mode, name, metric):
        return np.array([s[metric] for s in cv[mode][name]], float)

    def compare(mode_a, mode_b, metric, names):
        rows = []
        for n in names:
            t = corrected_ttest(fold_metric(mode_a, n, metric) - fold_metric(mode_b, n, metric),
                                n_tr_fold, n_va_fold)
            rows.append({"Model": n, **t})
        avg = np.mean([fold_metric(mode_a, n, metric) - fold_metric(mode_b, n, metric) for n in names], axis=0)
        overall = corrected_ttest(avg, n_tr_fold, n_va_fold)
        wins = sum(r["mean"] > 0 for r in rows)
        losses = sum(r["mean"] < 0 for r in rows)
        return pd.DataFrame(rows), overall, (wins, len(rows) - wins - losses, losses)

    def mode_summary(mode, names):
        m = {k: np.mean([fold_metric(mode, n, k).mean() for n in names]) for k in ["AUC", "F1", "Recall"]}
        m["AUC_SD"] = np.std([fold_metric(mode, n, "AUC").mean() for n in names])
        return m

    ind = list(MODELS)
    divider(f"TABLE 6 — Ablation ({len(ind)} models x {J} folds, nested discovery, corrected t-test)")
    section("(A) Feature-set ablation (mean over models)")
    abl_rows = [("Baseline (prior)", mode_summary("full", ["Baseline"]))] + \
               [(f"Full ({X.shape[1]} features)", mode_summary("full", ind))] + \
               [(f"MB {s}{' ◄' if s == MB_STRATEGY else ''}", mode_summary(f"mb:{s}", ind)) for s in strategies]
    print(f"  {'Configuration':<28} {'AUC':>7} {'±SD(models)':>12} {'F1@0.5':>8} {'Rec@0.5':>8}")
    for lbl, m in abl_rows:
        print(f"  {lbl:<28} {m['AUC']:>7.4f} {m['AUC_SD']:>12.4f} {m['F1']:>8.4f} {m['Recall']:>8.4f}")
    mb_sizes = {s: np.mean([len(fold_disc[j]["mb"][s]) for j in range(J)]) for s in strategies}
    print("  Mean MB size per fold: " + ", ".join(f"{s}={v:.1f}" for s, v in mb_sizes.items()))
    save_table(pd.DataFrame([{"Configuration": l, **m} for l, m in abl_rows]), "table6_ablation.csv")

    section(f"(B) {proposed_mode} vs Full — per-model ΔAUC (Nadeau–Bengio corrected)")
    per_model, overall, wtl = compare(proposed_mode, "full", "AUC", ind)
    print(per_model.round(4).to_string(index=False))
    print(f"\n  Averaged over models: ΔAUC={overall['mean']:+.4f} "
          f"[{overall['lo']:+.4f}, {overall['hi']:+.4f}], p={overall['p']:.4f}")
    print(f"  Win/Tie/Loss (MB > Full by mean AUC across models): {wtl[0]}/{wtl[1]}/{wtl[2]}")
    _, o_f1, _ = compare(proposed_mode, "full", "F1", ind)
    print(f"  ΔF1@0.5 averaged over models: {o_f1['mean']:+.4f} [{o_f1['lo']:+.4f}, {o_f1['hi']:+.4f}], p={o_f1['p']:.4f}")

    section("(C) MB strategy pairwise comparisons (ΔAUC averaged over models)")
    pair_rows = []
    for a, b in [("union", "consensus"), ("union", "ges_only"), ("union", "pc_only"), ("ges_only", "consensus")]:
        _, o, w = compare(f"mb:{a}", f"mb:{b}", "AUC", ind)
        pair_rows.append({"Comparison": f"{a} vs {b}", "dAUC": o["mean"], "CI_lo": o["lo"], "CI_hi": o["hi"],
                          "p": o["p"], "Win/Tie/Loss": f"{w[0]}/{w[1]}/{w[2]}"})
    print(pd.DataFrame(pair_rows).round(4).to_string(index=False))
    print("  Corrected t-test accounts for overlapping training sets across folds; p-values")
    print("  are not adjusted for the number of comparisons.")

    divider(f"GRAPH VALIDATION — Subsampling Stability ({N_STABILITY} x {STABILITY_FRAC:.0%} of TRAIN, no replacement)")
    rng = np.random.default_rng(RANDOM_STATE)
    sss = []
    pos_i, neg_i = np.where(y_train.values == 1)[0], np.where(y_train.values == 0)[0]
    for _ in range(N_STABILITY):
        sss.append(np.concatenate([rng.choice(pos_i, int(len(pos_i) * STABILITY_FRAC), replace=False),
                                   rng.choice(neg_i, int(len(neg_i) * STABILITY_FRAC), replace=False)]))
    sub_disc = Parallel(n_jobs=N_JOBS)(
        delayed(discover)(X_train.iloc[ix], y_train.iloc[ix], spec) for ix in sss)
    algos = [("PC", "pc_only"), ("GES", "ges_only"), ("Union", "union")]
    jacc = {a: [] for a, _ in algos}
    sel = {a: Counter() for a, _ in algos}
    for sd in sub_disc:
        for a, k in algos:
            ref, got = set(train_disc["mb"][k]), set(sd["mb"][k])
            jacc[a].append(len(ref & got) / len(ref | got) if ref | got else 1.0)
            sel[a].update(got)
    jacc = {a: np.array(v) for a, v in jacc.items()}
    feats_any = sorted({f for a in sel for f in sel[a]},
                       key=lambda f: (-max(sel[a][f] for a in sel), f))
    print(f"  {'Neighbour of sICH':<40} {'PC%':>6} {'GES%':>6} {'Union%':>7}  in MB")
    stab_rows = []
    for f in feats_any:
        vals = [sel[a][f] / N_STABILITY * 100 for a, _ in algos]
        if max(vals) < 10:
            continue
        mk = "  ◄" if f in mb_features else ""
        print(f"  {f:<40} {vals[0]:>6.1f} {vals[1]:>6.1f} {vals[2]:>7.1f}{mk}")
        stab_rows.append({"Feature": f, "PC": vals[0], "GES": vals[1], "Union": vals[2], "InMB": f in mb_features})
    save_table(pd.DataFrame(stab_rows), "stability_selection.csv")
    print(f"\n  Jaccard similarity of subsample MB with the algorithm's own full-TRAIN MB")
    print(f"  {'Algorithm':<8} {'Mean':>6} {'SD':>6} {'Min':>6} {'Max':>6}")
    for a, _ in algos:
        v = jacc[a]
        print(f"  {a:<8} {v.mean():>6.3f} {v.std():>6.3f} {v.min():>6.3f} {v.max():>6.3f}")
    print(f"\n  Paired Wilcoxon on Jaccard (not tautological: union can lose precision)")
    for la, lb in [("Union", "PC"), ("Union", "GES"), ("PC", "GES")]:
        d = jacc[la] - jacc[lb]
        if np.all(d == 0):
            w, p = 0.0, 1.0
        else:
            w, p = stats.wilcoxon(jacc[la], jacc[lb])
        print(f"  {la + ' vs ' + lb:<14} W={w:>8.1f}  p={p:.4f}  mean Δ={d.mean():+.3f}")

    print("\n\nGenerating figures...")
    names = train_disc["names"]
    cons_set, mb_set = set(mb_consensus), set(mb_features)

    def node_col(n):
        if n == TARGET: return PAL[0]
        if n in cons_set: return PAL[1]
        if n in mb_set: return PAL[3]
        return PAL[2]

    G_all = nx.Graph(); G_all.add_nodes_from(names); G_all.add_edges_from(adj_pairs(g_union))
    pos = nx.spring_layout(G_all, seed=RANDOM_STATE, k=2.8 / np.sqrt(max(len(names), 1)))
    labels = unique_labels(names)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
    for ax, g, title in zip(axes, [g_pc, g_ges, g_union],
                            [f"PC (α={PC_ALPHA})", "GES (BIC)" if train_disc["ges_ok"] else "GES (failed)",
                             f"Union PC ∪ GES ({len(adj_pairs(g_union))} adjacencies)"]):
        Gd = nx.DiGraph(); Gd.add_nodes_from(names); Gd.add_edges_from(g["dir"])
        Gu = nx.Graph(); Gu.add_nodes_from(names); Gu.add_edges_from(g["und"])
        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_color=[node_col(n) for n in names], node_size=1200, alpha=0.92)
        nx.draw_networkx_labels(Gd, pos, ax=ax, labels=labels, font_size=5, font_color="white", font_weight="bold")
        if g["dir"]:
            nx.draw_networkx_edges(Gd, pos, ax=ax, edge_color="#264653", arrows=True, arrowsize=10,
                                   connectionstyle="arc3,rad=0.12", width=1.1, alpha=0.65,
                                   min_source_margin=12, min_target_margin=12)
        if g["und"]:
            nx.draw_networkx_edges(Gu, pos, ax=ax, edgelist=list(g["und"]), edge_color=PAL[6],
                                   style="dashed", width=1.3)
        ax.set_facecolor(BG); ax.axis("off"); ax.set_title(title, fontweight="bold", pad=8)
    fig.legend(handles=[mpatches.Patch(color=PAL[0], label="sICH (target)"),
                        mpatches.Patch(color=PAL[1], label="Consensus neighbour of sICH"),
                        mpatches.Patch(color=PAL[3], label=f"MB member ({MB_STRATEGY})"),
                        mpatches.Patch(color=PAL[2], label="Other variable"),
                        Line2D([0], [0], color=PAL[6], ls="--", lw=1.5, label="Undirected edge")],
               loc="lower center", ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out("fig1_dag.png"), dpi=DPI, facecolor=BG, bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax, res, title, hl in [
            (axes[0], results_full, f"Full ({X.shape[1]}) — CV-best: {best_full}", best_full),
            (axes[1], results_mb, f"MB ({len(mb_features)}) — proposed: {PROPOSED}", PROPOSED)]:
        d = pd.DataFrame([{k: v for k, v in r.items() if k != "proba"} for r in res]).sort_values("F1")
        bars = ax.barh(d["Model"], d["F1"], color=[PAL[0] if m == hl else PAL[1] for m in d["Model"]],
                       edgecolor="white", alpha=0.88, height=0.6)
        for bar, auc in zip(bars, d["AUC"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"AUC={auc:.3f}",
                    va="center", fontsize=7, color="#444")
        ax.set_xlabel("Test F1 (tuned threshold)"); ax.set_title(title, fontweight="bold"); ax.set_xlim(0, 1.15)
        ax.legend(handles=[mpatches.Patch(color=PAL[0], label=f"Selected on train CV: {hl}")],
                  loc="lower right", fontsize=8)
    for ax, res, title, hl in [(axes[2], results_full, "ROC — Full features (test)", best_full),
                               (axes[3], results_mb, f"ROC — MB features (test)", PROPOSED)]:
        for i, r in enumerate(res):
            fpr, tpr, _ = roc_curve(y_test, r["proba"])
            ax.plot(fpr, tpr, color=PAL[i % len(PAL)], lw=2.5 if r["Model"] == hl else 1,
                    label=f"{r['Model']} ({r['AUC']:.3f})", alpha=0.85)
        ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontweight="bold"); ax.legend(fontsize=6.5, loc="lower right")
    plt.savefig(out("fig2_model_comparison.png"), dpi=DPI, facecolor=BG); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    axes[0].hist(cate_vals, bins=35, color=PAL[1], edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color=PAL[0], lw=2, ls="--", label="No effect")
    axes[0].axvline(cate_vals.mean(), color=PAL[2], lw=2, label=f"Mean={cate_vals.mean():.3f}")
    axes[0].set_xlabel("CATE (risk difference)"); axes[0].set_ylabel("Patients")
    axes[0].set_title(f"(A) CATE — {short_label(TREATMENT)} (T=1 vs T=0)", fontweight="bold"); axes[0].legend()
    si = np.argsort(cate_vals); xp = np.arange(len(si))
    axes[1].fill_between(xp, cate_lb[si], cate_ub[si], alpha=0.2, color=PAL[1], label="95% CI")
    axes[1].plot(xp, cate_vals[si], color=PAL[2], lw=1.5, label="CATE")
    axes[1].axhline(0, color=PAL[0], lw=1.5, ls="--", label="No effect")
    axes[1].set_xlabel("Patient (sorted)"); axes[1].set_ylabel("CATE")
    axes[1].set_title("(B) Individual CATE with 95% CI", fontweight="bold"); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out("fig3_cate.png"), dpi=DPI, facecolor=BG); plt.close()

    rr_star = 1 / rr_point if rr_point < 1 else rr_point
    ci_star = sorted([1 / c if rr_point < 1 else c for c in ci_rr])
    rr_rng = np.linspace(1.001, max(5, rr_star * 1.5, ci_star[1] * 1.2), 400)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.plot(rr_rng, [evalue_rr(r) for r in rr_rng], color=PAL[2], lw=2, label="E-value(RR)")
    ax.axvspan(max(ci_star[0], 1), ci_star[1], alpha=0.15, color=PAL[1], label="95% CI of RR")
    ax.axvline(rr_star, color=PAL[0], ls="--", lw=2, label=f"Observed RR{'⁻¹' if rr_point < 1 else ''}={rr_star:.2f}")
    ax.scatter([rr_star], [ev_point], color=PAL[0], zorder=5, s=70)
    ax.axhline(ev_ci, color=PAL[1], ls=":", lw=1.5, label=f"E-value for CI = {ev_ci:.2f}")
    ax.set_xlabel("Risk ratio (oriented ≥ 1)"); ax.set_ylabel("E-value")
    ax.set_title("E-value sensitivity (g-computation RR, bootstrap CI)", fontweight="bold")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out("fig4_evalue.png"), dpi=DPI, facecolor=BG); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax, res, title, hl in [(axes[0], results_full, "Calibration — Full (test)", best_full),
                               (axes[1], results_mb, "Calibration — MB (test)", PROPOSED)]:
        ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1.5, label="Perfect")
        for i, r in enumerate(res):
            ptrue, ppred = calibration_curve(y_test, r["proba"], n_bins=8, strategy="quantile")
            ax.plot(ppred, ptrue, "o-", color=PAL[i % len(PAL)], lw=2.5 if r["Model"] == hl else 1, ms=4,
                    label=f"{r['Model']} (BS={r['Brier']:.3f})", alpha=0.85)
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction")
        ax.set_title(title, fontweight="bold"); ax.legend(fontsize=6, loc="upper left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(out("fig5_calibration.png"), dpi=DPI, facecolor=BG); plt.close()

    if top_feats:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
        cols6 = [PAL[0] if feat_delta[f] < 0 else PAL[1] for f in top_feats][::-1]
        axes[0].barh(top_feats[::-1], [freq[f] for f in top_feats[::-1]], color=cols6, edgecolor="white")
        axes[0].set_xlabel(f"Patients (of {n_success}) with a valid CF changing it")
        axes[0].set_title("(A) Most frequently changed features", fontweight="bold")
        axes[0].legend(handles=[mpatches.Patch(color=PAL[0], label="Decrease"),
                                mpatches.Patch(color=PAL[1], label="Increase")], fontsize=8)
        axes[1].barh(top_feats[::-1], [feat_delta[f] / sd_tr[f] for f in top_feats[::-1]], color=cols6, edgecolor="white")
        axes[1].axvline(0, color=GRAY, lw=1)
        axes[1].set_xlabel("Mean change (training SD units)")
        axes[1].set_title("(B) Direction and magnitude", fontweight="bold")
        plt.tight_layout(); plt.savefig(out("fig6_counterfactual.png"), dpi=DPI, facecolor=BG); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=BG)
    metrics, mc, w = ["AUC", "F1", "Recall"], [PAL[0], PAL[1], PAL[2]], 0.25
    xa = np.arange(len(abl_rows))
    for i, (m, c) in enumerate(zip(metrics, mc)):
        axes[0].bar(xa + i * w, [r[m] for _, r in abl_rows], w, color=c, alpha=0.85,
                    label=m if m == "AUC" else f"{m}@0.5")
    axes[0].set_xticks(xa + w); axes[0].set_xticklabels([l for l, _ in abl_rows], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05); axes[0].set_title("(A) Feature-set ablation (train CV)", fontweight="bold")
    axes[0].legend(fontsize=8)
    pr = pd.DataFrame(pair_rows)
    axes[1].errorbar(pr["dAUC"], np.arange(len(pr)), xerr=[pr["dAUC"] - pr["CI_lo"], pr["CI_hi"] - pr["dAUC"]],
                     fmt="o", color=PAL[2], capsize=4)
    axes[1].axvline(0, color=GRAY, ls="--"); axes[1].set_yticks(np.arange(len(pr)))
    axes[1].set_yticklabels(pr["Comparison"]); axes[1].set_xlabel("ΔAUC (corrected 95% CI)")
    axes[1].set_title("(B) MB strategy comparisons", fontweight="bold")
    pm = per_model.sort_values("mean")
    axes[2].errorbar(pm["mean"], np.arange(len(pm)), xerr=[pm["mean"] - pm["lo"], pm["hi"] - pm["mean"]],
                     fmt="o", color=PAL[0], capsize=4)
    axes[2].axvline(0, color=GRAY, ls="--"); axes[2].set_yticks(np.arange(len(pm)))
    axes[2].set_yticklabels(pm["Model"]); axes[2].set_xlabel(f"ΔAUC, MB({MB_STRATEGY}) − Full")
    axes[2].set_title("(C) Per-model MB vs Full", fontweight="bold")
    plt.tight_layout(); plt.savefig(out("fig7_ablation.png"), dpi=DPI, facecolor=BG); plt.close()

    if stab_rows:
        sr = pd.DataFrame(stab_rows).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(sr) + 1)), facecolor=BG)
        yy = np.arange(len(sr))
        for i, (a, c) in enumerate(zip(["PC", "GES", "Union"], [PAL[2], PAL[4], PAL[1]])):
            ax.barh(yy + (i - 1) * 0.27, sr[a], 0.27, color=c, label=a)
        ax.set_yticks(yy); ax.set_yticklabels([f"{f}{' ◄' if m else ''}" for f, m in zip(sr["Feature"], sr["InMB"])], fontsize=8)
        ax.set_xlabel("Selection frequency as sICH neighbour (%)"); ax.set_xlim(0, 100)
        ax.set_title("Subsampling stability of sICH neighbours (◄ = in MB)", fontweight="bold")
        ax.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out("fig8_stability.png"), dpi=DPI, facecolor=BG); plt.close()

    divider("FINAL SUMMARY")
    te_prop = df_te_mb[df_te_mb.Model == PROPOSED].iloc[0]
    te_full = df_te_full[df_te_full.Model == best_full].iloc[0]
    print(f"  Dataset        : {len(df)} patients, {X.shape[1]} features, {y.sum()} sICH ({y.mean() * 100:.1f}%)")
    print(f"  MB ({MB_STRATEGY:<9}): {mb_features}")
    print(f"  Treatment      : {t_label}")
    print(f"  ATE            : {ate_dowhy:+.4f} [{ci_ate[0]:+.4f}, {ci_ate[1]:+.4f}]  RR={rr_point:.3f}  "
          f"E-value={ev_point:.2f} (CI {ev_ci:.2f})")
    print(f"  Best Full      : {best_full}  test AUC={te_full['AUC']:.3f} {te_full['AUC 95% CI']}  F1={te_full['F1']:.3f}")
    print(f"  Proposed (MB)  : {PROPOSED}  test AUC={te_prop['AUC']:.3f} {te_prop['AUC 95% CI']}  F1={te_prop['F1']:.3f}")
    print(f"  MB vs Full     : ΔAUC={overall['mean']:+.4f} [{overall['lo']:+.4f}, {overall['hi']:+.4f}] "
          f"(train CV, corrected p={overall['p']:.4f})")
    print(f"  Outputs        : figures and CSV tables in '{OUT_DIR}/' ({DPI} DPI)")


if __name__ == "__main__":
    main()
