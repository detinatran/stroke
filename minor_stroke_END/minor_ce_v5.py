"""Version 5: goal-based interventional counterfactual explanations.
Instead of pushing flagged patients just below a risk threshold, the explanation is the smallest intervention on
admission systolic pressure and glucose (DBP following SBP through the structural equation, as in version 4) that
lowers the predicted END risk by at least rho (10/20/30%), offered to EVERY patient with room above the clinical
floors. This removes the dependence on the Youden threshold and addresses the low coverage of threshold-based
explanations. Arms: threshold CE (uncalibrated and recalibrated, as before), goal CE at the three rho values,
the DR-learned policy tree and the SBP rule.
Run with:  python3 -c "import minor_ce_v5; minor_ce_v5.main()"
"""
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import minor_ce as ce                         

ms, cf = ce.ms, ce.cf
import numpy as np                            
import pandas as pd                           
from joblib import Parallel, delayed          
from scipy.special import expit, logit        
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LogisticRegression            
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve   
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold   
from sklearn.preprocessing import OneHotEncoder                 
from sklearn.tree import DecisionTreeRegressor                  
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression

OUT = os.path.join(HERE, "outputs_ce_v5")
DBP = "admissionDBP"
os.makedirs(OUT, exist_ok=True)

SBP, GLU = ce.SBP, ce.GLU
FLOOR_S, FLOOR_G = ce.ACTION[SBP], ce.ACTION[GLU]
STEP_S, STEP_G = ce.STEP[SBP], ce.STEP[GLU]
SEED = cf.SEED

# ---------------------------------------------------------------- settings (pre-specified; edit here, not per run)
INNER_K = 3                                    # inner folds used to learn thresholds / DR scores inside a training fold
CAP = float(os.environ.get("V2_CAP", 30))      # plausibility cap on the SBP drop (mmHg) for the capped variant
ACTION_COST = float(os.environ.get("V2_ACTION_COST", 0.002))   # minimum worthwhile gain per treated patient (0.2 pp)
TREE_RATIO_KIND = os.environ.get("V2_TREE_RATIO", "linear")    # density-ratio learner used to score actions
TREE_COLS = ["admissionSBP", "Blood glucose on admission", "Age", "NIHSS score", "Diabetes", "Hypertension"]
ACTIONS = [("none", 0, 0), ("SBP-10", 10, 0), ("SBP-20", 20, 0), ("SBP-30", 30, 0),
           ("GLU-2", 0, 2), ("SBP-20+GLU-2", 20, 2)]

P_PLAIN = "CE: threshold, uncalibrated"
P_THR = "CE: threshold"
P_G10 = "CE: goal 10%"
P_G20 = "CE: goal 20%"
P_G30 = "CE: goal 30%"
P_TREE = "Tree: DR-learned"
P_RULE = "Rule: SBP -20, floor 140"
POLICIES = [P_PLAIN, P_THR, P_G10, P_G20, P_G30, P_TREE, P_RULE]
CE_POLICIES = [P_PLAIN, P_THR, P_G10, P_G20, P_G30]
CE_VARIANTS = {"xgb": {P_PLAIN: (None, None)},
               "mono_cal_dbp": {P_THR: ("scm", None), P_G10: ("scm", 0.10), P_G20: ("scm", 0.20), P_G30: ("scm", 0.30)}}


# ---------------------------------------------------------------- small utilities
def make_folds(y, groups, k, seed):
    """Stratified folds; with groups, resampled copies of a patient always share a fold."""
    y = np.asarray(y)
    if groups is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, np.asarray(groups)))


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def pct_ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) if len(a) > 1 else (np.nan, np.nan)


def ident(p):
    return p


class Platt:
    """Logistic recalibration p -> expit(a * logit(p) + b), a > 0 so the ranking (and the flag set) never changes."""

    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = a, b

    def __call__(self, p):
        return expit(self.a * logit(np.clip(p, 1e-6, 1 - 1e-6)) + self.b)

    @classmethod
    def fit(cls, p, y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.clip(p, 1e-6, 1 - 1e-6))[:, None], y)
        return cls(max(float(lr.coef_[0, 0]), 1e-3), float(lr.intercept_[0]))


# ---------------------------------------------------------------- risk model with optional monotone constraints
class Risk:
    """XGBoost on a training-fold-imputed frame. mono=True forces risk to be non-decreasing in SBP and glucose."""

    def __init__(self, num, cat, mono, n_jobs):
        self.num, self.cat, self.mono, self.n_jobs = num, cat, mono, n_jobs
        self._thr = None

    def fit(self, Z, y):
        oh = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
        self.ct = ColumnTransformer([("num", "passthrough", self.num), ("cat", oh, self.cat)])
        M = self.ct.fit_transform(Z)
        self.iS, self.iG = self.num.index(SBP), self.num.index(GLU)      # numeric columns come first in M
        self.iD = self.num.index(DBP) if DBP in self.num else None
        cons = None
        if self.mono:
            cons = [0] * M.shape[1]
            cons[self.iS] = cons[self.iG] = 1
            cons = "(" + ",".join(map(str, cons)) + ")"
        self.clf = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5,
                                 min_child_weight=5, reg_lambda=5.0, random_state=SEED, n_jobs=self.n_jobs, eval_metric="logloss",
                                 monotone_constraints=cons).fit(M, y)
        return self

    def transform(self, Z):
        return np.asarray(self.ct.transform(Z), float)

    def raw(self, M):
        return self.clf.predict_proba(M)[:, 1]

    def split_thresholds(self):
        """Split points of the ensemble on SBP and glucose: the model is constant between them. XGBoost sends
        x < split left, so thresholds are shifted just below the split to match the v <= threshold convention used
        by axis_values."""
        if self._thr is None:
            df = self.clf.get_booster().trees_to_dataframe()
            self._thr = {}
            for k in [i for i in (self.iS, self.iG, self.iD) if i is not None]:
                s = df.loc[df.Feature == f"f{k}", "Split"].astype(float).unique()
                self._thr[k] = np.sort(s - 1e-6)
        return self._thr


# ---------------------------------------------------------------- explanation search (exact, on the cell representatives)
def axis_values(x, floor, step, thr, cap):
    """Candidate values for one actionable variable: x itself plus, for every cell of the tree ensemble, the grid point
    nearest to x (the cheapest point of that cell). The published search scans the full grid; the optimum is the same."""
    if x <= floor + 1e-9:
        return np.array([x])
    g = floor + step * np.arange(int(np.ceil((x - floor) / step - 1e-9)))          # grid points below x
    if cap is not None:
        g = g[g >= x - cap - 1e-9]
    if len(g) == 0:
        return np.array([x])
    cid = np.searchsorted(thr, g, side="left")                                      # LightGBM sends v <= threshold left
    keep = np.r_[np.where(np.diff(cid) != 0)[0], len(g) - 1]
    return np.r_[g[keep], x]


def explain_many(model, M, p, t, cal, A_rows, mode, mad, beta=None, rho=None):
    """Smallest MAD-weighted decrease of SBP and glucose that reaches the target risk, per explained row.
    rho None: threshold objective, rows with p >= t are pushed below t.
    rho in (0,1): goal objective, EVERY actionable row gets the smallest change with risk <= (1 - rho) * p (if feasible).
    mode None/'fixed': all other inputs (including DBP, if the model has it) stay as observed.
    mode 'scm': interventional explanation; DBP follows the SBP intervention, DBP' = DBP + beta * (SBP' - SBP)."""
    cap = None
    Ad = A_rows.copy()
    rec, prom = np.zeros(len(p), bool), np.zeros(len(p))
    thr = model.split_thresholds()
    for j in (np.where(p >= t)[0] if rho is None else range(len(p))):
        xs, xg = float(M[j, model.iS]), float(M[j, model.iG])
        if xs <= FLOOR_S + 1e-9 and xg <= FLOOR_G + 1e-9:
            continue                                                                # not actionable
        thr_s = thr[model.iS]
        if mode == "scm":                    # DBP crosses its own split points as SBP is lowered: add them on the SBP axis
            xd = float(M[j, model.iD])
            thr_s = np.sort(np.r_[thr_s, xs + (thr[model.iD] - xd) / beta])
        GS, GG = np.meshgrid(axis_values(xs, FLOOR_S, STEP_S, thr_s, cap),
                             axis_values(xg, FLOOR_G, STEP_G, thr[model.iG], None), indexing="ij")
        GS, GG = GS.ravel(), GG.ravel()
        cand = np.repeat(M[j:j + 1], GS.size, axis=0)
        cand[:, model.iS], cand[:, model.iG] = GS, GG
        if mode == "scm":
            cand[:, model.iD] = xd + beta * (GS - xs)
        pc = cal(model.raw(cand))
        ok = (pc < t) if rho is None else (pc <= (1 - rho) * p[j] + 1e-12)
        if not ok.any():
            continue
        cost = np.abs(GS - xs) / mad[0] + np.abs(GG - xg) / mad[1]
        b = np.lexsort((pc, cost))                                                  # minimum cost, then lowest risk
        b = b[ok[b]][0]
        Ad[j] = (GS[b], GG[b])
        rec[j], prom[j] = True, pc[b] - p[j]
    return Ad, rec, prom


def dbp_slope(Z, num):
    """Additive-noise structural equation for the descendant DBP: DBP = beta*SBP + h(baseline) + u, fitted by least squares
    on the training fold. With u abducted per patient, an SBP intervention moves DBP by beta*(SBP' - SBP)."""
    base = [c for c in num if ms.tier(c) == 0 and c not in (SBP, GLU, DBP)]
    lr = LinearRegression().fit(Z[[SBP] + base].astype(float).values, Z[DBP].astype(float).values)
    return max(float(lr.coef_[0]), 1e-3)


def learn_ce(kind, variants, X, Y, grp, A, tr, te, num, cat, seed, lgbm_jobs):
    """Learn the explanation policy on `tr` only. Training patients receive explanations from inner out-of-fold models
    (so that they behave like unseen patients when the density ratio is trained); held-out patients from the model fitted
    on all of `tr`."""
    mono = recal = kind in ("mono_cal", "mono_cal_dbp")
    if kind == "mono_cal_dbp":
        num = num + [DBP]
    f = ce.prep_fit(X.iloc[tr], num, cat)
    Ztr, Zte = f(X.iloc[tr]), f(X.iloc[te])
    beta = dbp_slope(Ztr, num) if kind == "mono_cal_dbp" else None
    ytr, Atr, Ate = Y[tr], A[tr], A[te]
    mad = np.maximum(np.median(np.abs(Atr - np.median(Atr, axis=0)), axis=0), 1e-9)
    raw, inner_models = np.zeros(len(tr)), []
    for itr, iva in make_folds(ytr, grp[tr], INNER_K, seed):
        m = Risk(num, cat, mono, lgbm_jobs).fit(Ztr.iloc[itr], ytr[itr])
        Mva = m.transform(Ztr.iloc[iva])
        raw[iva] = m.raw(Mva)
        inner_models.append((m, iva, Mva))
    cal = Platt.fit(raw, ytr) if recal else ident
    p_oof = cal(raw)
    t = youden(ytr, p_oof)
    final = Risk(num, cat, mono, lgbm_jobs).fit(Ztr, ytr)
    Mte = final.transform(Zte)
    p_te = cal(final.raw(Mte))
    out = {}
    for name, (mode, rho) in variants.items():
        Ad_te, rec_te, prom_te = explain_many(final, Mte, p_te, t, cal, Ate, mode, mad, beta, rho)
        Ad_tr = Atr.copy()
        for m, iva, Mva in inner_models:
            Ad_tr[iva] = explain_many(m, Mva, p_oof[iva], t, cal, Atr[iva], mode, mad, beta, rho)[0]
        out[name] = dict(Ad_tr=Ad_tr, Ad_te=Ad_te, rec_te=rec_te, promise_te=prom_te)
    return out


# ---------------------------------------------------------------- action menu, DR scores and the policy tree
def action_shifts(A):
    """One shifted exposure matrix per action in ACTIONS (decreases only, never below the clinical floors)."""
    outs = []
    for _, ds, dg in ACTIONS:
        Ad = A.copy()
        if ds:
            Ad[:, 0] = np.where(A[:, 0] > FLOOR_S, np.maximum(A[:, 0] - ds, FLOOR_S), A[:, 0])
        if dg:
            Ad[:, 1] = np.where(A[:, 1] > FLOOR_G, np.maximum(A[:, 1] - dg, FLOOR_G), A[:, 1])
        outs.append(Ad)
    return outs


def dr_scores(A, W, Y, Ads, folds, rf_jobs, r_kind):
    """Cross-fitted doubly robust score of every action for every patient: r_k (Y - m) + m_k, conditionally unbiased
    for E[Y^{action k} | W], so a policy can be chosen by comparing scores. Returns (scores n x K, changed n x K)."""
    n, K = len(Y), len(Ads)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    za = (A - mu) / sd
    zds = [(Ad - mu) / sd for Ad in Ads]
    chg = np.column_stack([np.abs(Ad - A).sum(1) > 1e-9 for Ad in Ads])
    m_obs, m_k, R = np.zeros(n), np.zeros((n, K)), np.ones((n, K))
    for tr, te in folds:
        base_tr = np.hstack([za[tr], W[tr]])
        mo = cf.AvgClf("flexible", rf_jobs).fit(base_tr, Y[tr])
        m_obs[te] = mo.predict(np.hstack([za[te], W[te]]))
        for k in range(K):
            m_k[te, k] = mo.predict(np.hstack([zds[k][te], W[te]]))
            if not chg[:, k].any():
                continue
            Zs = np.vstack([base_tr, np.hstack([zds[k][tr], W[tr]])])
            lab = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
            p = cf.AvgClf(r_kind, rf_jobs).fit(Zs, lab).predict(np.hstack([za[te], W[te]]))
            R[te, k] = p / (1 - p)
    for k in range(K):
        if chg[:, k].any():
            R[:, k] = np.minimum(R[:, k], np.quantile(R[:, k], cf.RATIO_Q))
    return R * (Y[:, None] - m_obs[:, None]) + m_k, chg


class TreePolicy:
    """Depth-2 tree on doubly robust action scores; each leaf receives the action with the lowest mean score plus a
    small per-treated-patient cost (ACTION_COST), so it only treats where the estimated gain is worthwhile."""

    def __init__(self, cols):
        self.cols = cols

    def _feat(self, X):
        return X[self.cols].apply(pd.to_numeric, errors="coerce").astype(float)

    def fit(self, X, A, W, Y, inner_folds, rf_jobs):
        self.cols = [c for c in self.cols if c in X.columns]
        F = self._feat(X)
        self.med = F.median()
        F = F.fillna(self.med).values
        G, chg = dr_scores(A, W, Y, action_shifts(A), inner_folds, rf_jobs, TREE_RATIO_KIND)
        Gp = G + ACTION_COST * chg
        self.tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(40, len(Y) // 15), random_state=SEED).fit(F, Gp)
        leaf = self.tree.apply(F)
        self.leaf_action = {}
        for l in np.unique(leaf):
            m = leaf == l
            score = Gp[m].mean(0)
            for k in range(1, len(ACTIONS)):                    # an action that changes nobody in the leaf is 'none'
                if not chg[m, k].any():
                    score[k] = score[0]
            self.leaf_action[int(l)] = int(np.argmin(score))
        self.train_F, self.train_G, self.train_chg = F, G, chg
        return self

    def apply(self, X, A):
        F = self._feat(X).fillna(self.med).values
        k = np.array([self.leaf_action[int(l)] for l in self.tree.apply(F)])
        Ad, shifts = A.copy(), action_shifts(A)
        for kk in range(len(ACTIONS)):
            Ad[k == kk] = shifts[kk][k == kk]
        return Ad, k

    def leaves(self):
        t = self.tree.tree_
        rules = {}

        def rec(node, conds):
            if t.children_left[node] == -1:
                rules[node] = " & ".join(conds) if conds else "all"
                return
            f, th = self.cols[t.feature[node]], t.threshold[node]
            rec(t.children_left[node], conds + [f"{f} <= {th:.1f}"])
            rec(t.children_right[node], conds + [f"{f} > {th:.1f}"])
        rec(0, [])
        leaf = self.tree.apply(self.train_F)
        rows = []
        for l, rule in rules.items():
            m = leaf == l
            row = dict(rule=rule, n=int(m.sum()), action=ACTIONS[self.leaf_action[l]][0])
            row.update({f"DRscore_{a[0]}": float(self.train_G[m, i].mean()) for i, a in enumerate(ACTIONS)})
            rows.append(row)
        return pd.DataFrame(rows)


def learn_tree(X, Y, grp, A, W, tr, te, seed, rf_jobs):
    tp = TreePolicy(TREE_COLS).fit(X.iloc[tr], A[tr], W[tr], Y[tr], make_folds(Y[tr], grp[tr], INNER_K, seed), rf_jobs)
    Ad_tr, _ = tp.apply(X.iloc[tr], A[tr])
    Ad_te, _ = tp.apply(X.iloc[te], A[te])
    rec = np.abs(Ad_te - A[te]).sum(1) > 1e-9
    return dict(Ad_tr=Ad_tr, Ad_te=Ad_te, rec_te=rec, promise_te=np.zeros(len(te)))


def learn_rule(A, tr, te):
    sh = action_shifts(A)[2]
    rec = np.abs(sh[te] - A[te]).sum(1) > 1e-9
    return dict(Ad_tr=sh[tr], Ad_te=sh[te], rec_te=rec, promise_te=np.zeros(len(te)))


# ---------------------------------------------------------------- one full pass: learn policies in folds, score them causally
def run_once(X, Y, grp, A, W, num, cat, folds, seed, rf_jobs, lgbm_jobs):
    n = len(Y)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    za = (A - mu) / sd
    m_obs = np.zeros(n)
    S = {p: dict(m_shift=np.zeros(n), ratio=np.ones(n), rec=np.zeros(n, bool), prom=np.zeros(n), Ad=A.copy()) for p in POLICIES}
    for k, (tr, te) in enumerate(folds):
        pol = {}
        for kind, variants in CE_VARIANTS.items():
            pol.update(learn_ce(kind, variants, X, Y, grp, A, tr, te, num, cat, seed + k, lgbm_jobs))
        pol[P_TREE] = learn_tree(X, Y, grp, A, W, tr, te, seed + k, rf_jobs)
        pol[P_RULE] = learn_rule(A, tr, te)
        base_tr = np.hstack([za[tr], W[tr]])
        mo = cf.AvgClf("flexible", rf_jobs).fit(base_tr, Y[tr])
        m_obs[te] = mo.predict(np.hstack([za[te], W[te]]))
        for name, d in pol.items():
            s = S[name]
            s["m_shift"][te] = mo.predict(np.hstack([(d["Ad_te"] - mu) / sd, W[te]]))
            if (np.abs(d["Ad_tr"] - A[tr]).sum(1) > 1e-9).any():
                Zs = np.vstack([base_tr, np.hstack([(d["Ad_tr"] - mu) / sd, W[tr]])])
                lab = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
                p = cf.AvgClf("flexible", rf_jobs).fit(Zs, lab).predict(np.hstack([za[te], W[te]]))
                s["ratio"][te] = p / (1 - p)
            s["rec"][te], s["prom"][te], s["Ad"][te] = d["rec_te"], d["promise_te"], d["Ad_te"]
    res = {}
    for name, s in S.items():
        r = s["ratio"]
        if s["rec"].any():
            r = np.minimum(r, np.quantile(r, cf.RATIO_Q))
        phi = r * (Y - m_obs) + s["m_shift"]
        d = phi - Y
        rd, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
        n_rec = int(s["rec"].sum())
        ind = s["m_shift"] - m_obs
        rec = s["rec"]
        corr = np.nan
        if name in CE_POLICIES and n_rec > 2 and np.std(s["prom"][rec]) > 0 and np.std(ind[rec]) > 0:
            corr = float(np.corrcoef(s["prom"][rec], ind[rec])[0, 1])
        res[name] = dict(RD=rd, RD_lo=rd - 1.96 * se, RD_hi=rd + 1.96 * se, n_rec=n_rec,
                         promise=float(s["prom"][rec].mean()) if n_rec and name in CE_POLICIES else np.nan,
                         ESS=float(r.sum() ** 2 / (r ** 2).sum()), Ratio_max=float(r.max()), corr=corr)
    return res


def boot_once(b, X, Y, A, W, num, cat):
    """One resample of the WHOLE procedure (model, threshold, explanations, tree, estimator)."""
    rng = np.random.default_rng(SEED + 7000 + b)
    idx = rng.integers(0, len(Y), len(Y))
    folds = make_folds(Y[idx], idx, cf.FOLDS, SEED + b)
    res = run_once(X.iloc[idx].reset_index(drop=True), Y[idx], idx, A[idx], W[idx], num, cat, folds, SEED + b, 1, 1)
    return {k: dict(RD=v["RD"], n_rec=v["n_rec"], promise=v["promise"],
                    per_rec=v["RD"] * len(Y) / v["n_rec"] if v["n_rec"] else np.nan) for k, v in res.items()}


# ---------------------------------------------------------------- risk-model comparison (discrimination and calibration)
def model_performance(X, Y, num, cat):
    kinds = {"XGBoost": "xgb", "XGBoost monotone + recalibrated": "mono_cal", "XGBoost monotone + recalibrated, with DBP": "mono_cal_dbp"}
    rows, bins = [], []
    for label, kind in kinds.items():
        for r in range(cf.REPEATS):
            oof = np.zeros(len(Y))
            for k, (tr, te) in enumerate(make_folds(Y, None, 5, SEED + r)):
                nk = num + [DBP] if kind == "mono_cal_dbp" else num
                f = ce.prep_fit(X.iloc[tr], nk, cat)
                Ztr, Zte = f(X.iloc[tr]), f(X.iloc[te])
                mono = kind != "xgb"
                m = Risk(nk, cat, mono, 4).fit(Ztr, Y[tr])
                p = m.raw(m.transform(Zte))
                if mono:
                    raw = np.zeros(len(tr))
                    for itr, iva in make_folds(Y[tr], None, INNER_K, SEED + k):
                        mi = Risk(nk, cat, True, 4).fit(Ztr.iloc[itr], Y[tr][itr])
                        raw[iva] = mi.raw(mi.transform(Ztr.iloc[iva]))
                    p = Platt.fit(raw, Y[tr])(p)
                oof[te] = p
            slope, citl = ce.calib(Y, oof)
            rows.append(dict(Model=label, Repeat=r, AUC=roc_auc_score(Y, oof), AUPRC=average_precision_score(Y, oof),
                             Brier=brier_score_loss(Y, oof), Cal_slope=slope, Cal_in_large=citl))
            if r == 0:
                q = pd.qcut(pd.Series(oof), 10, duplicates="drop")
                g = pd.DataFrame(dict(p=oof, y=Y, bin=q)).groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
                bins.append(g.reset_index(drop=True).assign(Model=label))
    perf = pd.DataFrame(rows).groupby("Model")[["AUC", "AUPRC", "Brier", "Cal_slope", "Cal_in_large"]].mean().reindex(list(kinds))
    return perf, pd.concat(bins)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    X, Y = ms.load()
    Y = np.asarray(Y).astype(int)
    n = len(Y)
    feats = [c for c in X.columns if ms.tier(c) <= 2 and c != "admissionDBP"]
    cat = [c for c in feats if c in ms.CATEGORICAL]
    num = [c for c in feats if c not in ms.CATEGORICAL]
    A = X[[SBP, GLU]].astype(float).values
    assert not np.isnan(A).any(), "SBP / glucose contain missing values; the exposure must be complete"
    adj = [c for c in ms.pre_exposure(X, SBP, exclude={"admissionDBP", GLU})]
    W = cf.design(X, adj)
    print(f"n = {n}, END = {Y.sum()}; model features {len(feats)}; adjustment variables {len(adj)}; "
          f"policies: {POLICIES}")

    # 1 -------- risk models
    print("\n[1] RISK MODELS (OOF, 5-fold x %d)" % cf.REPEATS)
    perf, bins = model_performance(X, Y, num, cat)
    print(perf.round(3).to_string())
    perf.to_csv(os.path.join(OUT, "ce_v5_model_performance.csv"))
    bins.to_csv(os.path.join(OUT, "ce_v5_calibration_bins.csv"), index=False)       # decile aggregates only
    print(f"    ({time.time() - t0:.0f}s)")

    # 2 -------- policies learned in folds, scored by the cross-fitted DR estimator
    print("\n[2] POLICIES (learned inside folds, scored on held-out patients)")
    reps = [run_once(X, Y, np.arange(n), A, W, num, cat, make_folds(Y, np.arange(n), cf.FOLDS, SEED + r), SEED + 100 * r,
                     -1, 4) for r in range(cf.REPEATS)]
    point = {}
    for name in POLICIES:
        rr = [r[name] for r in reps]
        point[name] = {k: float(np.nanmedian([x[k] for x in rr])) if not all(np.isnan([x[k] for x in rr])) else np.nan
                       for k in rr[0]}
        p = point[name]
        print(f"  {name:40s}: DR {100 * p['RD']:+.2f} pp | recipients {p['n_rec']:.0f} | ESS {p['ESS']:.0f} | "
              f"promise {100 * p['promise'] if not np.isnan(p['promise']) else float('nan'):+.2f} | r {p['corr']:.2f}")
    print(f"    ({time.time() - t0:.0f}s)")

    # 3 -------- nested bootstrap (the explanation / policy-learning step is refitted in every resample)
    B = cf.B_BOOT
    print(f"\n[3] NESTED BOOTSTRAP ({B} resamples of the whole procedure)")
    draws = Parallel(n_jobs=-1)(delayed(boot_once)(b, X, Y, A, W, num, cat) for b in range(B))
    print(f"    ({time.time() - t0:.0f}s)")

    rows = []
    for name in POLICIES:
        p = point[name]
        rd = np.array([d[name]["RD"] for d in draws])
        bs = cf.boot_summary(rd, p["RD"])
        per_rec = 100 * p["RD"] * n / max(p["n_rec"], 1)
        pr_draw = 100 * np.array([d[name]["per_rec"] for d in draws], float)
        row = dict(Policy=name, Recipients=p["n_rec"], Pct_changed=100 * p["n_rec"] / n, Cohort_RD_pp=100 * p["RD"],
                   Boot_lo_pp=100 * bs["Boot_lo"], Boot_hi_pp=100 * bs["Boot_hi"], IF_lo_pp=100 * p["RD_lo"], IF_hi_pp=100 * p["RD_hi"],
                   Per_recipient_pp=per_rec, PerRec_lo=pct_ci(pr_draw)[0], PerRec_hi=pct_ci(pr_draw)[1],
                   ESS=p["ESS"], Ratio_max=p["Ratio_max"], Corr_model_causal=p["corr"])
        if name in CE_POLICIES:
            prom = 100 * np.array([d[name]["promise"] for d in draws], float)
            over = pr_draw - prom                       # causal minus promise, pp (positive: the model overstated the benefit)
            with np.errstate(divide="ignore", invalid="ignore"):
                fid = np.where(prom < -1e-3, pr_draw / prom, np.nan)
            promise_pp = 100 * p["promise"]
            row.update(Promise_pp=promise_pp, Promise_lo=pct_ci(prom)[0], Promise_hi=pct_ci(prom)[1],
                       Overstatement_pp=per_rec - promise_pp, Over_lo=pct_ci(over)[0], Over_hi=pct_ci(over)[1],
                       Fidelity=per_rec / promise_pp if promise_pp else np.nan, Fid_lo=pct_ci(fid)[0], Fid_hi=pct_ci(fid)[1])
        if name != P_THR:
            dd = 100 * (rd - np.array([d[P_THR]["RD"] for d in draws]))
            row.update(DiffVsThr_pp=100 * (p["RD"] - point[P_THR]["RD"]), Diff_lo=pct_ci(dd)[0], Diff_hi=pct_ci(dd)[1])
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "ce_v5_policies.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.round(2).to_string(index=False))

    # 4 -------- interpretable rule: the tree refitted on the whole cohort (descriptive, not used for inference)
    tp = TreePolicy(TREE_COLS).fit(X, A, W, Y, make_folds(Y, np.arange(n), INNER_K, SEED), -1)
    leaves = tp.leaves()
    leaves.to_csv(os.path.join(OUT, "ce_v5_tree_leaves.csv"), index=False)
    print("\n[4] DR-LEARNED RULE (whole cohort, descriptive)")
    print(leaves[["rule", "n", "action"]].to_string(index=False))

    Zall = ce.prep_fit(X, num + [DBP], cat)(X)
    base = [c for c in num if ms.tier(c) == 0]
    lr_all = LinearRegression().fit(Zall[[SBP] + base].astype(float).values, Zall[DBP].astype(float).values)
    dbp_beta, dbp_r2 = float(lr_all.coef_[0]), float(lr_all.score(Zall[[SBP] + base].astype(float).values, Zall[DBP].astype(float).values))
    print(f"DBP structural equation (whole cohort): beta {dbp_beta:.3f}, R2 {dbp_r2:.3f}")
    json.dump(dict(n=n, B=B, dbp_beta=dbp_beta, dbp_r2=dbp_r2, repeats=cf.REPEATS, inner_k=INNER_K, cap=CAP, action_cost=ACTION_COST, tree_cols=tp.cols,
                   actions=[a[0] for a in ACTIONS], tree_ratio_kind=TREE_RATIO_KIND, policies=tab.to_dict(orient="records")),
              open(os.path.join(OUT, "ce_v5_summary.json"), "w"), indent=1, default=float)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
