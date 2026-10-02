import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
from collections import Counter
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, brier_score_loss, roc_curve,
                             precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.neural_network import MLPClassifier

import sich_prediction_v2 as v2
import sich_learning_curve as lc
import sich_pipeline as sp

FILE_PATH   = os.environ.get("SICH_DATA", "dataset.xlsx")
OUT_DIR     = os.environ.get("SICH_OUT", "outputs_mtl")
QUICK       = os.environ.get("SICH_QUICK", "0") == "1"
OUTER_FOLDS = 3 if QUICK else 5
REPEATS     = int(os.environ.get("SICH_REPEATS", 1 if QUICK else 3))
INNER_FOLDS = 2 if QUICK else 3
MLP_SEEDS   = 2 if QUICK else 5
N_JOBS      = int(os.environ.get("SICH_JOBS", -1))
N_BOOT      = 200 if QUICK else 2000
SEED        = 42
F1_TARGET   = 0.60
DPI         = 300
S, H        = "sICH_bin", "HT_bin"
PAL         = v2.PAL
os.makedirs(OUT_DIR, exist_ok=True)


def out(name):
    return os.path.join(OUT_DIR, name)


def set_tiers():
    sp.TIER_MAP[H] = 4
    sp.TIER_MAP[S] = 5


def load():
    v2.OUTCOME, v2.POS_CODE, v2.NEG_CODE = "sICH", "1", "2"
    df = v2.load_data(FILE_PATH)
    ht = df["Hemorrhagic_transformation"]
    bad = ~ht.isin(["1", "2"])
    if bad.any():
        print(f"  [WARN] {int(bad.sum())} row(s) with HT code(s) {sorted(ht[bad].astype(str).unique())} excluded")
        df = df[~bad]
    ys = df["_y"].astype(int).reset_index(drop=True)
    yh = (df["Hemorrhagic_transformation"] == "1").astype(int).reset_index(drop=True)
    incons = int(((ys == 1) & (yh == 0)).sum())
    if incons:
        print(f"  [WARN] {incons} sICH-positive patient(s) coded HT-negative; set to HT-positive (sICH implies HT)")
        yh[(ys == 1) & (yh == 0)] = 1
    X = df.drop(columns=["_y"] + [c for c in v2.OUTCOME_COLS if c in df.columns]).reset_index(drop=True)
    return X, ys, yh


def prepare(X):
    Xf, new = v2.add_clinical_features(X)
    num = Xf.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in Xf.columns if c not in num]
    for c in cat:
        Xf[c] = Xf[c].astype(object).where(Xf[c].notna(), np.nan)
    Xb = X.copy()
    if "TICI_ordinal" in Xf.columns:
        Xb["TICI_ordinal"] = Xf["TICI_ordinal"]
        Xb = Xb.drop(columns=["TICI"], errors="ignore")
    for c in Xb.columns:
        if c not in Xb.select_dtypes(include=[np.number]).columns:
            Xb[c] = Xb[c].astype(object).where(Xb[c].notna(), np.nan)
    num_b = Xb.select_dtypes(include=[np.number]).columns.tolist()
    cat_b = [c for c in Xb.columns if c not in num_b]
    return Xf, num, cat, new, Xb, num_b, cat_b


def discover2(Xb, ys, yh, num_b, cat_b):
    set_tiers()
    spec, excluded = sp.discovery_spec(Xb, num_b, cat_b)
    Z = sp.encode_for_discovery(Xb, ys, spec)
    Z[H] = np.asarray(yh, float)
    Z = Z.loc[:, Z.std(ddof=0) > 0]
    Zs = (Z - Z.mean()) / Z.std(ddof=0)
    keep, vif_dropped = sp.vif_prune(Zs, protected=(S, H))
    Zs = Zs[keep]
    names = list(Zs.columns)
    data = Zs.values.astype(float)
    G_pc, bk_used = sp.run_pc(data, names)
    g_pc, _ = sp.orient_by_tiers(*sp.extract_edges(G_pc, names))
    try:
        g_ges, _ = sp.orient_by_tiers(*sp.extract_edges(sp.run_ges(data, names), names))
        g = sp.combine_graphs(g_pc, g_ges, "union")
        ges_ok = True
    except Exception:
        g, ges_ok = g_pc, False
    allowed = set(names) - {S, H}
    mb_s = sp.derive_mb(g, allowed, S)
    mb_h = sp.derive_mb(g, allowed, H)
    pa_h = sorted(({a for a, b in g["dir"] if b == H} | {x for e in g["und"] if H in e for x in e}) - {H, S})
    pa_s = sorted(({a for a, b in g["dir"] if b == S} | {x for e in g["und"] if S in e for x in e}) - {H, S})
    return dict(g=g, names=names, mb_s=mb_s, mb_h=mb_h, pa_h=pa_h, pa_s=pa_s,
                causal=sorted(set(mb_s) | set(mb_h)),
                ht_sich_edge=(H, S) in g["dir"] or tuple(sorted((H, S))) in g["und"],
                bk_used=bk_used, ges_ok=ges_ok, excluded=excluded, vif_dropped=vif_dropped)


class Avg3:
    def __init__(self, feats, num_all):
        self.feats, self.num_all = list(feats), set(num_all)

    def fit(self, X, y):
        num = [f for f in self.feats if f in self.num_all]
        cat = [f for f in self.feats if f not in self.num_all]
        y = np.asarray(y)
        spw = float((y == 0).sum() / max(y.sum(), 1))
        self.ms = [m.fit(X[self.feats], y) for m in lc.models(num, cat, spw).values()]
        return self

    def predict(self, X):
        return np.mean([m.predict_proba(X[self.feats])[:, 1] for m in self.ms], axis=0)


class SingleTask:
    def __init__(self, feats, num_all):
        self.feats, self.num_all = feats, num_all

    def fit(self, X, ys, yh):
        self.s = Avg3(self.feats, self.num_all).fit(X, ys)
        self.h = Avg3(self.feats, self.num_all).fit(X, yh)
        return self

    def predict(self, X):
        return self.s.predict(X), self.h.predict(X)


class Hierarchical:
    def __init__(self, feats, num_all):
        self.feats, self.num_all = feats, num_all

    def fit(self, X, ys, yh):
        self.h = Avg3(self.feats, self.num_all).fit(X, yh)
        m = np.asarray(yh) == 1
        self.c = Avg3(self.feats, self.num_all).fit(X[m], np.asarray(ys)[m])
        return self

    def predict(self, X):
        ph = self.h.predict(X)
        return ph * self.c.predict(X), ph


class Chain:
    def __init__(self, feats, num_all):
        self.feats, self.num_all = feats, num_all

    def fit(self, X, ys, yh):
        yh = np.asarray(yh)
        self.h = Avg3(self.feats, self.num_all).fit(X, yh)
        oof = np.zeros(len(X))
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, yh):
            oof[va] = Avg3(self.feats, self.num_all).fit(X.iloc[tr], yh[tr]).predict(X.iloc[va])
        Xa = X.copy()
        Xa["logit_p_HT"] = v2.logit(oof)
        self.s = Avg3(self.feats + ["logit_p_HT"], self.num_all | {"logit_p_HT"}).fit(Xa, ys)
        return self

    def predict(self, X):
        ph = self.h.predict(X)
        Xa = X.copy()
        Xa["logit_p_HT"] = v2.logit(ph)
        return self.s.predict(Xa), ph


class SharedMLP:
    def __init__(self, feats, num_all):
        self.feats, self.num_all = feats, set(num_all)

    def fit(self, X, ys, yh):
        num = [f for f in self.feats if f in self.num_all]
        cat = [f for f in self.feats if f not in self.num_all]
        self.pre = v2.make_pre(num, cat, False).fit(X[self.feats])
        Z = self.pre.transform(X[self.feats])
        Y = np.column_stack([np.asarray(ys), np.asarray(yh)])
        self.nets = [MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, learning_rate_init=1e-3,
                                   max_iter=3000, random_state=SEED + s).fit(Z, Y) for s in range(MLP_SEEDS)]
        return self

    def predict(self, X):
        Z = self.pre.transform(X[self.feats])
        P = np.mean([n.predict_proba(Z) for n in self.nets], axis=0)
        return P[:, 0], P[:, 1]


def approaches(all_feats, causal, num_all):
    return {
        "ST (all features)":           lambda: SingleTask(all_feats, num_all),
        "ST (causal features)":        lambda: SingleTask(causal, num_all),
        "MT-Hierarchical (all)":       lambda: Hierarchical(all_feats, num_all),
        "MT-Hierarchical (causal)":    lambda: Hierarchical(causal, num_all),
        "MT-Chain (all)":              lambda: Chain(all_feats, num_all),
        "MT-SharedMLP (all)":          lambda: SharedMLP(all_feats, num_all),
    }


ENSEMBLE_MEMBERS = ["MT-Hierarchical (all)", "MT-Hierarchical (causal)", "MT-Chain (all)", "MT-SharedMLP (all)"]


def run_fold(j, tr, te, Xf, Xb, ys, yh, num_all, num_b, cat_b):
    disc = discover2(Xb.iloc[tr], ys.iloc[tr], yh.iloc[tr], num_b, cat_b)
    causal = [f for f in disc["causal"] if f in Xf.columns] or list(Xf.columns)
    Xtr, Xte = Xf.iloc[tr], Xf.iloc[te]
    ystr, yhtr = ys.iloc[tr].reset_index(drop=True), yh.iloc[tr].reset_index(drop=True)
    Xtr = Xtr.reset_index(drop=True)
    strata = (ystr + yhtr).values
    inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + j).split(Xtr, strata))
    res = {}
    for name, make in approaches(list(Xf.columns), causal, num_all).items():
        oof_s, oof_h = np.zeros(len(Xtr)), np.zeros(len(Xtr))
        for itr, iva in inner:
            m = make().fit(Xtr.iloc[itr], ystr.iloc[itr], yhtr.iloc[itr])
            oof_s[iva], oof_h[iva] = m.predict(Xtr.iloc[iva])
        m = make().fit(Xtr, ystr, yhtr)
        ps, ph = m.predict(Xte)
        res[name] = dict(oof_s=oof_s, oof_h=oof_h, ps=ps, ph=ph)
    res["MT-Ensemble"] = {k: np.mean([res[n][k] for n in ENSEMBLE_MEMBERS], axis=0)
                          for k in ["oof_s", "oof_h", "ps", "ph"]}
    for name, r in res.items():
        r["thr_s"], r["inner_f1"] = v2.best_threshold(ystr.values, r["oof_s"])
        r["thr_h"], _ = v2.best_threshold(yhtr.values, r["oof_h"])
    chosen = max(res, key=lambda n: (res[n]["inner_f1"], average_precision_score(ystr, res[n]["oof_s"])))
    res["AutoSelect (nested)"] = dict(res[chosen])
    return dict(fold=j, te=te, res=res, chosen=chosen, disc={k: disc[k] for k in
                ["mb_s", "mb_h", "pa_s", "pa_h", "causal", "ht_sich_edge", "ges_ok"]},
                n_train=len(tr), n_test=len(te))


def main():
    X, ys, yh = load()
    Xf, num, cat, new, Xb, num_b, cat_b = prepare(X)
    num_all = set(num)
    ps_prev, ph_prev = ys.mean(), yh.mean()

    v2.divider("MULTITASK CAUSAL PIPELINE — sICH (primary) + HT (auxiliary)")
    print(f"  Patients                  : {len(ys)}")
    print(f"  sICH                      : {ys.sum()} ({ps_prev * 100:.1f}%)   flag-everyone F1 {2 * ps_prev / (1 + ps_prev):.3f}")
    print(f"  HT                        : {yh.sum()} ({ph_prev * 100:.1f}%)")
    print(f"  P(sICH | HT)              : {ys[yh == 1].mean():.3f}  (sICH ⊂ HT: {int(((ys == 1) & (yh == 0)).sum()) == 0})")
    print(f"  Modelling features        : {Xf.shape[1]} (incl. {len(new)} clinical features)")
    print("  Discovery variables       : base variables (+TICI ordinal); tiers ... → HT (4) → sICH (5)")
    print(f"  Evaluation                : {REPEATS}x{OUTER_FOLDS}-fold outer CV; causal discovery, thresholds and")
    print(f"                              model selection nested inside each training fold ({INNER_FOLDS}-fold inner CV)")

    print("\nFull-data causal discovery (for description only; never used for model selection)...")
    full = discover2(Xb, ys, yh, num_b, cat_b)
    print(f"  HT → sICH edge recovered  : {full['ht_sich_edge']}")
    print(f"  Direct causes of HT       : {full['pa_h']}")
    print(f"  Direct causes of sICH (beyond HT): {full['pa_s']}")
    print(f"  Joint causal feature set  : {full['causal']}")

    folds = list(RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=REPEATS,
                                         random_state=SEED).split(Xf, ys + yh))
    print(f"\nRunning {len(folds)} outer folds...")
    fr = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_fold)(j, tr, te, Xf, Xb, ys, yh, num_all, num_b, cat_b) for j, (tr, te) in enumerate(folds))

    names = list(fr[0]["res"].keys())
    rows, per_fold, curves = [], [], {}
    for name in names:
        pooled = []
        for rep in range(REPEATS):
            fs = [f for f in fr if f["fold"] // OUTER_FOLDS == rep]
            idx = np.concatenate([f["te"] for f in fs])
            p_s = np.concatenate([f["res"][name]["ps"] for f in fs])
            p_h = np.concatenate([f["res"][name]["ph"] for f in fs])
            pr_s = np.concatenate([(f["res"][name]["ps"] >= f["res"][name]["thr_s"]).astype(int) for f in fs])
            pr_h = np.concatenate([(f["res"][name]["ph"] >= f["res"][name]["thr_h"]).astype(int) for f in fs])
            yy_s, yy_h = ys.values[idx], yh.values[idx]
            pooled.append(dict(F1=f1_score(yy_s, pr_s, zero_division=0),
                               Precision=precision_score(yy_s, pr_s, zero_division=0),
                               Recall=recall_score(yy_s, pr_s, zero_division=0),
                               AUC=roc_auc_score(yy_s, p_s), PR_AUC=average_precision_score(yy_s, p_s),
                               Brier=brier_score_loss(yy_s, np.clip(p_s, 0, 1)),
                               HT_AUC=roc_auc_score(yy_h, p_h), HT_F1=f1_score(yy_h, pr_h, zero_division=0)))
            if rep == 0:
                curves[name] = (yy_s, p_s, pr_s)
        m = pd.DataFrame(pooled).mean()
        lo, hi = v2.boot_f1_ci(curves[name][0], curves[name][2], N_BOOT)
        rows.append({"Approach": name, "sICH F1": m.F1, "F1 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "Precision": m.Precision, "Recall": m.Recall, "sICH AUC": m.AUC,
                     "sICH PR-AUC": m.PR_AUC, "Brier": m.Brier, "HT AUC": m.HT_AUC, "HT F1": m.HT_F1})
        for f in fr:
            r = f["res"][name]
            yt = ys.values[f["te"]]
            per_fold.append(dict(Approach=name, Fold=f["fold"], AUC=roc_auc_score(yt, r["ps"]),
                                 F1=f1_score(yt, (r["ps"] >= r["thr_s"]).astype(int), zero_division=0)))
    summary = pd.DataFrame(rows).sort_values("sICH F1", ascending=False).round(4)
    pf = pd.DataFrame(per_fold)
    summary.to_csv(out("mtl_summary.csv"), index=False)
    pf.to_csv(out("mtl_per_fold.csv"), index=False)

    v2.divider("RESULTS — sICH is the primary task (pooled out-of-fold, every patient a test case)")
    print(summary.to_string(index=False))
    print(f"\n  Models picked by AutoSelect: { {k: int(v) for k, v in Counter(f['chosen'] for f in fr).items()} }")

    v2.divider("MULTITASK vs SINGLE-TASK — paired per-fold differences (Nadeau–Bengio corrected)")
    ref = pf[pf.Approach == "ST (all features)"].sort_values("Fold")
    comp = []
    for name in names:
        if name == "ST (all features)":
            continue
        cur = pf[pf.Approach == name].sort_values("Fold")
        for met in ["AUC", "F1"]:
            t = sp.corrected_ttest(cur[met].values - ref[met].values, fr[0]["n_train"], fr[0]["n_test"])
            comp.append(dict(Approach=name, Metric=met, Delta=t["mean"], CI_lo=t["lo"], CI_hi=t["hi"], p=t["p"]))
    comp = pd.DataFrame(comp).round(4)
    comp.to_csv(out("mtl_vs_singletask.csv"), index=False)
    print(comp.to_string(index=False))

    v2.divider("CAUSAL STRUCTURE STABILITY across training folds")
    nf = len(fr)
    freq = []
    for v in sorted({x for f in fr for x in f["disc"]["mb_s"] + f["disc"]["mb_h"]}):
        freq.append(dict(Variable=v, Domain=sp.get_clinical_domain(v),
                         In_MB_sICH=sum(v in f["disc"]["mb_s"] for f in fr) / nf,
                         Direct_cause_sICH=sum(v in f["disc"]["pa_s"] for f in fr) / nf,
                         Direct_cause_HT=sum(v in f["disc"]["pa_h"] for f in fr) / nf))
    freq = pd.DataFrame(freq).sort_values(["Direct_cause_sICH", "Direct_cause_HT"], ascending=False).round(3)
    freq.to_csv(out("mtl_causal_stability.csv"), index=False)
    print(freq.to_string(index=False))
    print(f"\n  HT → sICH edge recovered in {sum(f['disc']['ht_sich_edge'] for f in fr)}/{nf} folds")

    st = summary[summary.Approach == "ST (all features)"].iloc[0]
    auto = summary[summary.Approach == "AutoSelect (nested)"].iloc[0]
    best_mt = summary[summary.Approach.str.startswith("MT")].iloc[0]
    v2.divider("CONCLUSION")
    print(f"  Single-task sICH          : F1 {st['sICH F1']:.3f} {st['F1 95% CI']}, AUC {st['sICH AUC']:.3f}")
    print(f"  Best multitask ({best_mt['Approach']}): F1 {best_mt['sICH F1']:.3f} {best_mt['F1 95% CI']}, AUC {best_mt['sICH AUC']:.3f}")
    print(f"  Nested AutoSelect (unbiased): F1 {auto['sICH F1']:.3f} {auto['F1 95% CI']}, AUC {auto['sICH AUC']:.3f}")
    print(f"  F1 target {F1_TARGET}: {'REACHED' if auto['sICH F1'] >= F1_TARGET else 'NOT REACHED'} (unbiased estimate)")

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    s = summary.iloc[::-1]
    cis = [tuple(float(x) for x in c.strip("[]").split(",")) for c in s["F1 95% CI"]]
    cols = [PAL[0] if a.startswith("AutoSelect") else PAL[2] if a.startswith("ST") else PAL[1] for a in s.Approach]
    ax[0, 0].barh(s.Approach, s["sICH F1"], color=cols, capsize=3,
                  xerr=[[f - c[0] for f, c in zip(s["sICH F1"], cis)], [c[1] - f for f, c in zip(s["sICH F1"], cis)]])
    ax[0, 0].axvline(2 * ps_prev / (1 + ps_prev), color="grey", ls=":", label="Flag-everyone F1")
    ax[0, 0].axvline(F1_TARGET, color="k", ls="--", lw=1, label=f"Target {F1_TARGET}")
    ax[0, 0].legend(handles=[mpatches.Patch(color=PAL[2], label="Single-task"), mpatches.Patch(color=PAL[1], label="Multitask"),
                             mpatches.Patch(color=PAL[0], label="Nested AutoSelect"),
                             Line2D([0], [0], color="grey", ls=":", label="Flag-everyone"),
                             Line2D([0], [0], color="k", ls="--", label="Target 0.6")], fontsize=7)
    ax[0, 0].set_xlabel("sICH F1 (pooled OOF, 95% CI)"); ax[0, 0].set_title("(A) sICH F1", fontweight="bold")
    show = ["ST (all features)", "MT-Hierarchical (all)", "MT-Chain (all)", "MT-SharedMLP (all)", "MT-Ensemble"]
    for i, n in enumerate(show):
        yy, pp, _ = curves[n]
        fpr, tpr, _ = roc_curve(yy, pp)
        ax[0, 1].plot(fpr, tpr, color=PAL[i], lw=2, label=f"{n} ({roc_auc_score(yy, pp):.3f})")
        pr, rc, _ = precision_recall_curve(yy, pp)
        ax[1, 0].plot(rc, pr, color=PAL[i], lw=2, label=f"{n} ({average_precision_score(yy, pp):.3f})")
        pt, pp2 = calibration_curve(yy, np.clip(pp, 0, 1), n_bins=8, strategy="quantile")
        ax[1, 1].plot(pp2, pt, "o-", color=PAL[i], ms=3, label=n)
    ax[0, 1].plot([0, 1], [0, 1], "--", color="grey"); ax[0, 1].set_title("(B) sICH ROC", fontweight="bold")
    ax[0, 1].set_xlabel("FPR"); ax[0, 1].set_ylabel("TPR"); ax[0, 1].legend(fontsize=7)
    ax[1, 0].axhline(ps_prev, color="grey", ls="--"); ax[1, 0].set_title("(C) sICH precision–recall", fontweight="bold")
    ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot([0, 1], [0, 1], "--", color="grey"); ax[1, 1].set_title("(D) sICH calibration", fontweight="bold")
    ax[1, 1].set_xlabel("Predicted"); ax[1, 1].set_ylabel("Observed"); ax[1, 1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out("mtl_performance.png"), dpi=DPI); plt.close()

    g = full["g"]
    keep = set(full["causal"]) | {S, H}
    for a, b in list(g["dir"]) + list(g["und"]):
        if a in (S, H) or b in (S, H):
            keep |= {a, b}
    G = nx.DiGraph(); G.add_nodes_from(keep)
    G.add_edges_from([(a, b) for a, b in g["dir"] if a in keep and b in keep])
    Gu = nx.Graph(); Gu.add_nodes_from(keep); Gu.add_edges_from([(a, b) for a, b in g["und"] if a in keep and b in keep])
    labels = sp.unique_labels(list(G.nodes()))
    labels[S], labels[H] = "sICH", "HT"
    def col(n):
        if n == S: return PAL[0]
        if n == H: return PAL[4]
        if n in full["pa_s"] and n in full["pa_h"]: return PAL[3]
        if n in full["pa_s"]: return PAL[1]
        if n in full["pa_h"]: return PAL[2]
        return PAL[7]
    pos = nx.spring_layout(nx.compose(G.to_undirected(), Gu), seed=SEED, k=1.6 / np.sqrt(max(len(keep), 1)))
    fig, a = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, ax=a, node_color=[col(n) for n in G.nodes()], node_size=1500, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=a, font_size=7, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=a, arrows=True, arrowsize=14, width=1.3, edge_color="#264653",
                           min_source_margin=16, min_target_margin=16, connectionstyle="arc3,rad=0.08")
    if Gu.number_of_edges():
        nx.draw_networkx_edges(Gu, pos, ax=a, style="dashed", edge_color=PAL[6], width=1.3)
    a.legend(handles=[mpatches.Patch(color=PAL[0], label="sICH (primary)"), mpatches.Patch(color=PAL[4], label="HT (auxiliary)"),
                      mpatches.Patch(color=PAL[1], label="Direct cause of sICH"), mpatches.Patch(color=PAL[2], label="Direct cause of HT"),
                      mpatches.Patch(color=PAL[3], label="Cause of both"), mpatches.Patch(color=PAL[7], label="Other MB member"),
                      Line2D([0], [0], color=PAL[6], ls="--", label="Undirected")], fontsize=8, loc="lower left")
    a.set_title("Joint causal structure around HT and sICH (full data, description only)", fontweight="bold")
    a.axis("off"); plt.tight_layout(); plt.savefig(out("mtl_causal_graph.png"), dpi=DPI); plt.close()

    if len(freq):
        fq = freq.head(20).iloc[::-1]
        fig, a = plt.subplots(figsize=(9, max(3, 0.35 * len(fq) + 1)))
        yy = np.arange(len(fq))
        a.barh(yy - 0.2, fq.Direct_cause_sICH * 100, 0.4, color=PAL[1], label="Direct cause of sICH")
        a.barh(yy + 0.2, fq.Direct_cause_HT * 100, 0.4, color=PAL[2], label="Direct cause of HT")
        a.set_yticks(yy); a.set_yticklabels(fq.Variable, fontsize=8); a.set_xlim(0, 100)
        a.set_xlabel("% of training folds"); a.legend(fontsize=8)
        a.set_title("Stability of causal parents across folds", fontweight="bold")
        plt.tight_layout(); plt.savefig(out("mtl_causal_stability.png"), dpi=DPI); plt.close()
    print(f"\n  Outputs in '{OUT_DIR}/': mtl_summary.csv, mtl_vs_singletask.csv, mtl_causal_stability.csv,")
    print("  mtl_per_fold.csv, mtl_performance.png, mtl_causal_graph.png, mtl_causal_stability.png")


if __name__ == "__main__":
    main()
