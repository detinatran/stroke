"""Causal counterfactual analysis of early neurological deterioration (END) in minor ischaemic stroke.

Re-uses the doubly robust estimators of sich_counterfactual.py (modified treatment policies with a
classification-based density ratio, AIPW for binary treatments, grouped bootstrap) on the minor-stroke
cohort (932 patients). Haemorrhagic transformation was not analysed: 7 events are too few.

Exposures
  * admission systolic blood pressure  — MTP: lower by delta, not below 140 mmHg
  * admission glucose                  — MTP: lower by delta, not below 7.8 mmol/L
  * dual antiplatelet therapy (DAPT)   — static: everyone vs no one on DAPT (AIPW)

Adjustment sets follow temporal tiers (baseline -> admission -> in-hospital treatment -> outcome):
for every exposure we adjust for baseline and admission variables, never for treatments that may be
decided on the exposure or for anything measured after END.

Run:  python minor_stroke_counterfactual.py            (SICH_QUICK=1 SICH_BOOT=30 for a smoke test)
"""
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SICH_OUT", os.path.join(HERE, "outputs_minor"))
_lib = os.path.join(HERE, "..") if os.path.exists(os.path.join(HERE, "..", "sich_counterfactual.py")) else os.path.join(HERE, "..", "sich_counterfactual")
sys.path.insert(0, os.environ.get("CF_LIB", _lib))      # repo layout: this folder sits next to sich_counterfactual.py

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

import sich_counterfactual as cf

OUT = os.environ["SICH_OUT"]
DATA = os.environ.get("MINOR_DATA", os.path.join(HERE, "data_minor_stroke.xlsx"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- variables and tiers
OUTCOME = "END"
POST_OUTCOME = ["END appearance time", "NIHSS score increased", "hemorrhagic transformation",
                "degree of hemorrhagic transformation", "Days in hospital", "discharged mRS", "mRS 30 day",
                "stroke recurrence within 30 days", "mRS 90 day", "stroke recurrence within 90 days"]
DROP = ["No.", "ICD code"]
BASELINE = ["Age", "Sex", "Stroke risk factor", "Hypertension", "Atrial fibrillation or atrial flutter", "Diabetes",
            "Hypercholesterolemia", "Smoking", "Obesity", "BMI", "Coronary artery disease history", "Heart failure history",
            "Ischemic stroke or TIA history", "Mechanical heart valve", "Bioprosthetic heart valve", "mRS"]
TREATMENT = ["Reperfusion therapy", "thrombolytic", "rtPA dose", "DTN", "mechanical thrombectomy", "DTG", "antithrombotic",
             "SAPT", "DAPT", "anticoagulant", "anticoagulant vitamin K", "apixaban", "dabigatran", "rivaroxaban", "Statin",
             "Statin intensity level", "diabetes medication treatment", "hypertension medication treatment"]
TIER = {**{c: 0 for c in BASELINE}, **{c: 3 for c in TREATMENT}}      # everything else measured at admission: tier 2
CATEGORICAL = {"Sex", "Admission window time", "TOAST Classification", "number of cerebral infarction locations",
               "corresponding intracranial stenosis", "corresponding intracranial occlusion", "Dominant hemisphere lesion",
               "Statin intensity level", "mRS"}

EXPOSURES = {
    "admissionSBP": dict(label="Admission systolic BP", unit="mmHg", floor=140.0, grid=[5, 10, 20, 30], primary=20,
                         exclude={"admissionDBP"},
                         minimal=["Age", "Hypertension", "NIHSS score", "Admission window time", "Diabetes",
                                  "Ischemic stroke or TIA history"]),
    "Blood glucose on admission": dict(label="Admission glucose", unit="mmol/L", floor=7.8, grid=[1, 2, 3], primary=2,
                                       exclude=set(),
                                       minimal=["Age", "Diabetes", "NIHSS score", "Admission window time", "BMI"]),
}
DAPT_MINIMAL = ["Age", "NIHSS score", "Admission window time", "Atrial fibrillation or atrial flutter",
                "Ischemic stroke or TIA history", "TOAST Classification", "number of cerebral infarction locations"]


def load():
    d = pd.read_excel(DATA)
    d.columns = [str(c).strip() for c in d.columns]
    d = d.replace(["#NULL!", " ", ""], np.nan)
    d = d.drop(columns=[c for c in DROP if c in d.columns])
    d = d.drop(columns=[c for c in d.columns if d[c].nunique(dropna=True) <= 1])
    Y = (d[OUTCOME] == 1).astype(int).values
    X = d.drop(columns=[OUTCOME] + [c for c in POST_OUTCOME if c in d.columns])
    for c in X.columns:
        if c in CATEGORICAL:
            X[c] = X[c].astype("Int64").astype(str).where(X[c].notna(), np.nan).astype(object)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.reset_index(drop=True), Y


def tier(c):
    return TIER.get(c, 2)


def pre_exposure(X, T, exclude=()):
    """Baseline and admission variables (tier <= 2), minus the exposure and listed exclusions."""
    return [c for c in X.columns if c != T and c not in exclude and tier(c) <= 2]


def run_binary(X, Y, T_bin, adj, folds_by_rep, kind="flexible", boot=True, rows_mask=None):
    """AIPW for 'everyone treated' and 'no one treated' versus the natural course, and their difference."""
    if rows_mask is not None:
        keep = np.asarray(rows_mask, bool)
        idx = np.where(keep)[0]
        pos = {g: i for i, g in enumerate(idx)}
        folds_by_rep = [[(np.array([pos[i] for i in tr if i in pos]), np.array([pos[i] for i in te if i in pos])) for tr, te in fr]
                        for fr in folds_by_rep]
        X, Y, T_bin = X.loc[keep].reset_index(drop=True), Y[keep], T_bin[keep]
    W = cf.design(X, adj)
    out = {"everyone on DAPT": [], "no one on DAPT": [], "DAPT vs no DAPT": []}
    for folds in folds_by_rep:
        psi1, psi0 = cf.aipw_once(T_bin, W, Y, folds, kind)
        for name, phi in (("everyone on DAPT", psi1), ("no one on DAPT", psi0)):
            c = cf.contrast(phi, Y); c.update(ESS=np.nan, Ratio_max=np.nan, Pct_shifted=np.nan)
            out[name].append(c)
        n = len(Y); dd = psi1 - psi0; se = dd.std(ddof=1) / np.sqrt(n)
        rr = psi1.mean() / psi0.mean()
        se_l = (psi1 / psi1.mean() - psi0 / psi0.mean()).std(ddof=1) / np.sqrt(n)
        out["DAPT vs no DAPT"].append(dict(Risk_policy=psi1.mean(), Risk_natural=psi0.mean(), RD=dd.mean(), RD_se=se, RR=rr,
                                           RR_lo=np.exp(np.log(rr) - 1.96 * se_l), RR_hi=np.exp(np.log(rr) + 1.96 * se_l),
                                           Prevented_fraction=np.nan))
    res = {k: cf.combine_repeats(v) for k, v in out.items()}
    for k in res:
        res[k]["N"] = int(len(Y)); res[k]["N_treated"] = int(T_bin.sum())
    if boot:
        draws = np.array(Parallel(n_jobs=-1)(delayed(cf._boot_lysis_one)(b, T_bin, W, Y, kind) for b in range(cf.B_BOOT)))
        for j, k in enumerate(["everyone on DAPT", "no one on DAPT", "DAPT vs no DAPT"]):
            res[k].update(cf.boot_summary(draws[:, j], res[k]["RD"]))
    return res


def main():
    t0 = time.time()
    X, Y = load()
    print(f"n = {len(Y)}, END = {Y.sum()} ({Y.mean():.1%}); predictors = {X.shape[1]}")
    folds = list(RepeatedStratifiedKFold(n_splits=cf.FOLDS, n_repeats=cf.REPEATS, random_state=cf.SEED).split(np.zeros(len(Y)), Y))
    folds_by_rep = [folds[r * cf.FOLDS:(r + 1) * cf.FOLDS] for r in range(cf.REPEATS)]
    main_rows, dose_rows, abl_rows, strata_all, adj_log = [], [], [], [], {}

    for T, spec in EXPOSURES.items():
        cf.sp.divider(f"MODIFIED TREATMENT POLICY — {spec['label']}")
        adj = pre_exposure(X, T, spec["exclude"])
        adj_min = [c for c in spec["minimal"] if c in X.columns]
        adj_log[T] = dict(tier=adj, minimal=adj_min)
        print(f"  adjustment: tier-based {len(adj)} variables, minimal {len(adj_min)}")
        for delta in spec["grid"]:
            res, pp_, idx, A, Ad = cf.run_mtp(X, Y, T, spec, adj, folds_by_rep, delta, boot=True)
            dose_rows.append(dict(Exposure=T, Label=spec["label"], Delta=delta, Floor=spec["floor"], Unit=spec["unit"], **res))
            print(f"  −{delta} {spec['unit']} (floor {spec['floor']:g}): RD {100 * res['RD']:+.2f} pp | bootstrap "
                  f"[{100 * res['Boot_lo']:+.2f}, {100 * res['Boot_hi']:+.2f}] | IF [{100 * res['RD_lo']:+.2f}, {100 * res['RD_hi']:+.2f}] | "
                  f"shifted {res['Pct_shifted']:.0%}, ESS {res['ESS']:.0f}", flush=True)
            if delta == spec["primary"]:
                main_rows.append(dict(Exposure=T, Label=spec["label"], Policy=f"−{delta} {spec['unit']}, not below {spec['floor']:g}", **res))
                st, _ = cf.risk_strata(Y, pp_, idx)
                st.insert(0, "Exposure", T); strata_all.append(st)
        for lab, kw in [("PRIMARY: DR, tier-based set, flexible, ratio truncated", dict(adj=adj)),
                        ("− outcome model (IPW only)", dict(adj=adj, estimator="ipw")),
                        ("− density ratio (g-computation only)", dict(adj=adj, estimator="gcomp")),
                        ("linear instead of flexible nuisances", dict(adj=adj, kind="linear")),
                        ("minimal clinical adjustment set", dict(adj=adj_min)),
                        ("no ratio truncation", dict(adj=adj, ratio_q=None))]:
            res, *_ = cf.run_mtp(X, Y, T, spec, kw.pop("adj"), folds_by_rep, spec["primary"], **kw)
            abl_rows.append(dict(Exposure=T, Configuration=lab, **res))

    cf.sp.divider("STATIC INTERVENTION — dual antiplatelet therapy (AIPW)")
    T_dapt = (X["DAPT"].astype(float) == 1).astype(int).values
    adj_d = pre_exposure(X, "DAPT")
    adj_log["DAPT"] = dict(tier=adj_d, minimal=DAPT_MINIMAL)
    print(f"  DAPT {T_dapt.sum()} / {len(T_dapt)}; crude END {Y[T_dapt == 1].mean():.3f} vs {Y[T_dapt == 0].mean():.3f}; adjustment {len(adj_d)} variables")
    dres = run_binary(X, Y, T_dapt, adj_d, folds_by_rep)
    for k, v in dres.items():
        main_rows.append(dict(Exposure="DAPT", Label="Dual antiplatelet therapy", Policy=k, **v))
        print(f"  {k:18s}: RD {100 * v['RD']:+.2f} pp | bootstrap [{100 * v['Boot_lo']:+.2f}, {100 * v['Boot_hi']:+.2f}]")
    crude = dict(Risk_policy=Y[T_dapt == 1].mean(), Risk_natural=Y[T_dapt == 0].mean(), RD=Y[T_dapt == 1].mean() - Y[T_dapt == 0].mean())
    abl_rows.append(dict(Exposure="DAPT", Configuration="crude (unadjusted) difference", **crude))
    abl_rows.append(dict(Exposure="DAPT", Configuration="PRIMARY: AIPW, tier-based set, flexible", **dres["DAPT vs no DAPT"]))
    for lab, kw in [("linear instead of flexible nuisances", dict(adj=adj_d, kind="linear")),
                    ("minimal clinical adjustment set", dict(adj=[c for c in DAPT_MINIMAL if c in X.columns])),
                    ("restricted to DAPT or SAPT patients", dict(adj=adj_d, rows_mask=(X["DAPT"].astype(float) == 1) | (X["SAPT"].astype(float) == 1)))]:
        r = run_binary(X, Y, T_dapt, kw.pop("adj"), folds_by_rep, boot=False, **kw)["DAPT vs no DAPT"]
        abl_rows.append(dict(Exposure="DAPT", Configuration=lab, **r))

    pd.DataFrame(main_rows).to_csv(os.path.join(OUT, "minor_main_policies.csv"), index=False)
    pd.DataFrame(dose_rows).to_csv(os.path.join(OUT, "minor_dose_response.csv"), index=False)
    pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "minor_ablation.csv"), index=False)
    pd.concat(strata_all).to_csv(os.path.join(OUT, "minor_risk_strata.csv"), index=False)
    json.dump(adj_log, open(os.path.join(OUT, "minor_adjustment_sets.json"), "w"), indent=1)
    lines = [f"Minor stroke cohort: n = {len(Y)}, END = {Y.sum()}; {cf.REPEATS}x{cf.FOLDS} cross-fitting; {cf.B_BOOT} bootstrap resamples"]
    for r in main_rows:
        lines.append(f"{r['Label']}: {r['Policy']} -> {100 * r['Risk_policy']:.1f}% vs {100 * r['Risk_natural']:.1f}%, RD {100 * r['RD']:+.2f} pp, "
                     f"bootstrap 95% CI [{100 * r['Boot_lo']:+.2f}, {100 * r['Boot_hi']:+.2f}]")
    open(os.path.join(OUT, "minor_key_numbers.txt"), "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
