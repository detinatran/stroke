import numpy as np, sys
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import sich_counterfactual as cf
expit = lambda x: 1 / (1 + np.exp(-x))
def simulate(rng, n, nonlin=False):
    W = rng.normal(size=(n, 3))
    A = 145 + 12 * W[:, 0] - 6 * W[:, 1] + rng.normal(0, 15, n)
    if nonlin:
        lin = lambda a: -2.2 + 0.035 * (a - 145) + 0.8 * np.abs(W[:, 0]) + 0.5 * W[:, 2] * W[:, 1] + 0.02 * np.maximum(a - 160, 0)
    else:
        lin = lambda a: -2.2 + 0.035 * (a - 145) + 0.8 * W[:, 0] + 0.5 * W[:, 2]
    return W, A, lin
for nonlin in (False, True):
    rng = np.random.default_rng(7)
    W, A, lin = simulate(rng, 2_000_000, nonlin); tv = expit(lin(cf.shift(A, 20, 140))).mean() - expit(lin(A)).mean()
    variants = {"trunc0.99, raw (old)": (0.99, False), "trunc0.99, normalised": (0.99, True), "none, normalised": (None, True)}
    out = {k: dict(est=[], cov=0, g=[]) for k in variants}
    reps = 50
    for r in range(reps):
        W, A, lin = simulate(rng, 470, nonlin); Y = rng.binomial(1, expit(lin(A))); Ad = cf.shift(A, 20, 140)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=r).split(W, Y))
        for k, (q, nz) in variants.items():
            mo, ms, ra = cf.mtp_crossfit(A, W, Y, Ad, folds, "flexible", q, nz)
            phi = cf.mtp_estimates(Y, mo, ms, ra)
            c = cf.contrast(phi["dr"], Y); out[k]["est"].append(c["RD"]); out[k]["cov"] += c["RD_lo"] <= tv <= c["RD_hi"]
            out[k]["g"].append(phi["gcomp"].mean() - Y.mean())
    print(f"=== {'non-linear' if nonlin else 'linear'} outcome; truth {100*tv:+.3f} pp; g-comp bias {100*(np.mean(out[list(variants)[0]]['g'])-tv):+.3f}")
    for k, v in out.items():
        e = np.array(v["est"]); print(flush=True) if False else None; print(f"  {k:24s} bias {100*(e.mean()-tv):+.3f} pp | SD {100*e.std():.3f} | coverage {v['cov']/reps:.2f}")
