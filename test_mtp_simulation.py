"""Known-truth simulation for the doubly robust MTP estimator in sich_counterfactual.py."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sich_counterfactual as cf

rng = np.random.default_rng(1)
expit = lambda x: 1 / (1 + np.exp(-x))


def simulate(n):
    W = rng.normal(size=(n, 3))
    A = 145 + 12 * W[:, 0] - 6 * W[:, 1] + rng.normal(0, 15, n)           # confounded continuous exposure
    lin = lambda a: -2.2 + 0.035 * (a - 145) + 0.8 * W[:, 0] + 0.5 * W[:, 2]
    return W, A, lin


def truth(delta, floor, n=2_000_000):
    W, A, lin = simulate(n)
    return expit(lin(cf.shift(A, delta, floor))).mean() - expit(lin(A)).mean()


res = {"dr": [], "gcomp": [], "ipw": []}
cover = 0
reps, n, delta, floor = 60, 470, 20, 140
tv = truth(delta, floor)
for r in range(reps):
    W, A, lin = simulate(n)
    Y = rng.binomial(1, expit(lin(A)))
    Ad = cf.shift(A, delta, floor)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=r).split(W, Y))
    mo, ms, ra = cf.mtp_crossfit(A, W, Y, Ad, folds, "flexible", cf.RATIO_Q)
    phis = cf.mtp_estimates(Y, mo, ms, ra)
    for k in res:
        res[k].append(phis[k].mean() - Y.mean())
    c = cf.contrast(phis["dr"], Y)
    cover += c["RD_lo"] <= tv <= c["RD_hi"]
print(f"true risk difference: {100 * tv:+.3f} pp")
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:6s}: mean {100 * v.mean():+.3f} pp | bias {100 * (v.mean() - tv):+.3f} pp | SD {100 * v.std():.3f} pp")
print(f"  DR 95% CI coverage over {reps} replicates: {cover / reps:.2f}")
