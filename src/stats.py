"""Statistical inference for the non-independent 5x5 repeated-CV folds.

The 25 folds = 5 stratified 5-fold repeats. Folds within a repeat share the
training set, so they are not independent. We report three views and never
claim significance from any single one:

    wilcoxon_p       plain paired Wilcoxon (over-optimistic)
    corrected_t_p    Nadeau-Bengio resampled-t approximation (over-conservative
                     when applied to a repeated design, ~7x variance inflation)
    repeat_level_p   paired t over the 5 repeat means (df=4; structurally the
                     correct independent unit, but under-powered)

BH-FDR is applied over the full comparison family.
"""

import numpy as np
from scipy import stats


def wilcoxon_p(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def corrected_t_p(diffs, n2, n1):
    """Nadeau-Bengio corrected paired-t p. n=#folds, n1=train, n2=test size.

    t = mean(d) / sqrt( (1/n + n2/n1) * var(d, ddof=1) ),  df = n - 1.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0 or n < 2:
        return float("nan")
    se = np.sqrt((1.0 / n + n2 / n1) * var)
    t = d.mean() / se
    return float(2 * (1 - stats.t.cdf(abs(t), n - 1)))


def repeat_means(ci_by_fold, repeats=5, folds_per_repeat=5):
    """fold -> cindex dict  ==>  (repeats,) array of per-repeat means."""
    reps = []
    for r in range(repeats):
        vals = [ci_by_fold[f] for f in range(repeats * r + 1, repeats * r + 1 + folds_per_repeat)
                if f in ci_by_fold]
        reps.append(float(np.mean(vals)))
    return np.array(reps)


def repeat_level_p(variant_ci, base_ci, repeats=5, folds_per_repeat=5):
    """Paired t (df = repeats-1) over per-repeat mean c-index vs baseline.

    Returns (mean_diff, t, p).
    """
    v = repeat_means(variant_ci, repeats, folds_per_repeat)
    b = repeat_means(base_ci, repeats, folds_per_repeat)
    d = v - b
    t, p = stats.ttest_rel(v, b)
    return float(d.mean()), float(t), float(p)


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR q-values from a list of p-values (nan dropped)."""
    p = np.asarray([x for x in pvals if x == x], dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([])
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        qval = min(p[idx] * m / rank, prev)
        q[idx] = qval
        prev = qval
    return q