"""Interval-boundary strategies for discrete-time survival.

All boundaries are fit on the TRAINING fold only and reused on validation (no
leakage). The six strategies from the paper:

    quantile : event-time quantiles (baseline; conventional default)
    uniform  : equal-width (approx. DCS "linear" spacing)
    alltime  : quantiles of all observed times (approx. DCS "quantile" spacing)
    log      : logarithmic spacing (approx. DCS "log" spacing)
    custom   : explicit clinical edges
    aaic     : anchored + density-guided + censoring-removal (demoted in the paper)
"""

import numpy as np


# --------------------------------------------------------------------------- #
# box statistics (shared with AAIC)
# --------------------------------------------------------------------------- #
def box_stats(times, events, edges):
    """Per-box event / censored counts for strictly increasing edges."""
    edges = np.asarray(edges, dtype=np.float64)
    j = np.searchsorted(edges, times, side="right") - 1
    j = np.clip(j, 0, len(edges) - 2)
    n_ev = np.bincount(j, weights=events, minlength=len(edges) - 1)
    n_ce = np.bincount(j, minlength=len(edges) - 1) - n_ev
    return n_ev.astype(np.float64), n_ce.astype(np.float64)


def log_info(times, events, edges, kappa=1.0):
    """Concave log-form information criterion (censoring-aware)."""
    n_ev, n_ce = box_stats(times, events, edges)
    return float(np.log1p(n_ev).sum() - kappa * np.log1p(n_ce).sum())


def _enforce_monotone(e):
    e = np.asarray(e, dtype=np.float64)
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = e[i - 1] + 1e-6
    return e


# --------------------------------------------------------------------------- #
# edge builders
# --------------------------------------------------------------------------- #
def edges_quantile(times, events, n_intervals, max_time=None):
    """Event-time quantile edges (robust to the long survival tail)."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    if max_time is None or max_time <= 0:
        max_time = float(times.max())
    ev = times[events == 1]
    if ev.size < n_intervals:
        return np.linspace(0.0, max_time, n_intervals + 1)
    e = np.quantile(ev, np.linspace(0.0, 1.0, n_intervals + 1))
    e[0] = 0.0
    e[-1] = max(float(e[-1]), max_time) + 1e-6
    return _enforce_monotone(e)


def edges_uniform(times, n_intervals, max_time=None):
    """Equal-width edges over [0, max_time]."""
    times = np.asarray(times, dtype=np.float64)
    if max_time is None or max_time <= 0:
        max_time = float(times.max())
    return np.linspace(0.0, max_time, n_intervals + 1)


def edges_alltime(times, n_intervals):
    """Quantiles of ALL observed times (events and censored) = DCS 'quantile'."""
    times = np.asarray(times, dtype=np.float64)
    e = np.quantile(times, np.linspace(0.0, 1.0, n_intervals + 1))
    e[0] = 0.0
    e[-1] = float(times.max()) + 1e-6
    return _enforce_monotone(e)


def edges_log(times, n_intervals):
    """Logarithmic edges [0, logspace(1, log10(t_max), n+1)[1:]] = DCS 'log'."""
    times = np.asarray(times, dtype=np.float64)
    max_t = float(times.max())
    e = np.concatenate([[0.0], np.logspace(1, np.log10(max_t), n_intervals + 1)[1:]])
    e[-1] = max_t + 1e-6
    return _enforce_monotone(e)


def edges_custom(times, edges_csv):
    """Explicit clinical edges from a comma-separated string."""
    e = np.array([float(x) for x in edges_csv.split(",")], dtype=np.float64)
    if e[-1] < float(np.asarray(times, dtype=np.float64).max()):
        e[-1] = float(np.asarray(times, dtype=np.float64).max()) + 1e-6
    return e


# --------------------------------------------------------------------------- #
# AAIC (anchored + density + censoring-removal). Demoted in the paper; kept here
# as a fixed heuristic with NO learned parameters (but fixed hyper-parameters).
# --------------------------------------------------------------------------- #
DEFAULT_H = np.array([6.0, 12.0, 24.0, 36.0, 48.0])


def _pool(times, events, H, quant_sets=(2, 5, 8, 12, 16)):
    cand = set([0.0, float(np.asarray(times, dtype=np.float64).max())])
    for nq in quant_sets:
        e = edges_quantile(times, events, nq)
        cand.update(float(x) for x in e[1:-1])
    cand.update(float(h) for h in H if 0.0 < float(h) < float(np.asarray(times, dtype=np.float64).max()))
    return np.array(sorted(cand), dtype=np.float64)


def _min_events_below(edges, times, events, thr):
    n_ev, _ = box_stats(times, events, edges)
    return n_ev.min() < thr


def aaic_edges(times, events, H=DEFAULT_H, m=2, min_ev=15.0, kappa=1.0, verbose=False):
    """Anchored clinical nodes + up to m density edges + censoring-removal test."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    max_t = float(times.max())

    cur = np.sort(np.array(
        list(set([0.0, max_t]) | {float(h) for h in H if 0.0 < float(h) < max_t}),
        dtype=np.float64))

    cand = _pool(times, events, H)
    for _ in range(m):
        best, best_d = None, 0.0
        for c in cand:
            if np.any(np.isclose(cur, c, atol=1e-6)):
                continue
            trial = np.sort(np.append(cur, c))
            if _min_events_below(trial, times, events, min_ev):
                continue
            d = log_info(times, events, trial, kappa) - log_info(times, events, cur, kappa)
            if d > best_d:
                best_d, best = d, c
        if best is None or best_d <= 1e-9:
            break
        cur = np.sort(np.append(cur, best))
        if verbose:
            print(f"  [aaic add] +{best:.2f} (info delta={best_d:+.3f})")

    removed = []
    for h in H:
        if not np.any(np.isclose(cur, h, atol=1e-6)):
            continue
        i = int(np.argmin(np.abs(cur - h)))
        if i == 0 or i == len(cur) - 1:
            continue
        if log_info(times, events, np.delete(cur, i), kappa) > log_info(times, events, cur, kappa):
            cur = np.delete(cur, i)
            removed.append(float(h))
    return cur.astype(np.float64), removed


def build_edges(bin_mode, times, events, n_intervals=12, edges_csv=None):
    """Dispatch to a boundary strategy. Returns a strictly increasing edge array."""
    if bin_mode == "quantile":
        return edges_quantile(times, events, n_intervals)
    if bin_mode == "uniform":
        return edges_uniform(times, n_intervals)
    if bin_mode == "alltime":
        return edges_alltime(times, n_intervals)
    if bin_mode == "log":
        return edges_log(times, n_intervals)
    if bin_mode == "custom":
        if not edges_csv:
            raise ValueError("edges_csv required for bin_mode='custom'")
        return edges_custom(times, edges_csv)
    if bin_mode == "aaic":
        e, _ = aaic_edges(times, events)
        return e
    raise ValueError(f"unknown bin_mode: {bin_mode}")