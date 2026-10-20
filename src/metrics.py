"""Survival evaluation metrics (self-contained, no lifelines/sksurv dependency).

    harrell_cindex      — Harrell's concordance index
    kaplan_meier        — KM survival estimate (used to build IPCW weights)
    make_censoring_survival — callable G(t) = P(censoring > t)
    ipcw_brier          — IPCW Brier score at a horizon (Graf et al. 1999)
    time_dependent_auc  — IPCW cumulative/dynamic AUC (Uno et al. 2011)
    uno_cindex          — IPCW concordance index at tau (Uno et al. 2011)

Conventions:  times = months, events = 1 death / 0 censored,
              risk = scalar (higher = worse), surv = P(T > horizon) per patient.
"""

import numpy as np


def harrell_cindex(risk, times, events):
    """Harrell's C-index. Returns nan if no comparable pairs exist."""
    risk = np.asarray(risk, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    n = len(times)
    num, den = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            ri, rj = risk[i], risk[j]
            if ei and ej:
                if ti != tj:
                    den += 1
                    if (ti < tj and ri > rj) or (ti > tj and ri < rj):
                        num += 1
                    elif ri == rj:
                        num += 0.5
            elif ei and not ej:
                if ti <= tj:
                    den += 1
                    if ri > rj:
                        num += 1
                    elif ri == rj:
                        num += 0.5
            elif not ei and ej:
                if tj <= ti:
                    den += 1
                    if rj > ri:
                        num += 1
                    elif rj == ri:
                        num += 0.5
    return num / den if den > 0 else float("nan")


def kaplan_meier(times, events):
    """KM survival estimate; s_km[k] = P(T > t_km[k])."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    order = np.argsort(times)
    t = times[order]
    e = events[order]
    t_out, s_out = [], []
    surv = 1.0
    i = 0
    m = len(t)
    while i < m:
        ti = t[i]
        j = i
        d = 0
        while j < m and t[j] == ti:
            d += e[j]
            j += 1
        n_risk = m - i
        if n_risk > 0 and d > 0:
            surv *= (1.0 - d / n_risk)
        t_out.append(ti)
        s_out.append(surv)
        i = j
    return np.array(t_out), np.array(s_out)


def make_censoring_survival(times, events):
    """Return callable G(t) = P(censoring time > t), via KM on 1-events."""
    t_km, s_km = kaplan_meier(times, 1 - events)

    def g(t):
        if t <= 0:
            return 1.0
        idx = int(np.searchsorted(t_km, t, side="right")) - 1
        if idx < 0:
            return 1.0
        return max(float(s_km[idx]), 1e-6)

    return g


def ipcw_brier(surv, times, events, horizon, g):
    """IPCW Brier score at `horizon` (Graf et al. 1999)."""
    surv = np.asarray(surv, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    n = len(times)
    acc = 0.0
    for i in range(n):
        ti, ei = times[i], events[i]
        if ei and ti <= horizon:
            w = 1.0 / g(ti)
            acc += w * (0.0 - surv[i]) ** 2
        elif ti > horizon:
            w = 1.0 / g(horizon)
            acc += w * (1.0 - surv[i]) ** 2
    return acc / n


def time_dependent_auc(risk, times, events, horizon, g):
    """IPCW cumulative/dynamic AUC at `horizon` (Uno et al. 2011).

    ``risk`` is a scalar per patient with higher = higher risk (feed -S(h)).
    Ties (equal risk) contribute 0.5; if every patient has the same risk the
    AUC is exactly 0.5 in this implementation.
    """
    risk = np.asarray(risk, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    n = len(times)
    case_w = np.zeros(n)
    ctrl_w = np.zeros(n)
    for i in range(n):
        ti, ei = times[i], events[i]
        if ei and ti <= horizon:
            case_w[i] = 1.0 / g(ti)
        elif ti > horizon:
            ctrl_w[i] = 1.0 / g(horizon)
    num, den = 0.0, 0.0
    for i in range(n):
        if case_w[i] == 0.0:
            continue
        for j in range(n):
            if ctrl_w[j] == 0.0:
                continue
            w = case_w[i] * ctrl_w[j]
            den += w
            if risk[i] > risk[j]:
                num += w
            elif risk[i] == risk[j]:
                num += 0.5 * w
    return num / den if den > 0 else float("nan")


def uno_cindex(risk, times, events, tau, g):
    """Uno et al. IPCW C-index at ``tau`` (weights cases by 1/G(t)^2)."""
    risk = np.asarray(risk, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    n = len(times)
    num, den = 0.0, 0.0
    for i in range(n):
        if not events[i] or times[i] >= tau:
            continue
        wi = 1.0 / (g(times[i]) ** 2)
        for j in range(n):
            if times[j] > times[i]:
                den += wi
                if risk[i] > risk[j]:
                    num += wi
                elif risk[i] == risk[j]:
                    num += 0.5 * wi
    return num / den if den > 0 else float("nan")