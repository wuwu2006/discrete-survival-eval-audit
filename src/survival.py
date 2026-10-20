"""Discrete-time survival head, loss, and scalar-risk helper.

Shapes throughout:
    logit  : (B, n)   one logit per time interval
    times  : (B,)     months
    events : (B,)     1 = death observed, 0 = censored
    edges  : (n+1,)   strictly increasing interval boundaries (e_0=0 ... e_n)

Discrete hazard / survival:
    lambda_j = sigmoid(logit_j)              conditional death prob in interval j
    S(e_j)   = prod_{m<=j} (1 - lambda_m)    survive past the j-th interval
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteSurvivalHead(nn.Module):
    """Two-layer MLP mapping a pooled representation z -> n interval logits."""

    def __init__(self, in_dim, hidden, n_intervals, dropout=0.1):
        super().__init__()
        self.n_intervals = n_intervals
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_intervals),
        )

    def forward(self, z):
        return self.net(z)  # (B, n) logits


def discrete_survival_nll(logit, times, events, edges):
    """Censored negative log-likelihood of the discrete hazard (scalar).

    event in interval j        : log lambda_j + sum_{m<j} log(1 - lambda_m)
    censored in interval j     : sum_{m<j} log(1 - lambda_m)
    """
    n = logit.shape[1]
    log_h = F.logsigmoid(logit)                # log lambda_j
    log_s = F.logsigmoid(-logit)               # log (1 - lambda_j)

    edges_t = torch.as_tensor(np.asarray(edges, dtype=np.float64),
                              device=logit.device, dtype=logit.dtype)
    j = torch.searchsorted(edges_t, times, right=True) - 1   # interval index (0-based)
    j = j.clamp(0, n - 1)

    cum = torch.cumsum(log_s, dim=1)           # cum[b,k] = sum_{m<=k} log(1-lambda_m)
    shift = F.pad(cum, (1, 0), value=0.0)[:, :n]   # shift[b,k] = cum[b,k-1]
    log_h_j = log_h.gather(1, j.unsqueeze(1)).squeeze(1)
    prefix = shift.gather(1, j.unsqueeze(1)).squeeze(1)

    ll = events * (log_h_j + prefix) + (1.0 - events) * prefix
    return -ll.mean()


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def hazard_survival_risk(logits, edges):
    """From logits (B,n) return (hazard (B,n), survival (B,n), risk (B,)).

    survival[:, j] = S(e_{j+1}) = P(T > e_{j+1}).

    risk = -expected_lifetime with expected_lifetime = sum_j S(e_{j+1}); higher
    risk <=> shorter expected survival. Summing the survival curve (not the
    cumulative hazard) is what ranks early vs late deaths correctly.
    """
    logits = np.asarray(logits, dtype=np.float64)
    hazard = sigmoid(logits)
    survival = np.cumprod(1.0 - hazard, axis=1)     # (B, n)
    expected = survival.sum(axis=1)                 # discrete expected lifetime
    risk = -expected
    return hazard, survival, risk


def survival_at(logits, edges, horizon):
    """(DEPRECATED / DOCUMENTED BUG) naive survival ``P(T > horizon)``.

    Returns a constant 1.0 whenever ``horizon`` falls inside the FIRST box
    (i.e. ``horizon < edges[1]``), and returns ``S(previous right edge)``
    otherwise. This is the "first-box constant" shortcut that fakes a
    node-level boundary-geometry effect (the AUC is pinned at exactly 0.5).
    Kept only as a reference / regression target; use
    :func:`src.interpolation.surv_pc` in actual evaluations.
    """
    logits = np.asarray(logits, dtype=np.float64)
    hazard = sigmoid(logits)
    survival = np.cumprod(1.0 - hazard, axis=1)     # survival[:, j] = S(e_{j+1})
    upper = np.asarray(edges[1:], dtype=np.float64)  # right edges e_1..e_n
    k = int(np.searchsorted(upper, horizon, side="right"))  # #intervals ending <= horizon
    if k <= 0:
        return np.ones(logits.shape[0])
    return survival[:, k - 1]