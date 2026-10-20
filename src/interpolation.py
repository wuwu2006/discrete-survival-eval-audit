"""The four survival-interpolation schemes (the object of the audit).

A discrete survival curve is a step function defined only at the interval
edges. Evaluating it at a horizon ``h`` inside an interval therefore requires
an interpolation. These four schemes are compared in the paper (Section III-E).

All functions take ``(logits, edges, horizon)`` and return S(h) per patient.

    logits : (B, n) interval logits     edges : (n+1,) boundaries     horizon : float
"""

import numpy as np

from .survival import sigmoid


def _survival_edges(logits):
    """S[:, j] = S(e_{j+1}) = P(T > e_{j+1})."""
    hazard = sigmoid(logits)
    return np.cumprod(1.0 - hazard, axis=1)


def surv_naive(logits, edges, horizon):
    """naive-constant: S=1.0 inside the first box, else floor (right-edge) value.

    This is the flawed shortcut; it pins node AUC at 0.5 for any horizon inside
    the first box.
    """
    S = _survival_edges(logits)
    upper = np.asarray(edges[1:], dtype=np.float64)
    k = int(np.searchsorted(upper, horizon, side="right"))
    if k <= 0:
        return np.ones(logits.shape[0])
    return S[:, k - 1]


def surv_step(logits, edges, horizon):
    """step (floor): S at the largest edge <= horizon; first box uses S(e_1).

    Identical to ``surv_naive`` everywhere EXCEPT inside the first box, which is
    what localizes the artifact to the single "return 1.0 in the first box" line.
    """
    S = _survival_edges(logits)
    upper = np.asarray(edges[1:], dtype=np.float64)
    k = int(np.searchsorted(upper, horizon, side="right"))
    if k <= 0:
        return S[:, 0]  # S(e_1), not 1.0
    return S[:, k - 1]


def surv_linear(logits, edges, horizon):
    """linear (DCS scheme): arithmetic interpolation of S between box edges."""
    n = logits.shape[1]
    S = _survival_edges(logits)
    S_left = np.concatenate([np.ones((logits.shape[0], 1)), S], axis=1)  # S(e_j)
    j = int(np.clip(np.searchsorted(edges, horizon, side="right") - 1, 0, n - 1))
    width = edges[j + 1] - edges[j]
    frac = float(np.clip((horizon - edges[j]) / max(width, 1e-9), 0.0, 1.0))
    return S_left[:, j] + frac * (S_left[:, j + 1] - S_left[:, j])


def surv_pc(logits, edges, horizon):
    """piecewise-constant-hazard (PCH; CHI constant-hazard special case).

    Each interval j carries a constant hazard rate h_j = -log(1-lambda_j)/width_j
    (units 1/month), so S(t) = S(tau_j) * exp(-h_j * (t - tau_j)). At the edges
    this reproduces the discrete probabilities S(tau_{j+1}) = S(tau_j) * (1-lambda_j).

    NOTE: dividing by width_j is essential; omitting it collapses S to ~0 in a
    wide first box and breaks monotonicity at the edges.
    """
    logits = np.asarray(logits, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n = logits.shape[1]
    B = logits.shape[0]
    hazard = sigmoid(logits)                        # lambda_j, (B, n)
    widths = np.clip(edges[1:] - edges[:-1], 1e-9, None)   # (n,)
    h = -np.log1p(-hazard) / widths[None, :]        # (B, n) constant hazard, 1/month

    cum = np.cumsum(h * widths[None, :], axis=1)    # cum[:, j] = sum_{m<=j} h_m * w_m
    log_S_left = np.concatenate([np.zeros((B, 1)), -cum[:, :-1]], axis=1)  # (B, n)
    j = np.searchsorted(edges, horizon, side="right") - 1
    j = j.clip(0, n - 1).astype(np.int64)
    dt = np.maximum(horizon - edges[j], 0.0)
    logS = log_S_left[np.arange(B), j] - dt * h[np.arange(B), j]
    return np.exp(logS)


# canonical aliases used by the rest of the repo
SURVIVAL_SCHEMES = {
    "naive": surv_naive,
    "step": surv_step,
    "linear": surv_linear,
    "pc": surv_pc,
}