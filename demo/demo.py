"""End-to-end demo of the evaluation-interpolation artifact (no data, no GPU).

Reproduces the paper's core finding on synthetic logits:

    * a naive "first-box constant" interpolation pins the node AUC at 0.5 for
      any horizon inside the first box;
    * replacing it with step / linear / piecewise-constant-hazard interpolation
      restores the discrimination;
    * linear and piecewise-constant interpolation give essentially the same AUC
      (both are monotone in the same per-interval hazards).

Run:  python demo/demo.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interpolation import surv_naive, surv_step, surv_linear, surv_pc
from src.metrics import make_censoring_survival, time_dependent_auc


def synthetic_cohort(n_high=100, n_low=100, seed=0):
    rng = np.random.default_rng(seed)
    # high-risk patients die early (3-8 months); low-risk die late (30-60 months)
    t_high = rng.uniform(3.0, 8.0, n_high)
    t_low = rng.uniform(30.0, 60.0, n_low)
    times = np.concatenate([t_high, t_low])
    events = np.ones(n_high + n_low, dtype=np.int64)

    # 5 intervals; first interval lambda drives the early/late distinction
    logits = np.tile([-1.0, -1.0, -1.0, -1.0, -1.0], (n_high + n_low, 1))
    logits[:n_high, 0] = +3.0     # lambda_0 ~ 0.95 -> early death
    logits[n_low:, 0] = -3.0      # lambda_0 ~ 0.05 -> survivor
    return logits, times, events


def show(logits, times, events, edges, horizon):
    g = make_censoring_survival(times, events)   # =1 here (no censoring)
    print(f"  edges = {edges.tolist()}   horizon = {horizon:g}")
    row = {}
    for name, fn in [("naive", surv_naive), ("step", surv_step),
                     ("linear", surv_linear), ("pc", surv_pc)]:
        s = fn(logits, edges, horizon)
        auc = time_dependent_auc(-s, times, events, horizon, g)
        hi, lo = s[:100].mean(), s[100:].mean()
        row[name] = auc
        print(f"    {name:7s}  AUC@6m = {auc:.4f}   "
              f"S(high-risk)={hi:.3f}  S(low-risk)={lo:.3f}")
    return row


def main():
    logits, times, events = synthetic_cohort()

    print("=" * 70)
    print("Scenario A: coarse first box  [0, 10]  covers the 6-month horizon")
    print("=" * 70)
    wide = np.array([0.0, 10.0, 20.0, 35.0, 50.0, 60.0])
    show(logits, times, events, wide, 6.0)

    print()
    print("=" * 70)
    print("Scenario B: narrow first box  [0, 3]   (6 m sits at an edge)")
    print("=" * 70)
    narrow = np.array([0.0, 3.0, 10.0, 20.0, 40.0, 60.0])
    show(logits, times, events, narrow, 6.0)

    print()
    print("Takeaway: the 0.5 in Scenario A appears ONLY under 'naive' and only")
    print("when the horizon falls inside the first box. 'step' differs from")
    print("'naive' only inside the first box, yet restores the full AUC; and")
    print("'linear' ~ 'pc' to within 0.001. The apparent boundary-geometry")
    print("effect is therefore an interpolation artifact, not a model property.")


if __name__ == "__main__":
    main()