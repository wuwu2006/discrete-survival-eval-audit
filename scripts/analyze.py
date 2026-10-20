"""Reproduce every table in the paper from a set of trained variants.

Reads `<variant>.foldlog.jsonl` (C-index + edges) and `<variant>.perpatient.jsonl`
(logits + risk + time + event + cohort) produced by
`train_boundary_experiment.py`, and prints the six paper tables:

    Table I  : naive vs piecewise-constant node AUC (the artifact)
    Table II : Harrell + Uno IPCW C-index, paired vs baseline
    Table III: three statistical views of the C-index differences
    Table IV : node-level AUC deltas (piecewise-constant)
    Table V  : the four interpolation schemes head-to-head
    Table VI : per-cohort C-index

Usage:
    python scripts/analyze.py --result-dir results [--base S_t0_q12]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.interpolation import SURVIVAL_SCHEMES
from src.metrics import (harrell_cindex, make_censoring_survival,
                         time_dependent_auc, uno_cindex)
from src.stats import bh_fdr, corrected_t_p, repeat_level_p, wilcoxon_p

HORIZONS = (6.0, 12.0, 24.0, 36.0, 48.0)
TAU = 60.0

DEFAULT_VARIANTS = [
    ("Baseline (event-quantile)", "S_t0_q12"),
    ("Equal-width", "S_t1a_u12"),
    ("All-time quantile", "S_t1b_a12"),
    ("Log-spacing", "S_t_dcslog"),
    ("Clinical-anchored no36", "S_t2a2_no36"),
    ("AAIC", "S_s2_aaic_fast"),
]


def load_foldlog(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["fold"]] = r
    return out


def load_perpatient(path):
    folds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = folds.setdefault(r["fold"], {"time": [], "event": [], "risk": [],
                                             "cohort": [], "logits": []})
            d["time"].append(r["time"])
            d["event"].append(r["event"])
            d["risk"].append(r["risk"])
            d["cohort"].append(r["cohort"])
            d["logits"].append(r["logits"])
    return folds


def node_auc_across(logits_by_fold, time_event_folds, edges_by_fold, fn, horizon):
    vals = []
    for f in sorted(logits_by_fold):
        logits = np.array(logits_by_fold[f], dtype=np.float64)
        edges = np.array(edges_by_fold[f], dtype=np.float64)
        t = np.array(time_event_folds[f]["time"])
        e = np.array(time_event_folds[f]["event"], dtype=np.int64)
        g = make_censoring_survival(t, e)
        s = fn(logits, edges, horizon)
        vals.append(time_dependent_auc(-s, t, e, horizon, g))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--base", default="S_t0_q12")
    ap.add_argument("--n-total", type=int, default=1106,
                    help="total patients (for the Nadeau-Bengio factor)")
    args = ap.parse_args()

    variants = DEFAULT_VARIANTS
    base = args.base

    # load everything
    fl, pp = {}, {}
    for _, key in variants:
        f1, f2 = os.path.join(args.result_dir, f"{key}.foldlog.jsonl"), \
                 os.path.join(args.result_dir, f"{key}.perpatient.jsonl")
        if not (os.path.isfile(f1) and os.path.isfile(f2)):
            print(f"[skip] {key} (missing)", file=sys.stderr)
            continue
        fl[key] = load_foldlog(f1)
        pp[key] = load_perpatient(f2)

    base_edges = {f: fl[base][f]["edges"] for f in fl[base]}
    base_pp = pp[base]
    n2 = float(np.mean([len(base_pp[f]["time"]) for f in base_pp]))
    n1 = args.n_total - n2

    # ---------------- helper to compute node AUC per scheme ----------------
    def auc_table(key, scheme, horizons):
        fn = SURVIVAL_SCHEMES[scheme]
        edges = {f: fl[key][f]["edges"] for f in fl[key]}
        out = {}
        for h in horizons:
            out[h] = node_auc_across(pp[key], pp[key], edges, fn, h)
        return out

    # ========================= Table I =========================
    print("=" * 72)
    print("Table I. naive vs piecewise-constant node AUC (the artifact)")
    print("=" * 72)
    print(f"{'strategy':26s} | {'AUC@6m naive':>13s} {'-> pc':>13s} | "
          f"{'AUC@12m naive':>14s} {'-> pc':>13s}")
    for name, key in variants:
        if key not in fl:
            continue
        n6 = node_auc_across(pp[key], pp[key], {f: fl[key][f]["edges"] for f in fl[key]},
                             SURVIVAL_SCHEMES["naive"], 6.0).mean()
        p6 = node_auc_across(pp[key], pp[key], {f: fl[key][f]["edges"] for f in fl[key]},
                             SURVIVAL_SCHEMES["pc"], 6.0).mean()
        n12 = node_auc_across(pp[key], pp[key], {f: fl[key][f]["edges"] for f in fl[key]},
                              SURVIVAL_SCHEMES["naive"], 12.0).mean()
        p12 = node_auc_across(pp[key], pp[key], {f: fl[key][f]["edges"] for f in fl[key]},
                              SURVIVAL_SCHEMES["pc"], 12.0).mean()
        print(f"{name:26s} | {n6:13.4f} {p6:13.4f} | {n12:14.4f} {p12:13.4f}")

    # ========================= Table II + III =========================
    cb = {f: fl[base][f]["metrics"]["cindex"] for f in fl[base]}
    print("\n" + "=" * 72)
    print("Table II. Harrell / Uno IPCW C-index and repeat-level significance")
    print("=" * 72)
    print(f"{'strategy':26s} {'Harrell':>8s} {'dC':>7s} {'Uno(60)':>8s} "
          f"{'dUno':>7s} {'rep-p':>7s} {'win':>6s}")

    rows3 = []  # for Table III
    for name, key in variants:
        if key not in fl or key == base:
            continue
        cv = {f: fl[key][f]["metrics"]["cindex"] for f in fl[key]}
        fs = sorted(set(cb) & set(cv))
        a = np.array([cv[f] for f in fs])
        b = np.array([cb[f] for f in fs])
        d = a - b
        # uno
        ua, ub = [], []
        for f in fs:
            t = np.array(pp[key][f]["time"]); e = np.array(pp[key][f]["event"], dtype=np.int64)
            tb = np.array(base_pp[f]["time"]); eb = np.array(base_pp[f]["event"], dtype=np.int64)
            ua.append(uno_cindex(np.array(pp[key][f]["risk"]), t, e, TAU, make_censoring_survival(t, e)))
            ub.append(uno_cindex(np.array(base_pp[f]["risk"]), tb, eb, TAU, make_censoring_survival(tb, eb)))
        du = np.array(ua) - np.array(ub)
        mdiff, tstat, rp = repeat_level_p(cv, cb)
        wp = wilcoxon_p(d)
        nbp = corrected_t_p(d, n2, n1)
        rows3.append((name, d.mean(), wp, rp, nbp))
        print(f"{name:26s} {a.mean():8.4f} {d.mean():+7.4f} {np.mean(ua):8.4f} "
              f"{du.mean():+7.4f} {rp:7.3f} {int((d > 0).sum()):5d}/{len(fs)}")

    print("\n" + "=" * 72)
    print("Table III. three statistical views of the C-index differences")
    print("=" * 72)
    print(f"{'strategy':26s} {'dC':>8s} {'Wilcoxon':>9s} {'repeat':>8s} {'NB-approx':>10s}")
    for name, dc, wp, rp, nbp in rows3:
        print(f"{name:26s} {dc:+8.4f} {wp:9.4f} {rp:8.4f} {nbp:10.4f}")

    # ========================= Table IV =========================
    print("\n" + "=" * 72)
    print("Table IV. node-level AUC deltas vs baseline (piecewise-constant)")
    print("=" * 72)
    print(f"{'strategy':26s}" + "".join(f"{'A'+str(int(h)):>9s}" for h in HORIZONS))
    base_pc = auc_table(base, "pc", HORIZONS)
    for name, key in variants:
        if key not in pp or key == base:
            continue
        vpc = auc_table(key, "pc", HORIZONS)
        cells = [f"{(vpc[h] - base_pc[h]).mean():+9.4f}" for h in HORIZONS]
        print(f"{name:26s}" + "".join(cells))

    # ========================= Table V =========================
    print("\n" + "=" * 72)
    print("Table V. four interpolation schemes head-to-head (6/12/48 m)")
    print("=" * 72)
    for name, key in variants:
        if key not in fl:
            continue
        print(f"--- {name} ---")
        edges = {f: fl[key][f]["edges"] for f in fl[key]}
        print(f"{'scheme':8s} " + "".join(f"{'A'+str(int(h)):>8s}" for h in (6, 12, 48)))
        for scheme in ("naive", "step", "linear", "pc"):
            fn = SURVIVAL_SCHEMES[scheme]
            cells = [f"{node_auc_across(pp[key], pp[key], edges, fn, float(h)).mean():8.4f}"
                     for h in (6, 12, 48)]
            print(f"{scheme:8s} " + "".join(cells))

    # ========================= Table VI =========================
    print("\n" + "=" * 72)
    print("Table VI. per-cohort Harrell C-index")
    print("=" * 72)
    cohorts = sorted({c for key in variants if key in pp
                      for f in pp[key] for c in pp[key][f]["cohort"]})
    print(f"{'strategy':26s}" + "".join(f"{c:>12s}" for c in cohorts))
    for name, key in variants:
        if key not in pp:
            continue
        cells = []
        for c in cohorts:
            vals = []
            for f in sorted(pp[key]):
                rows = pp[key][f]
                idx = [i for i, cc in enumerate(rows["cohort"]) if cc == c]
                if len(idx) < 10:
                    continue
                sub_risk = np.array([rows["risk"][i] for i in idx])
                sub_t = np.array([rows["time"][i] for i in idx])
                sub_e = np.array([rows["event"][i] for i in idx], dtype=np.int64)
                vals.append(harrell_cindex(sub_risk, sub_t, sub_e))
            cells.append(f"{np.mean(vals):12.4f}" if vals else f"{'—':>12s}")
        print(f"{name:26s}" + "".join(cells))


if __name__ == "__main__":
    main()