"""Train the boundary-geometry experiment and SAVE per-patient interval logits.

This is the reproduction entry point for the paper. It trains the same
DiscreteSurvivalHead on the provided frozen features under each boundary
strategy and writes two artifacts per variant:

    <variant>.foldlog.jsonl    : per-fold C-index, n_bins, and the edge set
    <variant>.perpatient.jsonl : per-patient risk/time/event/cohort AND logits

Saving the logits (rather than a single interpolated survival) lets the
downstream script `analyze.py` recompute node-level AUC under ANY of the four
interpolation schemes (naive / step / linear / pc), which is exactly the audit
the paper performs. This also removes the two defects the paper documents:
    1. the head width is len(edges)-1 (no untrained "ghost" intervals);
    2. no naive first-box-constant survival is ever persisted.

No data is bundled. Point the script at the pre-extracted features described in
data/README.md:

    python scripts/train_boundary_experiment.py \
        --bins log --n 12 --variant S_t_dcslog \
        --pkl data/patients.pkl --feat data/features --out results

Run one variant per boundary strategy (quantile/uniform/alltime/log/custom/aaic).
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.survival import (DiscreteSurvivalHead, discrete_survival_nll,
                          hazard_survival_risk)
from src.boundaries import build_edges
from src.metrics import harrell_cindex

HORIZONS = (6.0, 12.0, 24.0, 36.0, 48.0)
HIDDEN = 64
DROPOUT = 0.1
EPOCHS = 200
LR = 1e-4
BATCH = 64
WD = 1e-5


# --------------------------------------------------------------------------- #
# data loading (mirrors the paper's frozen-encoder protocol)
# --------------------------------------------------------------------------- #
def load_data(pkl_path, feat_dir):
    with open(pkl_path, "rb") as f:
        patients = pickle.load(f)
    n = len(patients)
    z_salp = np.zeros((n, 1152), dtype=np.float32)
    tg = np.zeros((n, 5), dtype=np.float32)
    cl = np.zeros((n, 2), dtype=np.float32)
    times = np.zeros(n, dtype=np.float64)
    events = np.zeros(n, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    cohorts, pids = [], []
    for i in range(n):
        pid = patients[i]["patient_id"]
        pids.append(pid)
        d = np.load(os.path.join(feat_dir, f"{pid}.npz"))
        z_salp[i] = d["z_salp"]
        tg[i] = d["tumor_global"]
        cl[i] = d["clinical"]
        times[i] = d["surv_time"]
        events[i] = d["event"]
        labels[i] = d["label"]
        cohorts.append(str(d["cohort"]))
    return z_salp, tg, cl, times, events, labels, cohorts, pids


def make_splits(n, labels, folds, repeats, seed):
    from sklearn.model_selection import StratifiedKFold
    splits = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for train_idx, val_idx in skf.split(np.zeros(n), labels):
            splits.append((train_idx, val_idx))
    return splits


def preprocess_features(tg, cl, tr):
    tg = np.log1p(np.asarray(tg, dtype=np.float64))
    lo, hi = np.percentile(tg[tr], 1.0, axis=0), np.percentile(tg[tr], 99.0, axis=0)
    tg = np.clip(tg, lo, hi)
    m, s = tg[tr].mean(axis=0), tg[tr].std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    tg = ((tg - m) / s).astype(np.float32)

    cl = np.asarray(cl, dtype=np.float64).copy()
    cm, cs = cl[tr][:, 0].mean(), cl[tr][:, 0].std()
    cs = 1.0 if cs < 1e-12 else cs
    cl[:, 0] = (cl[:, 0] - cm) / cs
    cl = cl.astype(np.float32)
    return tg, cl


def build_x(z_salp, tg, cl, tr, va):
    tg_p, cl_p = preprocess_features(tg, cl, tr)
    X = np.concatenate([z_salp, tg_p, cl_p], axis=1)
    return X[tr], X[va]


def train_head(Xtr, times_tr, events_tr, edges, n_bins, seed, device):
    import torch
    model = DiscreteSurvivalHead(Xtr.shape[1], HIDDEN, n_bins, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = len(times_tr)
    idx = np.arange(n)
    for ep in range(EPOCHS):
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(idx)
        model.train()
        for b in range(0, n, BATCH):
            bi = idx[b:b + BATCH]
            x = torch.from_numpy(Xtr[bi]).float().to(device)
            t = torch.from_numpy(times_tr[bi]).float().to(device)
            e = torch.from_numpy(events_tr[bi]).float().to(device)
            logit = model(x)
            loss = discrete_survival_nll(logit, t, e, edges)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", choices=["quantile", "uniform", "alltime", "custom", "aaic", "log"],
                    default="quantile")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--edges", default=None, help="csv, for --bins custom")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--pkl", required=True, help="patients pickle (list of dict)")
    ap.add_argument("--feat", required=True, help="dir of {patient_id}.npz features")
    ap.add_argument("--out", default="results", help="output dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    os.makedirs(args.out, exist_ok=True)
    foldlog = os.path.join(args.out, f"{args.variant}.foldlog.jsonl")
    perpatient = os.path.join(args.out, f"{args.variant}.perpatient.jsonl")

    z_salp, tg, cl, times, events, labels, cohorts, pids = load_data(args.pkl, args.feat)
    n = len(times)
    splits = make_splits(n, labels, args.folds, args.repeats, args.seed)

    results = []
    if os.path.isfile(foldlog):
        with open(foldlog, encoding="utf-8") as f:
            results = [json.loads(x) for x in f if x.strip()]
    print(f"[train] variant={args.variant} bins={args.bins} patients={n} "
          f"resume={len(results)}/{len(splits)}")

    for i, (tr, va) in enumerate(splits):
        if i < len(results):
            continue
        edges = build_edges(args.bins, times[tr], events[tr], args.n, args.edges)
        n_bins = len(edges) - 1
        Xtr, Xva = build_x(z_salp, tg, cl, tr, va)
        model = train_head(Xtr, times[tr], events[tr], edges, n_bins, args.seed, device)

        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(Xva).float().to(device)
            logit = model(x).cpu().numpy()
        _, _, risk = hazard_survival_risk(logit, edges)
        c = harrell_cindex(risk, times[va], events[va])

        m = {"variant": args.variant, "fold": i + 1, "metrics": {"cindex": float(c)},
             "n_bins": n_bins, "bins": args.bins, "edges": [float(e) for e in edges]}
        results.append(m)
        with open(foldlog, "a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        with open(perpatient, "a", encoding="utf-8") as f:
            for jj, idx in enumerate(va):
                rec = {"variant": args.variant, "fold": i + 1,
                       "patient_id": pids[idx], "cohort": cohorts[idx],
                       "risk": float(risk[jj]), "time": float(times[idx]),
                       "event": int(events[idx]),
                       "logits": [float(v) for v in logit[jj]]}
                f.write(json.dumps(rec) + "\n")

        ci = float(np.mean([r["metrics"]["cindex"] for r in results]))
        print(f"  [fold {i + 1}/{len(splits)}] cindex={c:.4f} running={ci:.4f}", flush=True)

    cs = [r["metrics"]["cindex"] for r in results]
    print(f"\n[train done] {args.variant} n_folds={len(cs)} "
          f"cindex={np.mean(cs):.4f} +/- {np.std(cs):.4f}")


if __name__ == "__main__":
    main()