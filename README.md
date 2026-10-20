# Discrete-Survival Evaluation-Interpolation Audit

A self-contained reproduction of a **deterministic evaluation artifact** in
discrete-time survival analysis, plus the corrected evaluation and the resulting
re-calibration of every boundary-geometry conclusion.

> **Status.** This is a research artifact / methodological note, not a completed
> journal submission. It exists to (i) document a failure mode that is easy to
> introduce and easy to miss, and (ii) make the corrected analysis fully
> reproducible. See `paper/manuscript.md` for the write-up.

## The core finding in one paragraph

Discrete-time survival models output a per-interval hazard and the survival
curve is a *step function* defined only at the interval edges. Evaluating it at
a horizon that falls **inside** an interval therefore needs an interpolation. A
common shortcut — "*survival = 1.0 whenever the horizon is inside the first
box*" — makes the predicted survival identical for every patient at that
horizon, which pins the time-dependent AUC at **exactly 0.5000 (standard
deviation 0)**. This is a bookkeeping error, not an empirical effect, but it
looks exactly like a dramatic "boundary-geometry effect" (an apparent ≈0.20 AUC
drop for boundary sets with a coarse first box). Replacing the shortcut with a
piecewise-constant-hazard interpolation (the constant-hazard case of
Kvamme & Borgan's CHI) restores the AUC to 0.70–0.72, and the collapse
disappears. Two further results follow:

1. **The choice among correct interpolations barely matters for discrimination.**
   Linear interpolation (the scheme used by DCS) and piecewise-constant-hazard
   interpolation give node AUCs differing by **< 0.001** — both are monotone in
   the same per-interval hazards, so the patient ranking is unchanged.
2. **The residual boundary-geometry effect is small and marginally significant.**
   On C-index, dense early subdivision beats event-quantile spacing by
   +0.006 to +0.009, but significance is sensitive to how the non-independent
   5×5 repeated-CV folds are handled (repeat-level paired test p = 0.002–0.028;
   Nadeau–Bengio approximation p = 0.13–0.49).

## Repository layout

```
src/            self-contained library (no data dependency)
  survival.py     head, discrete NLL, risk, and the documented naive bug
  interpolation.py  the four survival-interpolation schemes (naive/step/linear/pc)
  boundaries.py   the six interval-boundary strategies (+ demoted AAIC)
  metrics.py      Harrell C-index, KM, IPCW AUC/Brier, Uno C-index
  stats.py        inference for non-independent folds + BH-FDR
scripts/
  train_boundary_experiment.py   train variants, save per-patient logits
  analyze.py                     regenerate every table in the paper
demo/
  demo.py                        synthetic end-to-end demo (no data, no GPU)
paper/
  manuscript.md                  the write-up
  references.bib                 bibliography (incl. the two datasets)
data/README.md                   how to obtain the public datasets + feature format
```

## Quick start (no data, no GPU)

```bash
pip install numpy scipy torch scikit-learn   # torch/sklearn only needed for training
python demo/demo.py
```

`demo.py` reproduces the artifact on synthetic logits:

```
Scenario A: coarse first box [0, 10] covers the 6-month horizon
    naive    AUC@6m = 0.5000   S(high-risk)=1.000  S(low-risk)=1.000
    step     AUC@6m = 0.8356   S(high-risk)=0.047  S(low-risk)=0.953
    linear   AUC@6m = 0.8356   S(high-risk)=0.428  S(low-risk)=0.972
    pc       AUC@6m = 0.8356   S(high-risk)=0.161  S(low-risk)=0.971
```

`step` differs from `naive` *only* inside the first box, yet restores the full
AUC — which localizes the entire ≈0.2 collapse to the single "return 1.0 in the
first box" line.

## Full reproduction

The paper's experiment (1,106 patients, six boundary strategies, 25 folds)
requires the two public cohorts and a frozen SwinUNETR feature pass — see
`data/README.md`. Once the features are extracted:

```bash
# one run per boundary strategy (6 total)
python scripts/train_boundary_experiment.py \
  --bins quantile --n 12 --variant S_t0_q12 \
  --pkl data/patients.pkl --feat data/features --out results

python scripts/train_boundary_experiment.py \
  --bins uniform  --n 12 --variant S_t1a_u12 \
  --pkl data/patients.pkl --feat data/features --out results
# ... quantile | uniform | alltime | log | custom | aaic ...

# regenerate the six tables
python scripts/analyze.py --result-dir results
```

The trainer saves per-patient **logits** (not a single interpolated survival),
so `analyze.py` can recompute node AUC under any of the four interpolation
schemes — which is precisely the audit the paper performs.

## Results summary (25-fold paired)

| result | strength | value |
|---|---|---|
| naive first-box-constant → node AUC | deterministic (sd = 0) | **0.5000** |
| same node, piecewise-constant | — | 0.70–0.72 |
| linear vs pc node AUC | agrees | < 0.001 |
| C-index, dense early region vs baseline | marginal | +0.006 to +0.009 |
| all-time-quantile vs baseline (DCS "quantile") | stable null | ≈ +0.0015 (p ≈ 0.35–0.40) |

Full tables, statistical caveats, and the AAIC demotion are in
`paper/manuscript.md`.

## Citation

```bibtex
@misc{discrete-survival-eval-audit,
  author       = {Anonymous},
  title        = {Naive Survival Interpolation Fakes Boundary-Geometry Effects in Discrete-Time Survival},
  year         = {2026},
  note         = {Research artifact},
}
```

## License & data disclaimer

Code is MIT (`LICENSE`, copyright "Anonymous"); update the holder if you adopt it. **No
patient data is included in this repository.** The UCSF-PDGM and UPENN-GBM
cohorts are public but require a data-use agreement; re-distribute them under
their own terms only.