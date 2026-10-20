# Data

This repository intentionally ships **no patient data**. The experiments in the
paper were run on a harmonized two-center collection of 1,106 patients:

- **UCSF-PDGM** — 495 preoperative diffuse gliomas.
  *E. Calabrese et al., "The University of California San Francisco Preoperative
  Diffuse Glioma MRI Dataset," Radiology: Artificial Intelligence, 4(5), 2022.*
- **UPENN-GBM** — 611 glioblastomas.
  *S. Bakas et al., "The University of Pennsylvania glioblastoma (UPenn-GBM)
  cohort: advanced MRI, clinical, genomics, & radiomics," Scientific Data, 9, 2022.*

Both are public but require a data-use agreement; do not redistribute the
patient-level records. The derived artifacts below must never be committed here.

## Pipeline (what the code expects)

The training script (`scripts/train_boundary_experiment.py`) consumes pre-extracted
features, following the frozen-encoder protocol in the paper (Section III-B):

1. **Feature extraction** — pass each 3-channel volume (T1c/T2/FLAIR, 3×128³,
   clipped + z-scored per channel) through a frozen SwinUNETR.swinViT encoder,
   take the 4th hidden state, and pool it with anatomical SALP weights into
   `z_salp` (1152-vec). Also store `z_f4` (384), five tumor-global scalars, and
   two clinical scalars (age, sex).

2. **Per-patient feature file** — for each patient, a `.npz` with keys:
   `z_salp`, `z_f4`, `tumor_global`, `clinical`, `surv_time`, `event`, `label`, `cohort`.

3. **Patient table** — a pickle (or equivalent) list of records, each with at
   least `patient_id` and the folder containing `{patient_id}.npz`.

Place these under `data/` (ignored by git) and point the training script at them;
see `scripts/train_boundary_experiment.py --help`.

## Minimal reproduction without any data

`demo/demo.py` reproduces the paper's *core* finding — the first-box-constant
interpolation artifact — on synthetic logits and boundaries. It needs only
numpy and has no data/GPU dependency:

```
python demo/demo.py
```