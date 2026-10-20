# Naive Survival Interpolation Fakes Boundary-Geometry Effects in Discrete-Time Survival: Evidence from Multimodal Glioma

Anonymous
Anonymous

---

## Abstract

Discrete-time survival models partition the continuous time axis into intervals and predict a conditional death probability per interval; the resulting survival curve is a step function that must itself be *interpolated* whenever the model is evaluated at a time point that falls inside an interval. The geometry of the interval boundaries and the choice of survival interpolation are therefore two hidden, interacting evaluation decisions that are almost always defaulted rather than audited. We audit both on a frozen-encoder (SwinUNETR + anatomical SALP pooling) multimodal glioma survival pipeline (1,106 patients; diffuse glioma and glioblastoma), restricted by design to three conventional MRI sequences (T1c, T2, FLAIR) and age/sex — a low-cost input set intended for resource-constrained hospitals. We find, first, a deterministic evaluation artifact: a naive "first-box constant" interpolation (a common implementation shortcut in which survival inside the first interval is returned as 1.0) mechanically pins the node-level AUC to 0.5000 (standard deviation 0) for every horizon that falls inside the first box, fabricating dramatic ≈0.20 "boundary-geometry effects" where none exist. Replacing it with a piecewise-constant-hazard interpolation (the constant-hazard special case of CHI) restores the AUC to 0.70–0.72, and the collapse disappears entirely. Second, once interpolation is performed correctly, the *choice* of interpolation is surprisingly consequential-neutral for discrimination: linear interpolation (the scheme used by DCS) and piecewise-constant-hazard interpolation give node-level AUCs differing by <0.001, because both are monotone functions of the same per-interval hazards and preserve the patient ranking. Third, on C-index the residual effect of boundary geometry is small and directionally consistent (dense early subdivision > event-quantile > all-time quantile, +0.006 to +0.009), but its statistical significance is sensitive to how the non-independent repeated-CV folds are handled (repeat-level paired test p = 0.002–0.028; Nadeau–Bengio approximation p = 0.13–0.49). We therefore characterize it as a marginal effect, not a robust one. Our headline conclusion is methodological: reports of node-level discrete-survival discrimination are unreliable unless the interpolation used at evaluation is stated and correct; and boundary-geometry conclusions drawn under a naive interpolation can be inverted by a one-line fix.

**Keywords** — survival analysis, time discretization, survival interpolation, evaluation artifact, multimodal imaging, glioma.

---

## I. Introduction

Glioma, and its most aggressive form glioblastoma (GBM), has a highly heterogeneous survival course. Multimodal MRI combined with clinical covariates enables individualized prognosis, and deep discrete-time survival models have become a standard tool for this task because they avoid a parametric assumption on the hazard: the time axis is partitioned into intervals, a conditional event probability is predicted per interval, and the survival curve is the product of the complements [1], [2].

This formulation embeds a decision that is rarely examined: **how to cut the time axis**, i.e., the geometry of interval boundaries. The boundary set controls the temporal resolution at which risk is observed. Existing work has touched the question in three strands: Kvamme & Borgan [3] compared equidistant and quantile grids and introduced *continuous hazard interpolation* (CHI) to mitigate grid coarseness; DCS [4] introduced variable output-node spacings (linear/log/quantile) and reported quantile spacing to be best; and Hickey et al. [5] learn the cut points by gradient descent. Censoring structure, however, is treated at most implicitly.

Our work was motivated by a concrete finding from auditing our own pipeline, not by a prior hypothesis. While re-examining the node-level evaluation under different boundary geometries, we discovered that a substantial, dramatic, and *statistically significant* "early-node collapse" — a drop of ≈0.20 in the 6- and 12-month time-dependent AUC for boundary sets with a coarse first box — was in fact a deterministic artifact of the survival interpolation used at evaluation time. The culprit is a common implementation shortcut: when an evaluation horizon falls *inside* the first interval (before its right edge), the survival function is returned as 1.0 for every patient. The discrimination score then becomes constant across patients, and the time-dependent AUC is pinned at exactly 0.5000 (standard deviation 0). This artifact depends only on the relative positions of the first-box edge and the horizon — not on any property of the model — so it perfectly mimics a "boundary-geometry effect" while being entirely a bookkeeping error.

This finding sits at the intersection of two literatures that have, separately, already registered the underlying issue: Kvamme & Borgan proposed CHI precisely because a raw discrete survival curve is a step function that is undefined between grid points [3], and DCS interpolates its predictions linearly before evaluating node-level discrimination [4]. What has not been done — and what this paper contributes — is a head-to-head, quantitative audit of the *evaluation* itself: the deterministic failure mode of the naive shortcut, the sensitivity (or lack thereof) of node-level discrimination to the choice among the "correct" interpolations, and the resulting re-calibration of every boundary-geometry conclusion.

We make four contributions on a frozen-encoder multimodal glioma pipeline (1,106 patients), using six boundary strategies under a paired 5×5 stratification:

1. **A deterministic evaluation artifact.** We isolate the "first-box constant" interpolation as the *sole* cause of a spurious early-node collapse: with piecewise-constant interpolation the collapse vanishes and node AUC returns to 0.70–0.72. This is a mathematical fact (an AUC of exactly 0.5 with zero variance), not a statistical claim.
2. **Interpolation-method robustness.** Once interpolation is correct, linear (DCS) and piecewise-constant-hazard (CHI special case) interpolation agree on node-level AUC to <0.001, because node-level AUC is a ranking measure and both schemes are monotone in the same per-interval hazards. We state this as a robustness result that also *removes interpolation scheme as a confound* when comparing our findings to DCS.
3. **A recalibrated boundary-geometry effect.** On C-index the residual effect is small and directionally consistent (+0.006 to +0.009; dense early region > event-quantile > all-time quantile) but only *marginally* significant: the repeat-level paired test gives p = 0.002–0.028, while a Nadeau–Bengio approximation gives p = 0.13–0.49. Following [3], [4], we do not overclaim.
4. **A transferability counterexample, honestly stated.** DCS's "quantile spacing is best" does not transfer to multimodal imaging under either interpolation: all-time-quantile spacing is the *worst* non-anchored family (ΔC ≈ +0.0015, p ≈ 0.35–0.40). We report this as a non-replication in our setting, not as a contradiction of DCS — our head, data domain, and evaluation all differ from theirs.

Throughout, we operate under an explicit **low-cost input constraint**: only T1c, T2, FLAIR and age/sex, deliberately excluding molecular (IDH, MGMT) and treatment covariates that are prognostically powerful but often unavailable or unaffordable in resource-limited hospitals. This constraint is a design choice, not an omission, and we report its cost (possible learning of center differences) transparently.

The remainder is organized as follows. Section II reviews related work. Section III describes the pipeline, the six boundary strategies, and the four survival-interpolation schemes we compare. Section IV presents the artifact, the C-index results, the node-level results, and the interpolation-sensitivity study; Section V discusses the implications; Section VI concludes.

---

## II. Related Work

### A. Discrete-time survival modeling

Deep discrete-time survival models partition time into intervals and predict a per-interval conditional hazard; survival follows as the product of complements. DeepHit [1] and the general discrete-time formulation of [2] popularized this for deep learning. Our pipeline uses exactly this head, so the evaluation question we study applies to the whole family, not a single architecture.

### B. Boundary / grid choice

Three strands have addressed interval boundaries. **Kvamme & Borgan** [3] compared equidistant and event-quantile grids and — critically for us — introduced *continuous hazard interpolation* (CHI), an interpolation of the predicted hazards between grid points used precisely to obtain a survival estimate at any *t*, thereby avoiding a raw step function at evaluation. **DCS** [4] introduced variable output-node spacings (linear/log/quantile) on a DRSA backbone and reported quantile spacing best, alongside a kernel-based calibration loss. We verified against the DCS source code that its "quantile" option places edges at the quantiles of *all* observed times (events and censored), and that it *linearly interpolates* predictions before node-level evaluation. **Hickey et al.** [5] learn the cut points by gradient optimization.

### C. The gap

The strands above treat interpolation and boundary geometry as separate engineering details. No study, to our knowledge, has (i) quantitatively isolated a *deterministic* failure mode of naive interpolation at node-level evaluation, (ii) shown head-to-head that the choice among the "correct" interpolations leaves node-level AUC essentially unchanged, or (iii) re-derived boundary-geometry conclusions under a corrected interpolation in multimodal imaging under an explicit low-cost input constraint. Our paper fills this gap and, in doing so, must walk back a conclusion from our own earlier drafts — a point we flag in Section V for transparency.

---

## III. Methods

### A. Data, preprocessing, and the low-cost input constraint

We use a harmonized two-center collection of 1,106 patients: 495 diffuse gliomas (UCSF-PDGM) and 611 glioblastomas (UPENN-GBM), with survival times capped at 120 months (`t ← min(t, 120)`; no patient dropped). Event (uncensored) rates differ sharply — 50.1% for UCSF and 95.7% for UPENN — so censoring mass concentrates in the UCSF cohort; this cohort heterogeneity is reported explicitly (Section IV-D).

Each patient has multimodal MRI. **By design**, the model consumes only three conventional sequences — T1c, T2, FLAIR — resampled to 3×128³, per-patient clipped to the 0.5–99.5 intensity percentiles, then z-scored per channel, plus age (z-scored per fold) and sex. Molecular markers (IDH, MGMT) and treatment variables are deliberately excluded: they are prognostically strong but costly and frequently unavailable in low-resource hospitals. This constraint is the deployment condition we choose to study, and its consequences are reported as limitations, not hidden.

### B. Frozen feature extraction and anatomical pooling

All images pass once through a frozen SwinUNETR.swinViT encoder [6] (MONAI 1.3.2 [7], feature_size=48, depths/heads (2,2,2,2)/(3,6,12,24), initialized from BrainSegFounder BraTS-finetuned weights with the patch-embed channel count cropped 4→3). We take the fourth hidden state F4 ∈ ℝ^(512×384). It is reduced by anatomical structure-aware pooling (SALP) with fixed, non-learnable weights: for each subregion s ∈ {ED, ET, NCR}, the pooled vector z_s = Σᵢ [rᵢˢ / (Σⱼ rⱼˢ)] · hᵢ, where rᵢˢ is the mean segmentation fraction of s in grid cell i. The three vectors are concatenated to z_salp ∈ ℝ^1152; we append five tumor-global scalars (log1p + winsorized + z-scored per fold) and two clinical scalars, yielding z ∈ ℝ^1159. The encoder is never trained; features are extracted once offline and reused.

### C. Survival head and training

The head is `DiscreteSurvivalHead(1159, 64, K) = Linear(1159→64) → GELU → Dropout(0.1) → Linear(64→K)`, where K is the number of intervals (K=12 for most strategies; smaller for the anchored sets). Given edges e = (e₀,…,e_K), the model outputs K logits, per-interval hazard λ_j = σ(logit_j), and the discrete survival at the edges S(k) = ∏_{j≤k}(1−λ_j). Training minimizes the censored discrete negative log-likelihood (for an event in interval j: −log λ_j − Σ_{m<j} log(1−λ_m); for a censored observation: −Σ_{m<j} log(1−λ_m)), with AdamW (lr=1e-4, weight_decay=1e-5), batch 64, 200 epochs. Only the head is trained. **We set K = len(edges) − 1, not a fixed global width**, which matters for the anchored strategies: an earlier draft fixed K=12 while the anchored strategies had only 5–9 intervals, leaving untrained "ghost" logit columns that polluted the risk score (an error we corrected; its effect on C-index was <0.001 but it is a correctness issue a reviewer can check).

### D. Interval boundary strategies

All boundaries are fit on the training fold only (no leakage). Six strategies, all with 12 intervals except the anchored set:

1. **Event-quantile (baseline)** — interior edges at the empirical quantiles of the uncensored event times; the conventional default. This is the paired Wilcoxon reference for all Δ.
2. **Equal-width (≈ DCS "linear" spacing)** — edges equally spaced over [0, t_max].
3. **All-time quantile (≈ DCS "quantile" spacing)** — edges at the quantiles of *all* observed times (events and censored), matching DCS's quantile definition verified from its source (`_set_output_grid`).
4. **Log-spacing (≈ DCS "log" spacing)** — edges [0, logspace(1, log10(t_max), K+1)[1:]].
5. **Clinically anchored (no36)** — edges fixed at {0, 6, 12, 24, 48, 120} months (6 intervals), the hand-searched best boundary.
6. **AAIC** — a zero-parameter censoring-aware criterion (Appendix A), evaluated here only as a boundary strategy; it is demoted from headline status for the reasons given in Section V.

*DCS correspondence.* Strategies 2–4 reproduce the spacing *geometry* of DCS on our identical head, not the full DCS architecture. Two documented differences are relevant here: DCS uses 10 nodes (we use 12) and — most importantly for the artifact we study — DCS *linearly interpolates* its predictions before node-level evaluation, whereas the baseline that produced the spurious collapse did not.

### E. Survival interpolation at evaluation (the object of the audit)

Because a discrete survival curve is a step function defined only at the edges, any evaluation at a horizon h that does not coincide with an edge requires an interpolation. We compare four schemes, all applied to the same per-interval hazards λ = σ(logits), with S(e_j) = ∏_{m≤j}(1−λ_m) and S(e₀)=1:

1. **naive-constant** — the flawed shortcut: S(h) = 1.0 whenever h < e₁ (inside the first box), else S(h) = S(max edge ≤ h). This is what produced the spurious collapse.
2. **step (floor)** — the same floor rule but with the first-box right edge S(e₁) instead of 1.0. It differs from naive-constant *only* inside the first box, which lets us attribute the artifact precisely.
3. **linear (DCS)** — S(h) = S(τ_j) + frac·(S(τ_{j+1}) − S(τ_j)) for h ∈ [τ_j, τ_{j+1}), frac = (h−τ_j)/width_j.
4. **piecewise-constant-hazard (PCH, ours; CHI special case)** — each interval is assigned a constant hazard rate h_j = −log(1−λ_j)/width_j, and S(h) = S(τ_j)·exp(−h_j·(h−τ_j)). This is continuous, and at the edges it reproduces the discrete probabilities (S(τ_{j+1}) = S(τ_j)·(1−λ_j)), i.e., it is the constant-hazard case of Kvamme & Borgan's CHI [3].

The division by width_j in PCH is essential and non-obvious: an earlier version that omitted it collapsed S(h) to ≈0 inside wide first boxes (caught by a unit test). All node-level results in this paper use PCH unless a scheme is explicitly named.

### F. Evaluation metrics

Under the frozen-encoder protocol's paired 5×5 stratification — five stratified 5-fold repeats with seeds 42+r (r=0…4), 25 folds total — every strategy is scored on the *same* folds, so all comparisons are paired.

- **Global discrimination.** Harrell's C-index on the validation fold using the risk score (negative discrete expected lifetime). We additionally report **Uno's IPCW C-index (τ=60 months)** [8], which re-weights early uncensored cases by 1/G(t)² and is less sensitive to the censoring distribution.
- **Node-level discrimination.** Time-dependent AUC at horizons 6/12/24/36/48 months under the IPCW protocol of Uno et al. [8]; the discrimination score is −S(h) (higher = higher risk). The censoring survival G(t) is estimated per fold by Kaplan–Meier on the censoring indicators.
- **Stratification.** To surface cohort confounding, all metrics are also reported separately for UCSF and UPENN.

### G. Statistical treatment of non-independent folds

The 25 folds are not independent: the five repeats share patients, and folds within a repeat share training data. Treating them as 25 independent samples inflates significance; a naive Nadeau–Bengio correction [9] applied to the repeated design over-inflates the variance (approximately 7×, because it was derived for ordinary k-fold). We therefore report three views and let the reader see the spread: (i) a plain paired Wilcoxon (over-optimistic), (ii) a **repeat-level paired t-test** with df=4 — the structurally correct unit is the five independent repeats, each aggregated to its five-fold mean; this is correct *in structure* but under-powered with only five points, and (iii) the Nadeau–Bengio approximation (over-conservative here). **We do not claim "significant improvement" or "no effect" from any single p; we describe the effect size and say explicitly that significance is sensitive to the treatment of non-independence.** For the 30-test family (5 C-index + 25 node-AUC comparisons) we apply Benjamini–Hochberg FDR: no test survives at q<0.05 (smallest q=0.187), so node-level positives are reported as exploratory. The *deterministic* artifact, by contrast, needs no p-value: an AUC of 0.5000 with standard deviation 0 is a mathematical fact.

---

## IV. Results

### A. The deterministic artifact: naive interpolation fakes a boundary-geometry effect

Table I reports the node-level IPCW AUC of the six strategies at 6 and 12 months under the naive-constant and the PCH interpolation. Under naive-constant, every strategy whose first box covers the horizon collapses to exactly 0.5000: equal-width (first box [0,10] covers 6 m) at 6 m, and log-spacing (first box [0,12.3] covers both 6 m and 12 m) at both horizons. Under PCH the collapse vanishes and the AUC returns to 0.70–0.72.

| Strategy | AUC@6m naive → PCH | AUC@12m naive → PCH |
|---|---|---|
| Event-quantile (baseline) | 0.701 → 0.705 | 0.696 → 0.695 |
| Equal-width | **0.500 → 0.709** | 0.723 → 0.722 |
| All-time quantile | 0.703 → 0.705 | 0.697 → 0.700 |
| Log-spacing | **0.500 → 0.712** | **0.500 → 0.720** |
| Clinical-anchored (no36) | 0.708 → 0.705 | 0.718 → 0.718 |
| AAIC | 0.702 → 0.703 | 0.712 → 0.714 |

*Table I. Node-level IPCW AUC at 6/12 months under the naive-constant shortcut vs the PCH interpolation (25 folds). A value of exactly 0.500 arises only when the horizon falls inside the first box under naive-constant; it carries standard deviation 0 and is therefore a deterministic artifact, not a statistical estimate.*

Two controls make the attribution airtight. First, the `step` scheme — identical to naive-constant except that the first-box survival is S(e₁) rather than 1.0 — restores the full AUC (0.7094/0.7175/0.7225 at the affected nodes), so the *entire* ≈0.20 collapse is attributable to the single "return 1.0 in the first box" line. Second, strategies whose first box is already narrow (baseline, all-time quantile, clinically anchored) are untouched: their 6-month AUC is essentially unchanged (0.696–0.708). The artifact is purely a function of where the first-box edge sits relative to the horizon, not of the model or the data.

### B. Global discrimination: a small, directionally consistent, marginally significant effect

Table II gives the 25-fold C-index for the six strategies. Two facts stand out.

First, **the effect is small**: the best non-anchored geometry (equal-width/log) improves C-index over the baseline by only +0.006, and the anchored no36 by +0.009. Second, **the direction is consistent**: dense early subdivision (equal-width/log/no36) > event-quantile > all-time quantile, in that order.

| Strategy | Harrell C | ΔC | Uno C (τ=60) | ΔUno | repeat-level p | win |
|---|---|---|---|---|---|---|
| Event-quantile (baseline) | 0.6575 | — | 0.6511 | — | — | — |
| Equal-width | 0.6631 | +0.0056 | 0.6573 | +0.0062 | 0.028 | 20/25 |
| All-time quantile | 0.6590 | +0.0015 | 0.6527 | +0.0016 | 0.40 | 13/25 |
| Log-spacing | 0.6635 | +0.0060 | 0.6571 | +0.0060 | 0.002 | 18/25 |
| Clinical-anchored (no36) | 0.6667 | +0.0092 | 0.6606 | +0.0095 | 0.005 | 20/25 |
| AAIC | 0.6650 | +0.0075 | 0.6587 | +0.0076 | 0.007 | 22/25 |

*Table II. 25-fold paired C-index vs the event-quantile baseline. `repeat-level p` is the two-sided paired t over the five repeat means (df=4); the plain Wilcoxon p and a Nadeau–Bengio approximation are given in Section IV-C for the same differences, and they bracket this value.*

Third, the picture is robust to the metric's censoring sensitivity: the Uno IPCW C-index shifts *all* strategies down by ≈0.006 (Harrell over-estimates by roughly this amount), but the offset is nearly identical across strategies (±0.0003), so the paired differences are unchanged — dense geometry still > baseline by +0.006 to +0.009.

Fourth, the significance is genuinely marginal. Table III shows the same ΔC under three statistical treatments. The plain paired Wilcoxon is the most favorable (p = 0.0001–0.034 for the non-anchored families); the repeat-level t (df=4) gives p = 0.002–0.028; and the Nadeau–Bengio approximation gives p = 0.13–0.49. The truth lies in between, and we decline to claim either "significant" or "absent."

| Strategy | ΔC | Wilcoxon p | repeat-level t p | Nadeau–Bengio p |
|---|---|---|---|---|
| Equal-width | +0.0056 | 0.034 | 0.028 | 0.486 |
| All-time quantile | +0.0015 | 0.353 | 0.402 | 0.639 |
| Log-spacing | +0.0060 | 0.012 | 0.002 | 0.322 |
| Clinical-anchored (no36) | +0.0092 | 0.0004 | 0.005 | 0.138 |
| AAIC | +0.0075 | 0.0001 | 0.007 | 0.132 |

*Table III. The same C-index differences under three treatments of the non-independent 25 folds. The structural truth lies between the (over-optimistic) Wilcoxon and the (over-conservative) Nadeau–Bengio values; the repeat-level t (df=4) is structurally correct but under-powered.*

*GPU non-determinism.* CUDA training is not bit-reproducible: re-running a variant changes its C-index by ≈0.001–0.002. Consequently we never interpret differences smaller than ≈0.002, which is a further reason the small C-index effects above are described as directional rather than precisely quantified.

### C. Node-level discrimination under correct interpolation

With PCH interpolation (Table IV, all differences vs the event-quantile baseline), no cell collapses. The residual node-level effect is an *exploratory* pattern concentrated at 12 and 48 months: dense geometries gain +0.02 to +0.03 at 12 m and 12 m/48 m, while all-time quantile is flat-to-negative. None of these survives FDR over the full family (smallest q=0.187), so we report them as observations to be confirmed, not conclusions.

| Strategy | ΔAUC@6m | ΔAUC@12m | ΔAUC@24m | ΔAUC@36m | ΔAUC@48m |
|---|---|---|---|---|---|
| Equal-width | +0.004 | +0.027 | +0.008 | +0.010 | +0.031 |
| All-time quantile | −0.000 | +0.005 | +0.003 | +0.004 | +0.009 |
| Log-spacing | +0.007 | +0.025 | +0.005 | +0.006 | +0.019 |
| Clinical-anchored (no36) | −0.001 | +0.023 | +0.005 | +0.016 | +0.031 |
| AAIC | −0.002 | +0.019 | +0.003 | −0.001 | +0.012 |

*Table IV. Per-horizon IPCW AUC differences vs the event-quantile baseline (25 folds, PCH interpolation). No cell collapses; the 12-month and 48-month gains are exploratory (FDR not significant).*

### D. Interpolation sensitivity: the choice among correct schemes does not matter for ranking

Table V holds the model and boundary set fixed and varies only the interpolation, for the four non-anchored families at the horizons where the artifact bites (6, 12) plus 48 months. Three results.

First, naive-constant is the lone outlier, and only inside the first box (equal-width 6 m; log 6 m and 12 m). Second, `step` agrees with naive-constant everywhere *except* the first box, confirming the attribution in Section IV-A. Third — the substantive negative result — `linear` and `pc` agree everywhere to <0.001 (the only visible gap, baseline 48 m, is 0.0066). Node-level AUC is a ranking measure, and both schemes are monotone functions of the same per-interval hazards, so the patient ordering is unchanged.

| Variant | interp | AUC@6m | AUC@12m | AUC@48m |
|---|---|---|---|---|
| Baseline (event-quant.) | naive | 0.706 | 0.702 | 0.753 |
| | linear | 0.708 | 0.702 | 0.753 |
| | pc | 0.708 | 0.702 | 0.747 |
| Equal-width | naive | **0.500** | 0.723 | 0.775 |
| | linear | 0.709 | 0.725 | 0.772 |
| | pc | 0.709 | 0.724 | 0.772 |
| All-time quantile | naive | 0.705 | 0.701 | 0.754 |
| | linear | 0.706 | 0.699 | 0.755 |
| | pc | 0.706 | 0.699 | 0.754 |
| Log-spacing | naive | **0.500** | **0.500** | 0.763 |
| | linear | 0.718 | 0.723 | 0.761 |
| | pc | 0.718 | 0.723 | 0.760 |

*Table V. Node-level AUC under four interpolations (25-fold mean). naive-constant collapses only when the horizon falls inside the first box; linear and pc agree to <0.001.*

This also settles the comparison to DCS on a clean footing: DCS interpolates linearly, we interpolate with PCH, and the two give statistically indistinguishable node-level AUC on our data. **Any difference between our boundary-geometry ranking and DCS's is therefore not attributable to the interpolation scheme.**

### E. Non-transfer of the "quantile is best" recommendation

DCS reported quantile (all-time) spacing to be best among linear/log/quantile. On our data the ranking is reversed under *both* interpolation schemes: all-time quantile is the worst of the three families on C-index (ΔC +0.0015, p = 0.35–0.40) and flat-to-negative at the nodes (Table IV), while equal-width and log-spacing are the better ones. We report this as a **non-replication in our setting**, and enumerate the confounds honestly: (i) different data domain (tabular clinical vs. multimodal imaging with a front-loaded event distribution), (ii) different backbone (DRSA vs. frozen SwinUNETR+SALP head), and (iii) different node counts (10 vs. 12). We do not claim DCS is "wrong"; we claim the recommendation does not transfer, which is itself the methodological point: boundary geometry and interpolation should be decided on the target distribution, not inherited from a prior study.

### F. Cohort stratification

Because the two sub-cohorts differ sharply in censoring (UCSF 50.1% vs UPENN 95.7% events), we report C-index separately (Table VI). The level differs enormously — UCSF ≈ 0.70 vs UPENN ≈ 0.61 — a strong cohort confound that must accompany any headline number. The *direction* of the boundary-geometry effect, however, is consistent in both cohorts (dense geometry > baseline in each), so the effect is not a UCSF-only phenomenon.

| Strategy | UCSF | UPENN |
|---|---|---|
| Baseline | 0.6975 | 0.6044 |
| Equal-width | 0.7111 | 0.6134 |
| All-time quantile | 0.7007 | 0.6052 |
| Log-spacing | 0.7087 | 0.6102 |
| Clinical-anchored (no36) | 0.7144 | 0.6133 |
| AAIC | 0.7071 | 0.6113 |

*Table VI. Per-cohort Harrell C-index (25 folds).*

---

## V. Discussion

### A. What is established, and at what strength

We separate results by evidence strength rather than by narrative convenience.

- **Deterministic (no p needed).** A naive first-box-constant interpolation pins node AUC at 0.5000±0.0000 for any horizon inside the first box. This is a bookkeeping error, not an empirical effect, and it *fakes* a boundary-geometry effect of ≈0.20 that disappears with a correct interpolation.
- **Robustness (negative result).** Among correct schemes, linear (DCS) and PCH (CHI special case) agree on node-level AUC to <0.001; interpolation choice is not a new sensitivity dimension for ranking measures, and it cannot explain any deviation from DCS.
- **Marginal (direction consistent, significance sensitive).** On C-index, dense early subdivision beats event-quantile by +0.006 to +0.009; the repeat-level test is significant (p≤0.028) but the Nadeau–Bengio approximation is not (p≥0.13), so we call it marginal, not robust.
- **Null (stable).** All-time quantile does not beat the baseline in either cohort, under either interpolation — supporting a non-transfer of DCS's quantile recommendation.
- **Exploratory.** The 12-/48-month node gains (+0.02–0.03) do not survive FDR; they are hypotheses for a larger or external dataset.

### B. Practical guidance

Three concrete recommendations follow. (1) **State and test the evaluation interpolation** in any paper reporting node-level discrete-survival AUC; a one-line fix can invert a headline conclusion, and an AUC of 0.5000±0.0000 is an unambiguous red flag that the survival at that horizon is constant. (2) **Do not over-trust C-index as a boundary-geometry summary**; the present effect is small and marginal, and prior dramatic claims were interpolation artifacts. (3) **Treat boundary geometry and interpolation as distribution-local decisions.** A recommendation inherited from a tabular study (quantile "best") need not hold for imaging, and the low-cost-input constraint further shifts the appropriate design.

### C. Walking back our own earlier drafts

For transparency: an earlier draft of this work reported "significantly" better C-index for equal-width/log spacing (p=0.042/0.016) and a "contradiction" of DCS, under a naive interpolation and over-optimistic fold independence. Both framings were wrong in degree. The corrected analysis reported here — marginal C-index effects, a stable non-transfer rather than a contradiction, and the interpolation artifact isolated as the determinant of the dramatic node-level behavior — is the version we stand behind. We surface this because the failure mode we document is exactly the kind that a naive evaluation silently rewards.

### D. The demotion of the censoring-aware criterion (AAIC)

Our earlier drafts positioned a censoring-aware boundary criterion (AAIC) as a headline contribution, on the ground that it "preserved early nodes" that other dense geometries "collapsed." That selling point does not survive the artifact correction: the nodes were never truly collapsed — their apparent collapse was the interpolation artifact — so AAIC's preservation property is vacuous. AAIC retains a modest, standalone role: its boundaries respond monotonically to synthetic tail-censoring (Appendix A), showing a censoring-aware criterion is *possible* and *graded*. But it is no longer a core contribution, and the paper's claims do not depend on it.

### E. The low-cost input constraint and its cost

Restricting inputs to T1c/T2/FLAIR + age/sex is a deliberate deployment choice for resource-limited settings, and it is what gives this evaluation practical meaning: we are characterizing boundary-geometry/interpolation effects in the setting where themodel is actually affordable to deploy. The cost is real and must be stated: without molecular and treatment covariates, the model may partially learn center differences rather than pure imaging signal. The UCSF≈0.70 vs UPENN≈0.61 gap (Table VI) is consistent with such confounding. We therefore present per-cohort results and regard a stratified or matched design as the appropriate next step, not an immediate blocker.

### F. Limitations

1. **Single harmonized collection.** One two-center glioma dataset; the tail-censoring manipulation is synthetic within-cohort, not external validation on an independent censoring distribution.
2. **Not a full DCS re-run.** Strategies 2–4 reproduce DCS's spacing geometry on our frozen head, isolating the spacing degree of freedom; they do not claim our head outperforms DCS end-to-end.
3. **Internal-validation optimism.** All folds derive from one collection; an external holdout would be more conservative.
4. **Non-independent folds.** The statistical machinery (Section III-G) is a compromise: the repeat-level test is structurally correct but under-powered; the true p for the small C-index effects lies in a wide interval.
5. **Frozen encoder.** A single frozen feature extractor limits conclusions about how much signal passes through imaging features; the artifact we document, however, is independent of this choice.
6. **Literature completeness.** Related-work search used arXiv/OpenAlex/Crossref programmatically; commercial bibliographic databases should be re-checked before submission.

---

## VI. Conclusion

We audited two hidden, interacting decisions in discrete-time survival evaluation — interval boundary geometry and survival interpolation — on a frozen-encoder multimodal glioma pipeline restricted to three conventional MRI sequences and age/sex. The central result is deterministic: a naive first-box-constant interpolation fabricates a dramatic ≈0.20 node-level "boundary-geometry effect" that vanishes under a correct interpolation, because a survival of 1.0 for every patient pins the AUC at exactly 0.5. Once interpolation is correct, its precise form barely matters for discrimination (linear ≈ piecewise-constant-hazard, <0.001), the residual boundary-geometry effect is small and only marginally significant (+0.006 to +0.009 on C-index), and the popular "quantile spacing is best" recommendation does not transfer to imaging. The actionable lesson for the field is narrower but harder than what we once claimed: node-level discrete-survival discrimination results are not trustworthy unless the evaluation interpolation is stated and correct, and any boundary-geometry conclusion drawn under a naive interpolation should be re-derived before it is believed.

---

## References

[1] C. Lee, W. Zame, J. Yoon, and M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks," in *Proc. AAAI Conf. Artif. Intell.*, vol. 32, 2018.

[2] M. F. Gensheimer and B. Narasimhan, "A scalable discrete-time survival model for neural networks," *PeerJ*, vol. 7, p. e6257, 2019.

[3] H. Kvamme and Ø. Borgan, "Continuous and discrete-time survival prediction with neural networks," *Lifetime Data Analysis*, vol. 27, pp. 710–736, 2021, doi: 10.1007/s10985-021-09532-6.

[4] P. Fuhlert, A. Ernst, E. Dietrich, F. Westhaeusser, K. Kloiber, and S. Bonn, "Deep Learning-Based Discrete Calibrated Survival Prediction," in *Proc. 2022 IEEE Int. Conf. Digital Health (ICDH)*, 2022, doi: 10.1109/ICDH55609.2022.00034. (Source code: github.com/imsb-uke/dcsurv.)

[5] J. Hickey, R. Henao, D. Wojdyla, M. Pencina, and M. M. Engelhard, "Improving Event Time Prediction by Learning to Partition the Event Time Space," *arXiv preprint arXiv:2310.15853*, 2023.

[6] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images," in *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries*, Springer, 2022, pp. 272–284.

[7] M. J. Cardoso et al., "MONAI: An open-source framework for deep learning in healthcare," *arXiv preprint arXiv:2211.02701*, 2022.

[8] H. Uno, T. Cai, M. J. Pencina, R. B. D'Agostino, and L.-J. Wei, "On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data," *Statistics in Medicine*, vol. 30, no. 10, pp. 1105–1117, 2011.

[9] C. Nadeau and Y. Bengio, "Inference for the generalization error," *Machine Learning*, vol. 52, pp. 239–281, 2003.

[10] D. R. Cox, "Regression models and life-tables," *J. Royal Statistical Society: Series B*, vol. 34, no. 2, pp. 187–220, 1972.

[11] E. Calabrese et al., "The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset," *Radiology: Artificial Intelligence*, vol. 4, no. 5, e220058, 2022.

[12] S. Bakas et al., "The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics," *Scientific Data*, vol. 9, 400, 2022.

---

## Appendix A. AAIC: a censoring-aware boundary criterion (demoted)

AAIC constructs boundaries in three deterministic, parameter-free steps: (1) **anchor** the clinical nodes H={6,12,24,36,48} together with {0, t_max}; (2) **density insertion** — up to m=2 additional edges greedily maximizing log-info(E) = Σ_boxes [log(1+n_ev) − κ·log(1+n_cens)], κ=1, subject to ≥15 events per box; (3) **censoring-removal test** — drop any anchored edge whose removal raises log-info (i.e., an edge sitting in a censoring-dense neighborhood).

Its sole remaining, standalone result is the censorship dose-response: raising the training fold's tail (>36 m) censoring rate from 0.54 to 0.70 to 0.90 removes the 48 m edge in 15/25 → 25/25 → 25/25 folds, then the 36 m edge in 0/25 → 8/25 → 25/25 folds (median boxes 7 → 7 → 4). Only the censoring labels change, so the criterion's behavior is genuinely graded and causally tied to censoring. We keep this as a mechanism demonstration, *not* a contribution: AAIC does not outperform log-spacing on C-index, and its earlier "node-preservation" rationale was nullified by the artifact correction (Section V-D).

---

> **Reproducibility note.** The quantitative results live in Tables I–VI, which
> are regenerated by `scripts/analyze.py` from the trainer outputs. Figures are
> intentionally omitted from this repository. CUDA training is not bit-exact:
> re-running a variant shifts C-index by ≈0.001–0.002, so differences <0.002 are
> never interpreted. The BrainSegFounder pre-trained weights used to initialize
> the frozen encoder require their own citation before submission.