# aind-ophys-decrosstalk-roi-images

Estimate and apply **decrosstalk** for multiplane-ophys (2-photon mesoscope) data: for each
simultaneously-imaged plane pair, estimate the crosstalk coefficients (α, β) and reconstruct
the crosstalk-corrected movie.

> Branch **`speed-up`**: a faster, dependency-lighter version of the estimator (classical
> segmentation instead of CellPose; index-cached, coarse-to-fine grid search),
> reciprocity-averaged coefficients, and richer QC outputs (per-plane landscape figure,
> landscape-quality + background-correlation metrics). See **Speed-up changes** below.

## The decrosstalk model (linear unmixing)

The mesoscope images two planes **A** and **B** at the same XY position simultaneously; each
detector also collects a fraction of the *other* plane's fluorescence (temporal bleed-through
= crosstalk). The measured ("imaged") frames are therefore a **linear mixture** of the true
per-plane signals:

```
imaged_A = (1 - α) · real_A  +      β · real_B
imaged_B =      α · real_A  + (1 - β) · real_B
```

or, in matrix form with the mixing matrix **M**:

```
[ imaged_A ]   [ 1-α    β  ] [ real_A ]
[ imaged_B ] = [  α    1-β ] [ real_B ]
                   =  M
```

- **α** — the fraction of plane **A**'s own signal that leaks **OUT** into plane B's detector
  (a *loss* from A; it appears in `imaged_A` only as the gain `1-α` on `real_A`).
- **β** — the fraction of plane **B**'s signal that leaks **IN** and *contaminates* plane A.

### Reconstruction = inverting the mixture

The corrected signals are recovered by applying `M⁻¹` (det `M = 1-α-β`):

```
[ recon_A ]        1      [ 1-β    -β ] [ imaged_A ]
[ recon_B ] = --------- · [  -α   1-α ] [ imaged_B ]
              (1 - α - β)
```

so, for plane A,

```
recon_A = [ (1-β) · imaged_A  -  β · imaged_B ] / (1 - α - β)
```

**In words:** to clean plane A, subtract `β · imaged_B` (the partner plane scaled by β) from
`(1-β) · imaged_A`, then renormalize by the common gain `1/(1-α-β)`. Consequences worth
noting:

- **β is plane A's contamination-removal knob** — it sets how much of the partner is
  subtracted, i.e. whether A is under- or over-corrected.
- The factor `1/(1-α-β)` is a **common gain** on both `recon_A` and `recon_B`. It rescales
  brightness but not relative structure; mutual information (the estimation objective, below)
  is invariant to it. So **α does not affect plane A's cleaned image** — α is the coefficient
  that cleans the *partner* (`recon_B ∝ (1-α)·imaged_B − α·imaged_A`).
- **Reciprocity:** the physical leak `A→B` is estimated as `α` in A's run and as `β` in B's
  run; the leak `B→A` as `β` in A's run and `α` in B's run. Hence `α_A ≈ β_B` and
  `β_A ≈ α_B` (a data-integrity check).

### Estimating (α, β)

The coefficients are chosen to make the reconstructed planes maximally **independent**
(different neurons ⇒ their signals should share no information): minimize the **normalized
mutual information** between `recon_A` and `recon_B`, evaluated inside ROI bounding boxes on
the per-epoch mean-FOV images, grid-searched over (α, β) and averaged across epochs.

```
(α*, β*) = argmin_(α,β)  mean_over_ROI_boxes  NMI( recon_A(α,β),  recon_B(α,β) )
```

## Speed-up changes (this branch)

Baseline (per epoch, deep 456×473 crop): CellPose ROI segmentation ~22 s, **MI grid search
~80 s** (the bottleneck), classical segmentation ~0.3 s.

1. **Segmenter swap** — CellPose → classical `basic_segmentation` (Gaussian high-pass → Otsu
   → connected components). ~22 s → ~0.3 s/epoch; drops the torch/CellPose dependency;
   reproduces stored β within ~0.02 on 84 planes.
2. **Index-caching** — precompute each ROI box's pixel indices once instead of `np.where`
   per (α,β)×box. **Exact** (bit-identical argmin).
3. **Coarse-to-fine grid** — 0.04 grid over 0–0.36 to locate the basin, then 0.01 grid ±0.05
   around it (~5× fewer evaluations); reproduces the full-grid argmin within one grid step.
   Max raised 0.30 → 0.36 to avoid clipping heavy-crosstalk planes.
4. **Reciprocity-averaged coefficients** — average the two estimates of each physical leak
   (`α_A` with `β_B`) and apply them flipped to the pair (one leak → one coefficient).
5. **QC outputs** — applied α/β saved as h5 attrs + data-process JSON; a per-plane landscape
   QC figure; landscape-quality (curvature / flatness / SNR) and paired-plane
   background-correlation metrics.

Net: per-plane estimation drops from ~17 min toward ~1 min, with no change to the estimated
coefficients.

## Layout

```
code/
  decrosstalk_roi_image.py       mixing/unmixing, segmentation, grid search, QC figure + metrics
  paired_plane_registration.py   register each plane to its partner; episodic-mean-FOV
  run_capsule.py                 pipeline entry (estimate → reciprocity-average → apply → QC)
environment/                     Code Ocean environment
```
