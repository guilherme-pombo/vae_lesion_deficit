# Spatially-coherent substrate estimation

`vld.priors.spatial` adds a Laplacian-penalised (graph-Tikhonov) substrate estimator. This note explains
why it exists, what it buys, and — importantly — when it does *not* help.

Reproduce every number here with:

```bash
python -m vld.scripts.spatial_prior_demo --seeds 3
```

## The problem

Lesion-deficit mapping is an ill-posed linear inverse problem. With N patients and K ≫ N voxels the data
cannot pin the substrate on their own, so whatever penalty you choose decides the answer you get.

Ordinary ridge penalises `‖w‖²` — "prefer small weights". That says nothing about space: voxel *i* and its
immediate neighbour are treated as unrelated free parameters. For a brain that is plainly the wrong prior,
and it is why ridge maps come out speckled.

Penalising *roughness* instead:

```
min_w  ‖y − Xw‖²  +  λ · wᵀ(I + βL)w
```

where `L` is the graph Laplacian over the voxel adjacency graph. `wᵀLw` is the sum of squared differences
between adjacent voxels, so minimising it prefers spatially coherent maps. `β = 0` recovers ridge exactly
(there is a test for that).

This is the penalty family behind **LORETA** (Pascual-Marqui et al., 1994), used for EEG/MEG source
localisation, which is underdetermined in the same way — far more candidate sources than sensors. We use
the name only as precedent; the estimator here is plain Laplacian-penalised regression.

## The finding that motivates it

**Ridge is the best of these estimators at held-out prediction and the worst at recovering the substrate.**
On the shipped sample cohort, planting each known substrate and scoring the recovered map against it:

| substrate | noise | estimator | support-AUC | Dice@k | Moran's I | OOF ρ |
|---|---|---|---|---|---|---|
| gaze | 0.6 | **ridge (β=0)** | 0.799 | 0.307 | 0.355 | **0.206** |
| gaze | 0.6 | Laplacian β=10 | **0.933** | **0.520** | 0.523 | 0.185 |
| gaze | 1.2 | **ridge (β=0)** | 0.721 | 0.176 | 0.308 | **0.175** |
| gaze | 1.2 | Laplacian β=30 | **0.886** | 0.367 | 0.394 | 0.155 |
| language | 0.6 | **ridge (β=0)** | 0.636 | 0.260 | 0.469 | **0.184** |
| language | 0.6 | Laplacian β=300 | **0.890** | 0.173 | 0.346 | 0.150 |
| language | 1.2 | **ridge (β=0)** | 0.607 | 0.150 | 0.355 | **0.155** |
| language | 1.2 | Laplacian β=300 | **0.881** | 0.171 | 0.250 | 0.130 |
| dorsal_premotor | 0.6 | **ridge (β=0)** | 0.818 | 0.412 | 0.378 | **0.184** |
| dorsal_premotor | 0.6 | Laplacian β=30 | **0.910** | **0.475** | 0.432 | 0.160 |
| dorsal_premotor | 1.2 | **ridge (β=0)** | 0.763 | 0.304 | 0.319 | **0.154** |
| dorsal_premotor | 1.2 | Laplacian β=30 | **0.877** | **0.425** | 0.401 | 0.137 |

In all six conditions ridge has the highest out-of-fold ρ and the lowest support-AUC, and **every** β in
the swept range beats ridge on recovery. Gains run from +0.09 to +0.25 support-AUC.

The practical consequence: **do not select substrate hyperparameters on held-out prediction.** It is
measuring a different objective, and here it is anti-correlated with the one you want.

Be careful how far you generalise that, though. Across the full estimator ladder the prediction-vs-recovery
relationship is *not* a smooth monotone trade-off — it is a ridge-specific effect plus the general fact
that prediction is an unreliable selector. Pooling conditions of different difficulty even produces a
*positive* correlation (Simpson's paradox), so quote the within-condition comparison, not a pooled one.

## Choosing β

There is no universal value. The optimum depends on voxel size, grid extent and how distributed the true
substrate is — on the sample cohort above it ranges from 10 (`gaze`) to 300 (`language`). What is robust is
that a broad range beats ridge, so the cost of a bad-but-reasonable β is small and the cost of β=0 is large.

Since prediction is disqualified as a selector, use one of:

- a **planted-substrate benchmark**, when you can plant one in your own lesion matrix (the demo script is a
  template);
- **split-half stability** of the recovered map, when you cannot — but see the warning below.

## Two traps

**Smoothness inflates stability mechanically.** A smoother map has fewer effective degrees of freedom, so
split-half agreement rises whether or not the substrate is better identified. Always report `morans_i`
alongside a stability number, and compare at *matched* smoothness — blur the unsmoothed maps until their
Moran's I matches, then compare. If the gap survives, the prior is doing real work.

**The prior does not add information.** Smoothing moves no direction from the null space of `X` into its
row space; it only picks a different member of the same equivalence class. It makes better use of what the
cohort constrains — it does not enlarge it. In particular a smoothing prior can make a poorly-constrained
voxel *look* confident by borrowing from its neighbours, so any significance testing must push the null
through the identical estimator and prior (permutation or knockoff calibration), never a per-voxel
parametric test.

## When it does not help

- **When the problem is not ill-posed.** With N comparable to K, ridge is already near-perfect and
  smoothing only costs (there is a test pinning this boundary).
- **On genuinely scattered substrates.** If the truth has no spatial structure, nothing recovers it at
  realistic N — though the prior does not make matters worse.
- **Post hoc.** Smoothing an already-fitted map is not a substitute; the penalty has to be in the
  objective. Applied afterwards it consistently degraded recovery in our tests.
- **Anisotropic variants.** Weighting graph edges by tract- or lesion-profile similarity was worse than
  plain isotropic smoothing in every condition we tried, including on a tract-shaped ground truth.

## API

```python
from vld.priors import LaplacianInverse, fit_laplacian_substrate, voxel_adjacency
from vld.metrics import support_auc, morans_i

# one-shot
w = fit_laplacian_substrate(X, y, mask_idx=mask_idx, vol_shape=shape, beta=100.0, cov=cov)

# sweeping β, folds or bootstraps: build once, fit many times
solver = LaplacianInverse(X, mask_idx=mask_idx, vol_shape=shape, beta=100.0, cov=cov)
full = solver.fit(y)
half = solver.fit(y[idx], rows=idx)     # reuses the factorisation
```

`LaplacianInverse` precomputes `M⁻¹Xᵀ` once. Column *i* of that matrix is `M⁻¹x_i`, so any patient subset
is a column slice — cross-validation folds, split halves and bootstrap resamples all reuse one sparse
factorisation instead of refactorising per fit. That is what makes a β × fold × seed sweep tractable.
