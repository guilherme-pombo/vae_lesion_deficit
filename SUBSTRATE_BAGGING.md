# Substrate bagging (calibrated per-seed → ensemble)

A single VAE substrate fit is high-variance: different weight inits and fit/cal splits give maps that
correlate only ~0.5 across seeds, so one fit can under-perform a stable univariate baseline (VLSM)
despite the VAE's de-biasing. Bagging removes most of that variance — **but only if each bag member is
calibrated before it is ensembled.**

## Procedure (`bagging.fit_conv_vae_bagged`)

For each of `n_bag` seeds (seed drives both the weight init and the fit/cal split):

1. **Fit** `dlm.conv_vae.fit_conv_vae` at that seed. The returned `(V_cov,)` map is already calibrated
   on the fit's own **held-out calibration split** (the §3.2 substrate-rate sweep) and mass-scaled so
   `metrics.focal_mass_mask` self-sizes to the calibrated substrate size.
2. **Binarise** the map with `focal_mass_mask` (its own calibrated size).

Then **ensemble**: average the per-seed binaries into a consensus fraction (share of seeds selecting
each voxel) and deploy the voxels whose agreement meets a chosen level (`agreement=0.5` = majority).

```python
import sys; sys.path.insert(0, "src")
from bagging import fit_conv_vae_bagged

res = fit_conv_vae_bagged(X, y, mask_idx=mask_idx, vol_shape=vol_shape, n_bag=8,
                          cov=cov, continuous=False, epochs=300, readout_coverage_frac=0.1)
res.consensus_mask       # deployed boolean substrate (V_cov,)
res.consensus_fraction   # per-voxel seed agreement in [0, 1]
```

## Why calibrate per seed, not on the consensus

Each seed's map is calibrated to *its own* learned substrate rate on *its own* held-out split, so its
binary map is a valid deployment for that fit. Averaging those calibrated binaries yields an
interpretable agreement fraction.

What this deliberately does **not** do: re-threshold the consensus fraction with `focal_mass_mask` as
if it were a calibrated-rate map. The consensus is an average of binaries, not a calibrated probability
map, so `sum(fraction)` is just the mean per-seed size — using it to re-select voxels is not a
calibration. Calibration must happen per seed, on held-out data, before ensembling.

No ground-truth substrate size is used anywhere, so this is a valid deployment procedure, not an oracle.
