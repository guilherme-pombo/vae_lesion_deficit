# vae_lesion_deficit
Lesion-deficit mapping/inference with VAEs

Use simulation_viz.ipynb to get a good grasp of how to use this codebase

## Covariate-aware substrate recovery in voxel space

**`covariate_voxel.ipynb`** — recovering the lesion–deficit substrate in **4 mm MNI voxel space**,
comparing four methods **covariate-blind vs covariate-aware** on 512 real KCH stroke lesions:

- **VLSM** (mass-univariate Welch-t + permutation FDR),
- **SVR-LSM** (multivariate RBF SVR + back-projection),
- **MLP-VAE** (the two-arm VAE, membership × importance), and
- **Conv-VAE** — the same two-arm VAE with a **3D-convolutional** encoder/decoder (a spatial prior).

Headline: the **conv-VAE recovers the substrate best** (the 3D spatial prior beats the flat MLP and
the classical methods), and **covariate-awareness helps it** while leaking the least confound bias —
whereas the linear methods over-correct. Covariate control follows the standard **partialing-out**
(residualize the behaviour on the covariates) rather than appending covariates as features.

The code is organised under `src/` — classical **baselines** and the **dlm** (deep-lesion-mapping)
VAEs in their own subpackages; the notebook does `sys.path.insert(0, "src")` and imports them:

```
src/
  voxel_data.py    VoxelData.load(...) — the shipped 4 mm sample (X, substrate masks,
                   centroids, coverage mask_idx, affine); to_volume for rendering
  deficits.py      covariate-conditioned deficit generator (age main + interaction,
                   age↔volume confound rho, inert sex; rules coverage/bool_OR/hub_spoke/bool_AND)
  metrics.py       dice, dice_at_k, focal_mass_mask, age_exposure, leakage
  baselines/       VLSM (vlsm.py) & SVR-LSM (svr_lsm.py) — each covariate-blind/aware
                   (cov=, cov_mode="partial"); shared _residualize/_fdr_bh in _common.py
  dlm/             the two-arm VAEs — mlp_vae.py (fit_parcel_vae) and
                   conv_vae.py (fit_conv_vae, 3D-conv; GPU), each with a covariate head
```

Data (self-contained; no downloads): `voxel_data_sample/voxel_ds.npz` (V_cov=2541 voxels × 512
patients + 6 substrates), `voxel_results.json` (precomputed GPU means + per-substrate), `imgs/`
(glass-brain figures). The fast methods (VLSM / SVR-LSM / MLP-VAE) run live in the notebook; the
conv-VAE is GPU-trained, so its numbers are precomputed (a code snippet reproduces them on a GPU).

Install: `pip install -r requirements-covariate.txt`.

> `covariate_substrate.ipynb` is an earlier **parcel-space** prototype of this tutorial; its shared
> modules were refactored for the voxel version above, so it is superseded and won't run as-is.
