# vae_lesion_deficit
Lesion-deficit mapping/inference with VAEs

Use simulation_viz.ipynb to get a good grasp of how to use this codebase

## Covariate-aware substrate recovery

`covariate_substrate.ipynb` — recovering the lesion–deficit substrate when a **covariate**
(age) both modulates the deficit and is *confounded* with lesion volume, comparing each
method **covariate-blind vs covariate-aware** (VLSM, SVR-LSM, and the two-arm VAE with a
covariate deficit head). Runs in Glasser-360 parcel space on a shipped 512-lesion sample.

The baseline/model logic lives in importable modules (the notebook just imports them):

| module | contents |
|---|---|
| `covariate_deficits.py` | covariate-conditioned deficit generator (age main effect + interaction, age↔volume confound `rho`, inert sex; rules: coverage / bool_OR / hub_spoke / bool_AND) |
| `baselines.py` | VLSM & SVR-LSM, each covariate-**blind** and **aware** (`cov=`); `dice`, `dice_at_k`, `focality`, `age_exposure`, `leakage`, thresholding |
| `cov_vae.py` | `SubstrateVAE` two-arm VAE + covariate deficit head (covariates enter the head only, not the encoder); `fit_substrate_vae(X, y, cov=...)` |
| `data_utils.py` | load the shipped `new_data/` sample; paint parcel maps back to a volume |

Data in `new_data/` (self-contained; no external downloads): `episode_overlaps.npy`
`(6, 512, 360)`, `substrates.npy` `(6, 360)`, `parcel_centroids.npy`, `substrate_names.txt`,
and `glasser_360_2mm_labels.npy` (label volume for visualisation).

Install: `pip install -r requirements-covariate.txt`.
