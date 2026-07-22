# Covariate lesion-deficit tutorial — short spec

**Goal.** Add a tutorial showing lesion–deficit **substrate recovery under a covariate**
(age) that both modulates the deficit and confounds it (age correlates with lesion volume),
comparing each method **covariate-blind vs covariate-aware**. Baseline logic lives in
importable Python modules; the notebook just imports.

**Resolution / data (`new_data/`).** Our internal resolution: **Glasser-360 parcel space
(2 mm)**, using the KCH lesion sample we benchmark with. Shipped self-contained:
- `episode_overlaps.npy` `(S=6, N=512, K=360)` — parcel-overlap matrices (one 512-lesion
  sample per substrate).
- `substrates.npy` `(6, 360)` — binary substrates (6 NeuroQuery maps, 10 parcels each).
- `parcel_centroids.npy` `(360, 3)` — for the spatially-split deficit rules.
- `substrate_names.txt`; `glasser_360_2mm_labels.npy` `(91,109,91)` label volume for viz.

**Modules** (standalone, notebook imports):
- `covariate_deficits.py` — cov-v1 DGP: `build_covariates` (age = volume-confound via `rho`
  + inert sex), `make_covariate_deficits` (severity `g·clip(1+β_int·age,0)+β_main·age`,
  top-k), rules {coverage, bool_OR, hub_spoke, bool_AND}.
- `baselines.py` — `vlsm_stat`/`vlsm_binary` (Welch-t; `cov=` residualizes),
  `svr_lsm_weights`/`svr_lsm_binary` (`cov=` nuisance features, back-project parcel cols),
  `dice`, `dice_at_k` (primary metric), `focality`, `age_exposure`, `leakage`,
  `focal_mass_mask`. Each method: blind (`cov=None`) + aware (`cov=C`).
- `cov_vae.py` — `SubstrateVAE` two-arm VAE + covariate head
  `logit = scale·mean(X·p·w)·(1+Cγ)+Cβ+bias`; covariates to the head only, never the
  encoder (`E[p·w]` readout stays covariate-free). `fit_substrate_vae(X, y, cov=None, ...)`.
- `data_utils.py` — `load_episodes`, `load_atlas`, `to_volume` (parcel map → 32-ish volume).

**Tutorial (`covariate_substrate.ipynb`).** Markdown-narrated, imports the modules:
data → covariate deficits (show age↔volume confound, main + interaction) → blind vs aware
recovery (VLSM/SVR/VAE) for one substrate → glass-brain viz (GT vs blind vs aware) → full
6-substrate × 4-rule table (**dice@k** primary; binary Dice + focality optional/slower) +
**leakage** → **ρ sweep** (0/0.5/0.75) showing control helps when age⟂lesion and
over-corrects as they become collinear → takeaways (VAE robust; leakage always drops).

**Primary metric.** `dice@k` (top-|M| parcels of the score map) — standard, fast (no
permutations), robust to self-thresholding. Binary Dice (permutation FWER/FDR) shown as a
secondary, slower cell.

**Deps.** numpy, torch, scikit-learn, matplotlib. **Non-goals:** no S3/internal deps at
run time; no retraining the repo's conv model.

**Handoff.** Built here; repo owner reviews and pushes (or opens a PR).
