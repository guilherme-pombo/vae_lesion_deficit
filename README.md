# vae_lesion_deficit
Lesion-deficit mapping/inference with VAEs

Use simulation_viz.ipynb to get a good grasp of how to use this codebase

## Setup

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/) (recommended — one command creates the
virtual env and installs everything):

```bash
uv sync --extra notebook     # creates .venv/ with numpy/torch/scikit-learn/matplotlib/nibabel + Jupyter
uv run jupyter lab           # open covariate_voxel.ipynb or aphasia_examples.ipynb
uv run python my_script.py   # …or run any script inside the managed env
```

`uv sync` (without `--extra notebook`) installs just the compute dependencies. Run `uv lock` first if
you want a pinned `uv.lock` for reproducibility. Everything is standard PyPI — no cloud/infra deps.

Plain `pip` works too:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-covariate.txt jupyterlab
```

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
  build_voxel.py   build_voxel_data(...) — turn raw lesion NIfTIs into a VoxelData (nibabel only)
```

Data (self-contained; no downloads): `voxel_data_sample/voxel_ds.npz` (V_cov=2541 voxels × 512
patients + 6 substrates), `voxel_results.json` (precomputed GPU means + per-substrate), `imgs/`
(glass-brain figures). The fast methods (VLSM / SVR-LSM / MLP-VAE) run live in the notebook; the
conv-VAE is GPU-trained, so its numbers are precomputed (a code snippet reproduces them on a GPU).

`aphasia_examples.ipynb` — worked examples for the two common clinical setups: a **binary classifier**
(aphasia yes/no) and a **continuous severity regression**, covariate-aware (age + sex), featuring the
conv-VAE, at 4 mm.

Install: `pip install -r requirements-covariate.txt`.

## Bring your own data — no cloud infra required

Everything trains with **numpy / torch / scikit-learn / nibabel** only — **no plexus, S3, or cloud
dependency**. Go from MNI-registered lesion NIfTIs to a fitted substrate map on any machine (a GPU
just makes the conv-VAE faster; CPU/MPS also work):

```python
import sys; sys.path.insert(0, "src")
from glob import glob
import numpy as np
from build_voxel import build_voxel_data
from dlm.conv_vae import fit_conv_vae       # or dlm.mlp_vae.fit_parcel_vae (CPU-friendly)

# 1. lesions -> coverage-restricted 4 mm voxel matrix (resampled through affines; nibabel only)
data = build_voxel_data(sorted(glob("my_lesions/*.nii.gz")), min_overlap=10)   # 2 mm: resolution_mm=2

# 2. your outcome + covariates
y   = np.load("aphasia_label.npy")          # binary {0,1}  OR  a continuous severity score
cov = np.load("age_sex.npy")                # (N, 2) = [age, sex]; omit for a covariate-blind fit

# 3. recover the substrate (covariate-aware). continuous=True for a severity/regression outcome.
smap = fit_conv_vae(data.X, y, mask_idx=data.mask_idx, vol_shape=data.vol_shape,
                    cov=cov, continuous=False, epochs=300)          # -> (V_cov,) per-voxel map
vol  = data.to_volume(smap)                 # paint back to a (46,55,46) volume for viewing/saving
```

`build_voxel_data` resamples each lesion to a common grid through its affine (L/R and location
preserved for MNI-space inputs), builds the coverage mask (voxels lesioned in ≥ `min_overlap`
patients), and returns a `VoxelData`. Pass `substrate_paths={name: path}` if you have ground-truth
substrate maps to score against; omit for a purely data-driven analysis. **Resolution:** 4 mm is the
sensible default for cohorts in the hundreds (power per voxel, less overfitting); go to 2 mm as a
refinement (`resolution_mm=2`, or `template=<a reference NIfTI>` for a custom grid) once 4 mm looks
good.

## Ready-to-run training scripts (`src/scripts/`)

Standalone torch scripts that go end-to-end — build the voxel matrix, train the covariate-aware
conv-VAE, and write the recovered substrate as a NIfTI. They mirror `aphasia_examples.ipynb` but as
plain `.py` you can drop on a cluster. Both run **out-of-box on the shipped demo sample** (no args
needed) and switch to your own cohort with `--lesions/--labels/--covariates`:

```bash
# 4 mm — binary aphasia (yes/no), then continuous severity; both covariate-aware
python src/scripts/aphasia_4mm.py                        # demo on the shipped 4 mm sample
python src/scripts/aphasia_4mm.py --outcome severity

# 2 mm — same pipeline on the finer grid (~8× more voxels; use a GPU)
python src/scripts/aphasia_2mm.py

# your own cohort: X from lesion NIfTIs, y binary-or-continuous, cov = [age, sex]
python src/scripts/aphasia_4mm.py \
    --lesions 'my_lesions/*.nii.gz' --labels y.npy --covariates age_sex.npy \
    --outcome binary --out my_substrate.nii.gz
```

The two scripts differ only in a resolution config (grid + conv depth/batch); the shared pipeline is
`src/scripts/_aphasia_train.py`. A GPU is used automatically when present (`cuda` → `mps` → `cpu`).

**Running many fits in parallel (SLURM / a GPU cluster).** Each fit is one independent process, so
fan out with an array job — one GPU per task, no inter-task communication:

```bash
#!/bin/bash
#SBATCH --array=0-3            # 4 configs in parallel, one GPU each
#SBATCH --gpus-per-task=1
CONFIGS=(
  "aphasia_4mm.py --outcome binary"
  "aphasia_4mm.py --outcome severity"
  "aphasia_2mm.py --outcome binary"
  "aphasia_2mm.py --outcome severity"
)
srun python src/scripts/${CONFIGS[$SLURM_ARRAY_TASK_ID]} \
     --out substrate_${SLURM_ARRAY_TASK_ID}.nii.gz
```

Sweeps over `--epochs`, `--min-overlap`, or bootstrap resamples parallelise the same way — vary the
args per array index. (Within a single fit the conv-VAE already uses the whole GPU; parallelism buys
you throughput *across* fits, not a faster individual fit.)

> `covariate_substrate.ipynb` is an earlier **parcel-space** prototype of this tutorial; its shared
> modules were refactored for the voxel version above, so it is superseded and won't run as-is.
