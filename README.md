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
whereas the linear methods over-correct.

**Two covariate-control mechanisms** (both legitimate; they differ in *where* the control happens):
the **baselines** (VLSM/SVR-LSM) use classical **partialling-out** — residualize the behaviour on
`[1, cov]` before fitting (`cov_mode="partial"`), the standard LSM confound control. The **VAEs**
control covariates in the **deficit head**: `cov` enters only the head as a main effect (`cov·β`) and
a substrate interaction (`(1 + cov·γ)` scaling the overlap), fitted jointly, while the encoder never
sees `cov` — so the recovered substrate stays covariate-invariant. The head's main-effect term is
asymptotically equivalent to partialling out the main effect (Frisch–Waugh–Lovell), done inside the
model rather than as a pre-step, plus an interaction the pre-residualization can't express.

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

**Relationship to the pre-print.** `src/dlm/conv_vae.py` (`fit_conv_vae`) is the maintained
implementation and the one to build on — it keeps the pre-print's CoordConv / label-as-volume /
joint-`P(X,Y)` conv architecture but adds the fixes this repo depends on: the identifiable two-arm
`σ(γ)·w` (membership × importance) readout, the covariate deficit head, GroupNorm (batch-independent
for the small per-episode batches), a continuous/severity head, coverage-restriction, a **mask-size-
invariant loss** (`loss_norm="voxel_mean"`, default — per-voxel-mean reconstruction so one
`mask_weight` transfers across mask sizes/resolutions), a **readout coverage floor**, and **held-out
validation epoch selection** (`select_on="val"`, default). These matter most at small N (in-sample
selection gives seed-unstable maps; a sum-reduced loss lets reconstruction swamp the deficit as the
mask grows) — see `fit_conv_vae`'s docstring. The pre-print's `model.py`/`train.py` predate these.

### Model selection & bagging at small N

When N is in the hundreds, a few settings dominate map quality (from controlled 4 mm + 2 mm runs — real
KCH lesions, covariate-aware age/sex):

- **`mask_weight` is mask-size-invariant** under the default `loss_norm="voxel_mean"` — the
  reconstruction is a per-voxel mean, so the deficit isn't swamped as the mask grows. Verified: the
  same `mask_weight` recovers the substrate at 4 mm (V_cov 2 541) *and* 2 mm (V_cov 20 542) on a flat
  plateau (Dice@k 0.64–0.73 across `mask_weight` 1→10, always on left perisylvian cortex, beating VLSM
  0.51–0.54 / SVR 0.51–0.55). **Use `mask_weight=3`** (mid-plateau). With the legacy `loss_norm="sum"`
  you must instead scale `mask_weight` ∝ V_cov, which is the trap this replaces.
- **Coverage floor** (`--coverage-floor 0.1`, i.e. `readout_coverage_frac=0.1`) — the `E[σ(γ)·w]`
  readout is under-constrained on rarely-lesioned voxels and can spike there and capture the argmax;
  the floor zeros voxels lesioned in < 10 % of patients. (For the analysis mask itself, `build_voxel_data`
  takes `min_coverage_frac`.)
- **Held-out epoch selection** (`select_on="val"`, default) beats in-sample selection: single-fit
  Dice@k went **0.27 → 0.48** on the binary task (the continuous-severity task, already better-behaved,
  was unchanged). In-sample selection has no regularising stop, so it drifts to a seed-dependent
  optimum — worst on weak/binary outcomes.
- **Bag several seeds.** A single val-selected fit gets noisier the weaker the outcome, so average the
  per-seed maps and take the ensemble as the deliverable. Bagging raised split-half reproducibility
  monotonically (**binary 0.81 → 0.98, severity 0.90 → 0.99** as the bag grew 1 → 8) and the 16-fit
  ensemble was the most accurate; **held-out predictive validity was already high (AUC ≈ 0.90) and
  flat** — bagging buys *reproducibility*, not raw predictivity. Most of the gain is by ~4 fits and
  near-saturated by ~8. The honest stopping rule: **bag until split-half reproducibility and held-out
  predictive validity plateau** — a weak binary *presence* label plateaus much later than a graded
  severity score, so it needs more fits. To bag with the shipped scripts, run `aphasia_4mm.py` across
  seeds (one per GPU via the SLURM array recipe above) and average the output substrate NIfTIs.

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

Sweeps over `--epochs`, `--min-overlap`, or `--seed` parallelise the same way — vary the args per
array index. (Within a single fit the conv-VAE already uses the whole GPU; parallelism buys you
throughput *across* fits, not a faster individual fit.)

**Bagging across seeds** (recommended at small N — see the model-selection note above). Fit one seed
per GPU, then average the output NIfTIs into the ensemble map you actually report:

```bash
#SBATCH --array=0-15           # 16 seeds, one GPU each
#SBATCH --gpus-per-task=1
srun python src/scripts/aphasia_4mm.py --outcome binary --covariates age_sex.npy \
     --labels y.npy --lesions 'lesions/*.nii.gz' --seed $SLURM_ARRAY_TASK_ID --patience 100
# then, once all seeds finish, average them (this ensemble is the deliverable):
python -c "import glob,nibabel as nib,numpy as np; \
f=sorted(glob.glob('aphasia_4mm_binary_seed*.nii.gz')); i=[nib.load(x) for x in f]; \
nib.save(nib.Nifti1Image(np.mean([x.get_fdata() for x in i],0).astype('f4'), i[0].affine), \
'aphasia_4mm_binary_ensemble.nii.gz')"
```

> `covariate_substrate.ipynb` is an earlier **parcel-space** prototype of this tutorial; its shared
> modules were refactored for the voxel version above, so it is superseded and won't run as-is.

## Bagging (calibrated consensus)

A single conv-VAE fit is high-variance. `bagging.fit_conv_vae_bagged` bags several fits into a
low-variance substrate — but instead of averaging raw maps, it **binarises each seed at its own
held-out-calibrated size** (`metrics.focal_mass_mask`) and then averages the binaries into a per-voxel
consensus. This calibrate-per-seed-then-ensemble order is the point (see `SUBSTRATE_BAGGING.md`).

```python
import sys; sys.path.insert(0, "src")
from bagging import fit_conv_vae_bagged
res = fit_conv_vae_bagged(X, y, mask_idx=mask_idx, vol_shape=vol_shape, n_bag=8,
                          cov=cov, readout_coverage_frac=0.1)
res.consensus_mask       # deployed substrate (V_cov,)
res.consensus_fraction   # per-voxel seed agreement in [0, 1]
```
