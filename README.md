# vae_lesion_deficit (`vld`)

Lesion-deficit substrate recovery in voxel space — VLSM and SVR-LSM baselines, two-arm VAEs (MLP and
3D-convolutional) with a covariate-aware deficit head, and calibrated bagging. Everything runs with
**numpy / torch / scikit-learn / nibabel** only — no cloud/infra dependency.

Start with `notebooks/covariate_voxel.ipynb` (method comparison) or `notebooks/aphasia_examples.ipynb`
(clinical worked examples).

## Setup

Requires Python ≥ 3.10. The code is an installable package (`vld`) under `src/`.

```bash
pip install -e .                 # then `import vld` anywhere
pip install -e ".[notebook]"     # + Jupyter, to run the tutorials
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra notebook         # creates .venv/ with all deps + Jupyter
uv run jupyter lab               # open notebooks/covariate_voxel.ipynb
```

Everything is standard PyPI. `requirements-covariate.txt` mirrors the core deps for a bare `pip`.

## Package layout

```
src/vld/
  models/     conv_vae.py (fit_conv_vae, 3D-conv; GPU)  mlp_vae.py (fit_parcel_vae)
              bagging.py (fit_conv_vae_bagged — calibrated consensus)
  baselines/  vlsm.py (vlsm_stat/vlsm_binary)  svr_lsm.py (svr_weights/svr_binary)  _common.py
  data/       voxel_data.py (VoxelData)  build_voxel.py (build_voxel_data — NIfTI → VoxelData)
  deficits/   deficits.py (covariate benchmark generator)  aphasia_deficits.py
  priors/     spatial.py (LaplacianInverse, fit_laplacian_substrate — spatially-coherent
              substrate estimation; voxel_adjacency / voxel_laplacian)
  metrics.py  dice, dice_at_k, focal_mass_mask, support_auc, morans_i, age_exposure, leakage
  scripts/    aphasia_4mm.py  aphasia_2mm.py  _aphasia_train.py  spatial_prior_demo.py
notebooks/    covariate_voxel.ipynb  aphasia_examples.ipynb  aphasia_confounds.ipynb  assets/
docs/         covariate_tutorial_spec.md  SUBSTRATE_BAGGING.md  SPATIAL_PRIOR.md
voxel_data_sample/   the shipped 4 mm / 2 mm demo samples (voxel_ds.npz, voxel_ds_2mm.npz)
legacy/       superseded 2D-toy / parcel-space prototypes (see legacy/README.md)
```

Common imports: `from vld.models import fit_conv_vae, fit_conv_vae_bagged`,
`from vld.baselines import vlsm_stat, svr_weights`, `from vld.data import VoxelData, build_voxel_data`,
`from vld.metrics import dice_at_k, focal_mass_mask`. Headline symbols are also re-exported from the
top level (`from vld import fit_conv_vae`).

## Covariate-aware substrate recovery in voxel space

**`notebooks/covariate_voxel.ipynb`** recovers the lesion–deficit substrate in **4 mm MNI voxel space**,
comparing four methods **covariate-blind vs covariate-aware** on 512 real KCH stroke lesions:

- **VLSM** (mass-univariate Welch-t + permutation FDR),
- **SVR-LSM** (multivariate RBF SVR + back-projection),
- **MLP-VAE** (the two-arm VAE, membership × importance), and
- **Conv-VAE** — the same two-arm VAE with a **3D-convolutional** encoder/decoder (a spatial prior).

Headline: the **conv-VAE recovers the substrate best**, and **covariate-awareness helps it** while
leaking the least confound bias — whereas the linear methods over-correct.

**Two covariate-control mechanisms** (both legitimate; they differ in *where* the control happens): the
**baselines** use classical **partialling-out** — residualize the behaviour on `[1, cov]` before
fitting (`cov_mode="partial"`). The **VAEs** control covariates in the **deficit head**: `cov` enters
only the head as a main effect (`cov·β`) and a substrate interaction (`(1 + cov·γ)` scaling the
overlap), fitted jointly, while the encoder never sees `cov` — so the recovered substrate stays
covariate-invariant. The head main-effect is asymptotically equivalent to partialling out the main
effect (Frisch–Waugh–Lovell), done inside the model, plus an interaction the pre-residualization can't
express.

Data (self-contained; no downloads): `voxel_data_sample/voxel_ds.npz` (V_cov=2541 voxels × 512 patients
+ 6 substrates), `notebooks/assets/voxel_results.json` (precomputed GPU means), `imgs/` (figures). The
fast methods run live in the notebook; the conv-VAE is GPU-trained, so its numbers are precomputed (a
snippet reproduces them on a GPU).

**Relationship to the pre-print.** `vld.models.conv_vae` (`fit_conv_vae`) is the maintained
implementation — it keeps the pre-print's CoordConv / label-as-volume / joint-`P(X,Y)` conv
architecture but adds the identifiable two-arm `σ(γ)·w` readout, the covariate deficit head, GroupNorm,
a continuous/severity head, coverage-restriction, a **mask-size-invariant loss**
(`loss_norm="voxel_mean"`, default), a **readout coverage floor**, and **held-out validation epoch
selection** (`select_on="val"`, default). The pre-print's original `model.py`/`train.py` now live under
`legacy/`.

### Model selection & bagging at small N

When N is in the hundreds, a few settings dominate map quality (controlled 4 mm + 2 mm runs, real KCH
lesions, covariate-aware age/sex):

- **`mask_weight` is mask-size-invariant** under `loss_norm="voxel_mean"` — reconstruction is a
  per-voxel mean, so the deficit isn't swamped as the mask grows. The same `mask_weight` recovers the
  substrate at 4 mm (V_cov 2 541) *and* 2 mm (V_cov 20 542) on a flat plateau (Dice@k 0.64–0.73, always
  on left perisylvian cortex, beating VLSM 0.51–0.54 / SVR 0.51–0.55). **Use `mask_weight=3`.**
- **Coverage floor** (`readout_coverage_frac=0.1`) — zeros voxels lesioned in < 10 % of patients, where
  the `E[σ(γ)·w]` readout is under-constrained and can spuriously spike.
- **Held-out epoch selection** (`select_on="val"`, default) beats in-sample: single-fit Dice@k
  **0.27 → 0.48** on the binary task.
- **Bag several seeds** (see below): raises split-half reproducibility monotonically
  (**binary 0.81 → 0.98, severity 0.90 → 0.99** as the bag grows 1 → 8); held-out predictive validity is
  already high (AUC ≈ 0.90) and flat — bagging buys *reproducibility*. Most gain by ~4 fits,
  near-saturated by ~8; weak binary labels plateau later than graded severity.

## Bring your own data — no cloud infra required

```python
from glob import glob
import numpy as np
from vld.data import build_voxel_data
from vld.models import fit_conv_vae            # or fit_parcel_vae (CPU-friendly)

# 1. lesions -> coverage-restricted 4 mm voxel matrix (resampled through affines; nibabel only)
data = build_voxel_data(sorted(glob("my_lesions/*.nii.gz")), min_overlap=10)   # 2 mm: resolution_mm=2

# 2. outcome + covariates
y   = np.load("aphasia_label.npy")            # binary {0,1}  OR  a continuous severity score
cov = np.load("age_sex.npy")                  # (N, 2) = [age, sex]; omit for a covariate-blind fit

# 3. recover the substrate (covariate-aware). continuous=True for a severity/regression outcome.
smap = fit_conv_vae(data.X, y, mask_idx=data.mask_idx, vol_shape=data.vol_shape,
                    cov=cov, continuous=False, epochs=300)          # -> (V_cov,) per-voxel map
vol  = data.to_volume(smap)                   # paint back to a (46,55,46) volume for viewing/saving
```

**Resolution:** 4 mm is the sensible default for cohorts in the hundreds; go to 2 mm
(`resolution_mm=2`, or `template=<reference NIfTI>`) as a refinement once 4 mm looks good.

## Calibrated bagging

A single conv-VAE fit is high-variance. `vld.models.fit_conv_vae_bagged` bags several fits into a
low-variance substrate — and instead of averaging raw maps, it **binarises each seed at its own
held-out-calibrated size** (`focal_mass_mask`) and then averages the binaries into a per-voxel
consensus. This calibrate-per-seed-then-ensemble order is the point (see `docs/SUBSTRATE_BAGGING.md`).

```python
from vld.models import fit_conv_vae_bagged
res = fit_conv_vae_bagged(X, y, mask_idx=mask_idx, vol_shape=vol_shape, n_bag=8,
                          cov=cov, readout_coverage_frac=0.1)
res.consensus_mask       # deployed substrate (V_cov,)
res.consensus_fraction   # per-voxel seed agreement in [0, 1]
```

## Spatially-coherent substrates (and why prediction is the wrong yardstick)

Lesion-deficit mapping is an ill-posed inverse problem — with K ≫ N voxels the penalty, not the data,
decides the map. Ridge penalises `‖w‖²`, which treats neighbouring voxels as unrelated; penalising
*roughness* instead (`wᵀLw`, with `L` the voxel graph Laplacian) says neighbouring voxels should agree.
That is the LORETA penalty family from EEG source localisation, underdetermined in the same way.

It matters because **ridge is the best of these estimators at held-out prediction and the worst at
recovering the substrate**. On the shipped sample cohort with a planted ground truth, ridge has the highest
out-of-fold ρ and the lowest support-AUC in all six tested conditions, while every smoothing weight in a
broad range beats it on recovery by +0.09 to +0.25 AUC. Tuning a substrate on prediction picks a worse map.

```python
from vld.priors import LaplacianInverse, fit_laplacian_substrate
from vld.metrics import morans_i, support_auc

w = fit_laplacian_substrate(X, y, mask_idx=mask_idx, vol_shape=vol_shape, beta=100.0, cov=cov)

# sweeping beta / folds / bootstraps: build once, fit many times (one sparse factorisation)
solver = LaplacianInverse(X, mask_idx=mask_idx, vol_shape=vol_shape, beta=100.0, cov=cov)
half = solver.fit(y[idx], rows=idx)
```

```bash
python -m vld.scripts.spatial_prior_demo        # reproduces the benchmark table
```

Two cautions, both covered in `docs/SPATIAL_PRIOR.md`: smoothing inflates split-half stability
*mechanically*, so report `morans_i` alongside any reproducibility claim and compare at matched smoothness;
and the prior adds no information — it makes better use of what the cohort constrains rather than
enlarging it, so significance testing must push the null through the same estimator and prior.

## Ready-to-run training scripts

End-to-end torch scripts (build the voxel matrix, train the covariate-aware conv-VAE, write a NIfTI).
Both run **out-of-box on the shipped demo sample** and switch to your cohort with
`--lesions/--labels/--covariates`:

```bash
python -m vld.scripts.aphasia_4mm                        # demo, binary aphasia (yes/no)
python -m vld.scripts.aphasia_4mm --outcome severity     # continuous severity
python -m vld.scripts.aphasia_2mm                        # finer 2 mm grid (use a GPU)

python -m vld.scripts.aphasia_4mm \
    --lesions 'my_lesions/*.nii.gz' --labels y.npy --covariates age_sex.npy \
    --outcome binary --out my_substrate.nii.gz
```

A GPU is used automatically when present (`cuda` → `mps` → `cpu`). The shared pipeline is
`vld.scripts._aphasia_train`.

**Parallel / bagging on SLURM.** Each fit is one independent process — fan out with an array job (one
GPU per task):

```bash
#SBATCH --array=0-15           # 16 seeds, one GPU each
#SBATCH --gpus-per-task=1
srun python -m vld.scripts.aphasia_4mm --outcome binary --covariates age_sex.npy \
     --labels y.npy --lesions 'lesions/*.nii.gz' --seed $SLURM_ARRAY_TASK_ID --patience 100
# then average the per-seed substrate NIfTIs into the ensemble deliverable, or use
# vld.models.fit_conv_vae_bagged for the calibrated consensus.
```

## Legacy

The original 2D-toy / parcel-space prototypes (`model.py`, `train.py`, `utils.py`, `cov_vae.py`,
`data_utils.py` and their notebooks) live under `legacy/`, superseded by the voxel-space `vld` package.
See `legacy/README.md`.
