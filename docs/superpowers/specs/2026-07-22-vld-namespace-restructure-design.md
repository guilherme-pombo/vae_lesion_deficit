# Modularise into the `vld` namespace package

Date: 2026-07-22
Status: approved design (pre-implementation)

## Goal

Turn the current mixed repo (a self-contained `src/` tree imported via `sys.path.insert(0,"src")`,
plus an older 2D-toy/parcel-space world at root) into a single installable namespace package `vld`,
with the legacy world quarantined and root artifacts organised.

## Non-goals

- No behavioural changes to any module — this is relocation + import-rewrite + packaging only.
- No rewrite of notebook science — only import lines, `sys.path` removal, and asset paths change.
- Nothing pushed to `main` directly (feature branch → PR).

## Target structure

```
vae_lesion_deficit/
  pyproject.toml          # installable: vld under src/, deps, python -m entry points
  .gitignore              # + *.html, __pycache__, .venv, *.egg-info
  README.md               # pip install -e ., vld.* imports, new layout
  src/vld/
    __init__.py           # re-export key API
    models/    __init__.py conv_vae.py mlp_vae.py bagging.py     (was src/dlm/* + src/bagging.py)
    baselines/ __init__.py vlsm.py svr_lsm.py _common.py          (was src/baselines/*)
    data/      __init__.py voxel_data.py build_voxel.py           (was src/voxel_data.py, src/build_voxel.py)
    deficits/  __init__.py deficits.py aphasia_deficits.py        (was src/deficits.py, src/aphasia_deficits.py)
    metrics.py                                                     (was src/metrics.py)
    scripts/   __init__.py aphasia_2mm.py aphasia_4mm.py _aphasia_train.py
  notebooks/   aphasia_examples.ipynb aphasia_confounds.ipynb covariate_voxel.ipynb
    assets/    james_maps_binary.npz james_maps_severity.npz voxel_results.json
  docs/        covariate_tutorial_spec.md SUBSTRATE_BAGGING.md   (+ superpowers/ specs & plans)
  tests/       test_bagging.py
  legacy/
    README.md
    model.py train.py utils.py cov_vae.py data_utils.py
    notebooks/ 2d_toy_example.ipynb conditional_diffusion.ipynb fast_transformers_shap.ipynb
               simulation_viz.ipynb substrate_generative.ipynb covariate_substrate.ipynb
```

Note `dlm` → `models` (the subpackage is renamed), and `src/bagging.py` → `src/vld/models/bagging.py`.

## Import-rewrite rules

Within-subpackage relative imports are unchanged (`from .conv_vae import …`, `from ._common import …`).
Absolute imports that relied on `sys.path` are rewritten to the `vld` namespace:

| Current | New |
|---|---|
| `from dlm.conv_vae import fit_conv_vae` | `from vld.models.conv_vae import fit_conv_vae` |
| `from dlm import …` | `from vld.models import …` |
| `from baselines… import …` | `from vld.baselines… import …` |
| `from metrics import …` | `from vld.metrics import …` |
| `from voxel_data import VoxelData` | `from vld.data.voxel_data import VoxelData` |
| `from build_voxel import build_voxel_data` | `from vld.data.build_voxel import build_voxel_data` |
| `from deficits import …` / `from aphasia_deficits import …` | `from vld.deficits… import …` |
| `from bagging import …` | `from vld.models.bagging import …` |
| `from _aphasia_train import run_aphasia` | `from vld.scripts._aphasia_train import run_aphasia` |

Specific internal fixes after the move:
- `models/bagging.py`: `from dlm.conv_vae import fit_conv_vae` → `from .conv_vae import fit_conv_vae`;
  `from metrics import focal_mass_mask` → `from vld.metrics import focal_mass_mask`.
- `data/build_voxel.py`: `from voxel_data import VoxelData` → `from .voxel_data import VoxelData`;
  `from dlm.conv_vae import fit_conv_vae` → `from vld.models.conv_vae import fit_conv_vae`;
  self-referential `from build_voxel import build_voxel_data` (in the `__main__`/docstring example) →
  `from vld.data.build_voxel import build_voxel_data`.
- `scripts/_aphasia_train.py`: drop the `sys.path.insert` block; rewrite the four imports to `vld.*`.
- `scripts/aphasia_{2mm,4mm}.py`: drop `sys.path` shim; import `from vld.scripts._aphasia_train import
  run_aphasia`; runnable as `python -m vld.scripts.aphasia_4mm`.

## Packaging

`pyproject.toml` adds:
```toml
[tool.setuptools]
package-dir = {"" = "src"}
[tool.setuptools.packages.find]
where = ["src"]
```
so `pip install -e .` exposes `vld` and its subpackages. All `sys.path.insert(0,"src")` lines are
removed (src, tests, notebooks). `requirements-covariate.txt` is retained; its contents are mirrored in
`[project.dependencies]`.

## Notebooks & artifacts

- Current notebooks (`aphasia_examples`, `aphasia_confounds`, `covariate_voxel`) → `notebooks/`; their
  imports rewritten to `vld.*`, `sys.path.insert(0,"src")` removed, and data-load paths repointed to
  `assets/` (relative to `notebooks/`).
- `james_maps_*.npz`, `voxel_results.json` → `notebooks/assets/` (still tracked; referenced by the
  notebooks).
- `.html` exports → untracked + gitignored (regenerable).
- `docs/`: `covariate_tutorial_spec.md`, `SUBSTRATE_BAGGING.md` moved in.

## Verification

- `pip install -e .` succeeds; `import vld` and each subpackage import resolves.
- `pytest` (test_bagging) passes with the updated `from vld.models.bagging import …`.
- `python -m vld.scripts.aphasia_4mm --help` runs (argparse help, no import error).
- Structural notebook check: no `from dlm/baselines/metrics/… import`, no `sys.path.insert(0,"src")`,
  asset loads reference `assets/`; every notebook is valid JSON.
- Notebooks are NOT executed end-to-end (need full env + data) — flagged for a manual run.
