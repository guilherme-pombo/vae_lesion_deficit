# vld Namespace Restructure — Implementation Plan

> Mechanical relocation + import-rewrite + packaging. See the design spec (same date) for the full
> structure, import-rewrite table, and verification checklist. Commit after each task; branch → PR.

**Env:** run Python/pytest with the worktree venv
`/Users/guilherme.pombo/lesion-modelling/.claude/worktrees/covariate-deficit-vae/components/anatomical-priors/.venv/bin/python`,
`PYTHONPATH=src` (before packaging) or after `pip`-less editable via `PYTHONPATH=src`.

### Task 1 — Quarantine legacy
- `mkdir legacy legacy/notebooks`; `git mv` model.py train.py utils.py cov_vae.py data_utils.py → `legacy/`;
  `git mv` 2d_toy_example, conditional_diffusion, fast_transformers_shap, simulation_viz,
  substrate_generative, covariate_substrate `.ipynb` → `legacy/notebooks/`.
- Write `legacy/README.md` (superseded by the voxel-space `vld` package; kept for reference).
- Commit.

### Task 2 — Move src tree into src/vld namespace
- `mkdir -p src/vld`; `git mv src/dlm src/vld/models`; `git mv src/baselines src/vld/baselines`;
  `git mv src/scripts src/vld/scripts`.
- `mkdir src/vld/data src/vld/deficits`; `git mv src/voxel_data.py src/build_voxel.py src/vld/data/`;
  `git mv src/deficits.py src/aphasia_deficits.py src/vld/deficits/`.
- `git mv src/metrics.py src/vld/metrics.py`; `git mv src/bagging.py src/vld/models/bagging.py`.
- Add `src/vld/__init__.py`, `src/vld/data/__init__.py`, `src/vld/deficits/__init__.py`,
  `src/vld/scripts/__init__.py` (models/baselines already have `__init__`).
- Commit (imports still broken — fixed next).

### Task 3 — Rewrite imports (per spec table)
- Apply the rewrite table to every `.py` under `src/vld/` (models/bagging.py, data/build_voxel.py,
  scripts/_aphasia_train.py, scripts/aphasia_2mm.py, scripts/aphasia_4mm.py). Drop the `sys.path.insert`
  block in `_aphasia_train.py`.
- Verify: `grep -rnE "from (dlm|baselines|metrics|voxel_data|build_voxel|deficits|aphasia_deficits|bagging|_aphasia_train)\b" src/` returns nothing.
- Verify import: `PYTHONPATH=src python -c "import vld, vld.models, vld.baselines, vld.data, vld.deficits, vld.metrics, vld.models.bagging"`.
- Commit.

### Task 4 — Packaging + top-level API
- `pyproject.toml`: add `[tool.setuptools] package-dir={""="src"}` and `[tool.setuptools.packages.find] where=["src"]`; mirror `requirements-covariate.txt` into `[project.dependencies]`.
- `src/vld/__init__.py`: re-export `fit_conv_vae`, `fit_conv_vae_bagged`, `consensus_substrate`, `focal_mass_mask`, `vlsm_stat`/`vlsm_binary`, `svr_weights`/`svr_binary`, `VoxelData`, `build_voxel_data`.
- Verify: `pip install -e .` (in a throwaway venv is ideal; else `PYTHONPATH=src` import check) and `import vld; vld.fit_conv_vae`.
- Commit.

### Task 5 — Notebooks + assets + docs
- `mkdir -p notebooks/assets docs`; `git mv` aphasia_examples, aphasia_confounds, covariate_voxel `.ipynb` → `notebooks/`.
- `git mv james_maps_binary.npz james_maps_severity.npz voxel_results.json notebooks/assets/`.
- Notebook edits (JSON, programmatic): rewrite imports to `vld.*`, remove `sys.path.insert(0,"src")`,
  repoint data loads to `assets/<file>` (paths are now relative to `notebooks/`).
- `git mv covariate_tutorial_spec.md SUBSTRATE_BAGGING.md docs/`.
- `.gitignore`: add `*.html`; `git rm --cached *.html`.
- Verify (structural): no notebook has `from dlm/baselines/metrics import`, none has `sys.path.insert(0,"src")`, all valid JSON.
- Commit.

### Task 6 — Tests
- `tests/test_bagging.py`: drop the `sys.path.insert(0,"src")` shim; import `from vld.models.bagging import …`.
- Run: `PYTHONPATH=src <venv>/python -m pytest tests/ -q`. Expect pass.
- Commit.

### Task 7 — README + final verification
- Rewrite `README.md`: `pip install -e .`, `vld.*` import examples, new layout, legacy pointer.
- Full check: import all subpackages; `pytest`; `python -m vld.scripts.aphasia_4mm --help`; notebook structural grep.
- Commit.

### Task 8 — Push + PR
- Push `modularise-vld` (personal creds), open PR to `main`, restore gh → holo.
