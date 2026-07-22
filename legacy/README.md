# Legacy (superseded)

These are the original **2D-toy / parcel-space** prototypes, kept for reference. They are **superseded
by the voxel-space `vld` package** under `src/vld/` and are not part of the installed package.

- `model.py`, `train.py`, `utils.py` — the original joint-VAE lesion-deficit model, training loop, and
  data loaders (2D-toy scale). The maintained conv/MLP VAEs live in `vld.models`.
- `cov_vae.py`, `data_utils.py` — early covariate-VAE prototype and data helpers.
- `notebooks/` — the toy/parcel demos (`2d_toy_example`, `conditional_diffusion`,
  `fast_transformers_shap`, `simulation_viz`, `substrate_generative`, `covariate_substrate`). They
  import the legacy modules above, not `vld`.

For current usage see the top-level `README.md`, the `notebooks/` tutorials, and `vld.*`.
