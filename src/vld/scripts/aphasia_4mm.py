#!/usr/bin/env python
"""Train the conv-VAE for aphasia substrate recovery at 4 mm — binary or severity, covariate-aware.

Runs out-of-box on the shipped 4 mm demo sample, or point it at your own MNI-registered data.
Plexus-free (numpy / torch / scikit-learn / nibabel); a GPU is used automatically if present.

    # demo on the shipped sample
    python -m vld.scripts.aphasia_4mm                       # binary aphasia (yes/no)
    python -m vld.scripts.aphasia_4mm --outcome severity    # continuous severity

    # your own cohort (X built from lesion NIfTIs; y binary or continuous; cov = [age, sex])
    python -m vld.scripts.aphasia_4mm --lesions 'lesions/*.nii.gz' --labels y.npy \
        --covariates age_sex.npy --outcome binary --out my_substrate.nii.gz
"""
from pathlib import Path

from vld.scripts._aphasia_train import run_aphasia

# 4 mm config — conv downsamples the (46,55,46) grid 3× to a (6,7,6) latent.
_REPO = Path(__file__).resolve().parents[3]
run_aphasia(resolution_mm=4, conv_num_down=3, conv_batch=64,
            sample_npz=_REPO / "voxel_data_sample" / "voxel_ds.npz")
