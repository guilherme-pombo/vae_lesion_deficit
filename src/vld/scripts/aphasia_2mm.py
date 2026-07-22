#!/usr/bin/env python
"""Train the conv-VAE for aphasia substrate recovery at 2 mm — binary or severity, covariate-aware.

Same as aphasia_4mm.py but on the finer 2 mm grid (~8× more voxels): the conv downsamples one extra
level and uses a smaller batch to fit GPU memory. 2 mm is a refinement for when 4 mm already works
and you want finer localisation (needs more patients / a GPU); otherwise prefer 4 mm.

    python -m vld.scripts.aphasia_2mm                       # demo on the shipped 2 mm sample
    python -m vld.scripts.aphasia_2mm --outcome severity
    python -m vld.scripts.aphasia_2mm --lesions 'lesions/*.nii.gz' --labels y.npy --covariates age_sex.npy
"""
from pathlib import Path

from vld.scripts._aphasia_train import run_aphasia

# 2 mm config — conv downsamples the (91,109,91) grid 4× to a (6,7,6) latent; smaller batch for memory.
_REPO = Path(__file__).resolve().parents[3]
run_aphasia(resolution_mm=2, conv_num_down=4, conv_batch=32,
            sample_npz=_REPO / "voxel_data_sample" / "voxel_ds_2mm.npz")
