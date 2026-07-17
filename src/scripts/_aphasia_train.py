"""Shared training pipeline for the aphasia example scripts (aphasia_4mm.py / aphasia_2mm.py).

Plexus-free — numpy / torch / scikit-learn / nibabel only. The two entry-point scripts differ ONLY
in their resolution config (grid + conv depth/batch); everything else lives here.

Pipeline: lesions -> VoxelData (coverage-restricted voxel matrix) -> fit the two-arm conv-VAE
(covariate-aware, binary or continuous-severity head) -> save the recovered substrate as a NIfTI.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]   # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_voxel import build_voxel_data           # noqa: E402
from dlm.conv_vae import fit_conv_vae              # noqa: E402
from metrics import dice, dice_at_k, focal_mass_mask  # noqa: E402
from voxel_data import VoxelData                   # noqa: E402


def run_aphasia(*, resolution_mm: int, conv_num_down: int, conv_batch: int,
                sample_npz: Path, argv: list[str] | None = None) -> None:
    """Parse CLI args and train the conv-VAE for one resolution (see the entry-point scripts)."""
    ap = argparse.ArgumentParser(
        description=f"Aphasia substrate recovery at {resolution_mm} mm (conv-VAE, covariate-aware).")
    ap.add_argument("--lesions", help="glob of MNI-registered lesion NIfTIs "
                                       "(default: run on the shipped demo sample)")
    ap.add_argument("--labels", help=".npy of the outcome y — binary {0,1} or a continuous score")
    ap.add_argument("--covariates", help=".npy (N, d) nuisance covariates, e.g. [age, sex]; "
                                          "omit for a covariate-blind fit")
    ap.add_argument("--outcome", choices=["binary", "severity"], default="binary",
                    help="binary classifier (BCE head) or continuous severity (MSE head)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--min-overlap", type=int, default=10, help="coverage-mask threshold")
    ap.add_argument("--seed", type=int, default=0,
                    help="fit seed — vary it across a bag of runs, then average the output NIfTIs "
                         "(recommended at small N; see the README small-N / bagging note)")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop patience (epochs w/o held-out improvement); default runs all "
                         "--epochs and keeps the best held-out epoch")
    ap.add_argument("--coverage-floor", type=float, default=0.1,
                    help="zero the recovered map on voxels lesioned in < this fraction of patients "
                         "(stops the readout spiking on rarely-lesioned voxels; 0 disables)")
    ap.add_argument("--mask-weight", type=float, default=3.0,
                    help="deficit-vs-reconstruction weight. Mask-size-invariant under the default "
                         "voxel_mean loss; 3 sits mid-plateau (verified 4mm & 2mm). Raise if the "
                         "substrate looks under-learned/smeared")
    ap.add_argument("--out", default=None,
                    help="output NIfTI (default: aphasia_<res>mm_<outcome>_seed<seed>.nii.gz)")
    args = ap.parse_args(argv)
    continuous = args.outcome == "severity"
    out_path = args.out or f"aphasia_{resolution_mm}mm_{args.outcome}_seed{args.seed}.nii.gz"

    # 1. lesions -> VoxelData (X: N x V_cov coverage matrix + coverage mask + grid metadata)
    if args.lesions:
        paths = sorted(glob.glob(args.lesions))
        if not paths:
            raise SystemExit(f"no lesion NIfTIs matched: {args.lesions}")
        data = build_voxel_data(paths, resolution_mm=resolution_mm, min_overlap=args.min_overlap)
    else:
        data = VoxelData.load(str(sample_npz))
        print(f"[demo] no --lesions given → shipped {resolution_mm} mm sample")

    # 2. outcome y (+ optional covariates)
    if args.labels:
        y = np.load(args.labels).astype(np.float32)
        if len(y) != data.n_patients:
            raise SystemExit(f"--labels has {len(y)} rows but data has {data.n_patients} patients")
    else:  # demo target from the language substrate (an aphasia proxy)
        overlap = data.X[:, data.substrates["language"]].mean(1)
        y = (overlap if continuous
             else (overlap > np.quantile(overlap, 0.6)).astype(np.float32)).astype(np.float32)
        print(f"[demo] synthetic {'severity' if continuous else 'aphasia-yes/no'} target "
              f"from the language substrate")
    cov = np.load(args.covariates).astype(np.float32) if args.covariates else None

    # 3. train the conv-VAE — covariate-aware if cov given; continuous (MSE) head for severity
    print(f"[train] conv-VAE  {resolution_mm} mm  outcome={args.outcome}  aware={cov is not None}  "
          f"N={data.n_patients}  V_cov={data.n_voxels}  seed={args.seed}  (num_down={conv_num_down}, "
          f"batch={conv_batch}, epochs={args.epochs}, patience={args.patience})", flush=True)
    smap = fit_conv_vae(data.X, y, mask_idx=data.mask_idx, vol_shape=data.vol_shape, cov=cov,
                        continuous=continuous, epochs=args.epochs, num_down=conv_num_down,
                        batch=conv_batch, seed=args.seed, patience=args.patience,
                        mask_weight=args.mask_weight, readout_coverage_frac=args.coverage_floor)

    # 4. save the recovered substrate map as a NIfTI (+ score vs language if this is the demo)
    import nibabel as nib
    nib.save(nib.Nifti1Image(data.to_volume(smap).astype(np.float32), data.affine), out_path)
    print(f"[done] substrate map -> {out_path}", flush=True)
    if not args.labels and "language" in data.substrates:
        m = data.substrates["language"]; k = int(m.sum())
        print(f"[demo] recovery vs language substrate: Dice@k={dice_at_k(smap, m, k):.3f}  "
              f"binaryDice={dice(focal_mass_mask(smap), m):.3f}")
