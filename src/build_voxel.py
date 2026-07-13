"""Build a :class:`VoxelData` from raw lesion NIfTIs — nibabel only, no plexus / S3 / cloud infra.

Turns a folder of MNI-registered binary lesion masks into the coverage-restricted voxel matrix the
methods consume, on a 4 mm MNI grid by default (the same grid as the shipped sample, so results are
comparable). This is the missing link between raw imaging and ``fit_conv_vae`` / ``fit_parcel_vae``:

    from build_voxel import build_voxel_data
    from dlm.conv_vae import fit_conv_vae
    data = build_voxel_data(sorted(glob("lesions/*.nii.gz")), min_overlap=10)
    smap = fit_conv_vae(data.X, y, mask_idx=data.mask_idx, vol_shape=data.vol_shape, cov=cov)

Lesions may be on any grid/resolution — they are resampled to the target grid through their affines,
so L/R and location are preserved as long as they are in the same (MNI) stereotaxic space.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from voxel_data import VoxelData

# Default target grid: 4 mm MNI, matching the shipped voxel_ds.npz sample.
MNI_4MM_SHAPE: tuple[int, int, int] = (46, 55, 46)
MNI_4MM_AFFINE: np.ndarray = np.array(
    [[4.0, 0.0, 0.0, -90.0],
     [0.0, 4.0, 0.0, -126.0],
     [0.0, 0.0, 4.0, -72.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
# 2 mm MNI (LAS) grid, for a finer-resolution build (resolution_mm=2).
MNI_2MM_SHAPE: tuple[int, int, int] = (91, 109, 91)
MNI_2MM_AFFINE: np.ndarray = np.array(
    [[-2.0, 0.0, 0.0, 90.0],
     [0.0, 2.0, 0.0, -126.0],
     [0.0, 0.0, 2.0, -72.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_GRIDS = {4: (MNI_4MM_SHAPE, MNI_4MM_AFFINE), 2: (MNI_2MM_SHAPE, MNI_2MM_AFFINE)}


def _resample(path: Path, shape: tuple[int, int, int], affine: np.ndarray, order: int) -> np.ndarray:
    """Resample one NIfTI onto ``(shape, affine)`` through its own affine; return a float32 volume."""
    res = resample_from_to(nib.load(path), (shape, affine), order=order, cval=0.0)
    return np.asarray(res.dataobj, dtype=np.float32)


def build_voxel_data(
    lesion_paths: list[str | Path],
    *,
    min_overlap: int = 10,
    resolution_mm: int = 4,
    template: str | Path | None = None,
    substrate_paths: dict[str, str | Path] | None = None,
    substrate_thresh: float = 0.25,
    order: int = 1,
) -> VoxelData:
    """Build a :class:`VoxelData` from binary lesion NIfTIs (MNI-registered).

    Parameters
    ----------
    lesion_paths:
        Binary lesion masks (one per patient), in MNI space. Resampled to the target grid.
    min_overlap:
        Coverage mask keeps voxels lesioned (>0) in at least this many patients — standard voxel-LSM
        practice (untested voxels nobody lesions carry no information) and the tractability lever.
    resolution_mm:
        Target isotropic resolution when ``template`` is None: ``4`` (default) or ``2``. 4 mm is the
        sensible default for cohorts in the hundreds; 2 mm is a finer refinement (~8× more voxels).
    template:
        Optional reference NIfTI defining the target grid; overrides ``resolution_mm``. Use this for a
        grid that isn't the built-in 4 mm / 2 mm MNI grid.
    substrate_paths:
        Optional ``{name: path}`` of substrate maps (e.g. NeuroQuery terms) for *evaluation* — each is
        resampled, max-normalised, and thresholded at ``substrate_thresh`` within the coverage mask.
        Omit for a purely data-driven analysis (no ground-truth substrate needed to fit).
    order:
        Resampling interpolation order (1 = trilinear coverage fraction; 0 = nearest/binary).

    Returns a :class:`VoxelData` with ``X`` ``(N, V_cov)``, ``mask_idx``, ``centroids``, ``vol_shape``,
    ``affine`` and (optional) ``substrates`` — ready for ``fit_conv_vae`` / the baselines.
    """
    if not lesion_paths:
        raise ValueError("lesion_paths is empty")
    if template is not None:
        t = nib.load(template)
        shape = tuple(int(s) for s in t.shape[:3])
        affine = np.asarray(t.affine, dtype=np.float64)
    elif resolution_mm in _GRIDS:
        shape, affine = _GRIDS[resolution_mm]
    else:
        raise ValueError(f"resolution_mm must be 4 or 2 (got {resolution_mm}); or pass a template")

    vols = np.stack([np.clip(_resample(Path(p), shape, affine, order), 0.0, 1.0)
                     for p in lesion_paths])  # (N, *shape)
    n = vols.shape[0]
    touched = (vols > 0.0).sum(axis=0)  # (*shape,)
    mask3d = touched >= int(min_overlap)
    if not mask3d.any():
        raise ValueError(
            f"coverage mask empty at min_overlap={min_overlap} (max touch={int(touched.max())}); "
            f"lower min_overlap or check the lesions are co-registered to the target grid"
        )
    mask_idx = np.flatnonzero(mask3d.reshape(-1))
    X = vols.reshape(n, -1)[:, mask_idx].astype(np.float32)

    ijk = np.argwhere(mask3d).astype(np.float64)
    centroids = (np.column_stack([ijk, np.ones(len(ijk))]) @ affine.T)[:, :3].astype(np.float32)

    substrates: dict[str, np.ndarray] = {}
    if substrate_paths:
        for name, sp in substrate_paths.items():
            s = _resample(Path(sp), shape, affine, order=1).reshape(-1)[mask_idx].astype(np.float64)
            s = s / (s.max() + 1e-12)
            substrates[name] = (s >= substrate_thresh).astype(bool)

    return VoxelData(X=X, substrates=substrates, centroids=centroids, mask_idx=mask_idx,
                     vol_shape=shape, affine=affine)


__all__ = ["build_voxel_data", "MNI_4MM_SHAPE", "MNI_4MM_AFFINE", "MNI_2MM_SHAPE", "MNI_2MM_AFFINE"]
