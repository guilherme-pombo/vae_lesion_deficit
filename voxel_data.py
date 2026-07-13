"""Voxel-space dataset for the covariate lesion-deficit benchmark (4 mm MNI).

A lightweight loader for the voxel benchmark inputs: a coverage-restricted analysis
mask on a 4 mm grid, the per-voxel lesion coverage matrix ``X (N, V_cov)``, the
NeuroQuery substrate masks, and voxel mm centroids. These feed the dimension-agnostic
methods (VLSM / SVR-LSM / VAE) and the covariate deficit generator unchanged.

This is the public, self-contained counterpart to the internal builder: it reads the
``.npz`` the internal pipeline saves (no nibabel / S3 / build dependency) and exposes
the same fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VoxelData:
    """Voxel-space benchmark inputs on a coverage-restricted 4 mm analysis mask.

    ``X`` is ``(N, V_cov)`` continuous per-voxel lesion coverage in [0, 1] (the
    voxel analog of parcel overlap). ``substrates`` maps each substrate name to its
    ``(V_cov,)`` boolean ground-truth mask. ``centroids`` is ``(V_cov, 3)`` voxel mm
    coordinates (for the spatial deficit rules). ``mask_idx`` are the flat indices of
    the analysis voxels within the ``vol_shape`` volume, used by :meth:`to_volume` to
    scatter a ``(V_cov,)`` score map back for rendering. ``affine`` is the 4 mm RAS
    affine of that volume.
    """

    X: np.ndarray
    substrates: dict[str, np.ndarray]
    centroids: np.ndarray
    mask_idx: np.ndarray
    vol_shape: tuple[int, int, int]
    affine: np.ndarray

    @property
    def n_voxels(self) -> int:
        """Number of analysis voxels ``V_cov`` (feature dimension of ``X``)."""
        return int(self.X.shape[1])

    @property
    def n_patients(self) -> int:
        """Number of patients ``N`` (rows of ``X``)."""
        return int(self.X.shape[0])

    def to_volume(self, scores: np.ndarray) -> np.ndarray:
        """Scatter a ``(V_cov,)`` score map into a full ``vol_shape`` volume (zeros elsewhere)."""
        vol = np.zeros(int(np.prod(self.vol_shape)), dtype=np.float64)
        vol[self.mask_idx] = np.asarray(scores, dtype=np.float64)
        return vol.reshape(self.vol_shape)

    @classmethod
    def load(cls, path: Path) -> "VoxelData":
        """Reconstruct a :class:`VoxelData` from the benchmark ``.npz``.

        Reads the keys the internal ``VoxelBenchmarkData.save`` writes: ``X``,
        ``substrate_names``, ``substrate_masks``, ``centroids``, ``mask_idx``,
        ``vol_shape`` and ``affine``.
        """
        z = np.load(path, allow_pickle=False)
        names = [str(n) for n in z["substrate_names"]]
        substrates = {n: z["substrate_masks"][i].astype(bool) for i, n in enumerate(names)}
        return cls(
            X=z["X"].astype(np.float32),
            substrates=substrates,
            centroids=z["centroids"].astype(np.float32),
            mask_idx=z["mask_idx"],
            vol_shape=tuple(int(s) for s in z["vol_shape"]),
            affine=z["affine"].astype(np.float64),
        )


__all__ = ["VoxelData"]
