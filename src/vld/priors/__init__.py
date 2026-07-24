"""Spatial priors for substrate estimation.

``spatial`` — Laplacian-penalised (graph-Tikhonov) regression, the ill-posed-inverse counterpart to
ordinary ridge. See ``docs/SPATIAL_PRIOR.md`` for when it helps and when it does not.
"""

from vld.priors.spatial import (
    LaplacianInverse,
    fit_laplacian_substrate,
    voxel_adjacency,
    voxel_laplacian,
)

__all__ = [
    "LaplacianInverse",
    "fit_laplacian_substrate",
    "voxel_adjacency",
    "voxel_laplacian",
]
