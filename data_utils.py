"""Load the shipped parcel-space lesion sample and map parcel maps back to a volume.

The tutorial works in Glasser-360 parcel space at 2 mm (our internal resolution). We ship a
512-lesion sample per substrate as parcel-overlap matrices ``X (N, K=360)`` (fraction of each
parcel lesioned), the binary substrates ``M (K,)``, parcel centroids, and a compact Glasser
label volume for glass-brain visualisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_episodes(data_dir: str = "new_data") -> dict:
    """Load the per-substrate episodes.

    Returns a dict with:
      ``X_all``     -- ``(S, N, K)`` parcel-overlap matrices (one 512-lesion sample per substrate),
      ``M_all``     -- ``(S, K)`` binary substrate masks,
      ``centroids`` -- ``(K, 3)`` parcel centroids (for the spatially-split deficit rules),
      ``names``     -- list of ``S`` substrate names.
    """
    d = Path(data_dir)
    return {
        "X_all": np.load(d / "episode_overlaps.npy"),
        "M_all": np.load(d / "substrates.npy"),
        "centroids": np.load(d / "parcel_centroids.npy"),
        "names": (d / "substrate_names.txt").read_text().splitlines(),
    }


def load_atlas(data_dir: str = "new_data") -> np.ndarray:
    """Load the Glasser-360 label volume ``(91, 109, 91)``: 0 = background, 1..K = parcel id."""
    return np.load(Path(data_dir) / "glasser_360_2mm_labels.npy")


def to_volume(parcel_vec: np.ndarray, atlas_labels: np.ndarray) -> np.ndarray:
    """Paint a ``(K,)`` parcel map onto the atlas -> ``(91, 109, 91)`` float volume for viz."""
    vec = np.asarray(parcel_vec, dtype=np.float64)
    vol = np.zeros(atlas_labels.shape, dtype=np.float64)
    nz = atlas_labels > 0
    vol[nz] = vec[atlas_labels[nz] - 1]
    return vol


__all__ = ["load_episodes", "load_atlas", "to_volume"]
