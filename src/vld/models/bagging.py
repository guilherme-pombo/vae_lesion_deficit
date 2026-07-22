"""Bagged substrate recovery for the two-arm VAEs.

A single VAE fit is high-variance: different weight inits and fit/cal splits give maps that correlate
only ~0.5 across seeds, so one fit can under-perform a stable univariate baseline (VLSM) despite the
VAE's de-biasing. Bagging removes most of that variance — provided each bag member is calibrated
*before* it is ensembled.

Each member is a full :func:`dlm.conv_vae.fit_conv_vae` fit at a distinct ``seed`` (which drives both
the weight init and the fit/cal split). ``fit_conv_vae`` already calibrates the returned map on its own
held-out calibration split (the paper's §3.2 rate sweep, mass-scaled so :func:`metrics.focal_mass_mask`
self-sizes to the calibrated substrate size). So the recipe is:

    1. fit each seed  ->  calibrated ``(V_cov,)`` map;
    2. binarise each with ``focal_mass_mask`` (its own calibrated size);
    3. average the binaries into a per-voxel consensus fraction (seed agreement) and deploy the voxels
       whose agreement meets ``agreement`` (0.5 = simple majority).

The consensus fraction is an average of already-calibrated binaries, NOT a calibrated-rate map, so it is
never re-thresholded with ``focal_mass_mask`` — calibration happens per member, on held-out data, before
the ensemble. No ground-truth substrate size is used anywhere, so this is a valid deployment procedure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vld.metrics import focal_mass_mask

from .conv_vae import fit_conv_vae


@dataclass(frozen=True)
class BaggedSubstrate:
    """Result of a calibrated bagging run.

    ``consensus_mask`` is the deployed boolean substrate ``(V_cov,)``; ``consensus_fraction`` the
    per-voxel seed agreement in [0, 1]; ``per_seed_binaries``/``per_seed_maps`` the individual
    calibrated binaries and raw ``(V_cov,)`` maps.
    """

    consensus_mask: np.ndarray
    consensus_fraction: np.ndarray
    per_seed_binaries: list[np.ndarray]
    per_seed_maps: list[np.ndarray]


def consensus_substrate(binaries, *, agreement: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble already-calibrated per-seed BINARY maps into a consensus.

    ``binaries`` is an iterable of boolean/0-1 arrays (one per seed, same shape), each already binarised
    at its own held-out-calibrated size. Returns ``(mask, fraction)``: the per-voxel ``fraction`` (share
    of seeds selecting each voxel) and the deployed ``mask`` (``fraction >= agreement``). Do NOT pass raw
    continuous maps here — binarise each seed with :func:`metrics.focal_mass_mask` first.
    """
    stack = np.stack([np.asarray(b, dtype=np.float64) for b in binaries], axis=0)
    fraction = stack.mean(axis=0)
    return fraction >= agreement, fraction


def fit_conv_vae_bagged(X: np.ndarray, Y: np.ndarray, *, mask_idx: np.ndarray,
                        vol_shape: tuple[int, int, int], n_bag: int = 8, base_seed: int = 0,
                        agreement: float = 0.5, cov: np.ndarray | None = None,
                        continuous: bool = False, **conv_kwargs) -> BaggedSubstrate:
    """Bagged conv-VAE substrate: calibrate each seed on its own held-out split, then ensemble.

    Fits :func:`dlm.conv_vae.fit_conv_vae` ``n_bag`` times at seeds ``base_seed .. base_seed+n_bag-1``
    (each seed varies both the weight init and the fit/cal split), binarises each returned (already
    calibrated) map with :func:`metrics.focal_mass_mask`, and averages the binaries into a consensus
    (see module docstring). ``cov``/``continuous`` and any extra ``conv_kwargs`` (``epochs``, ``lr``,
    ``loss_norm``, ``readout_coverage_frac``, ...) are forwarded verbatim to each fit.
    """
    maps = [fit_conv_vae(X, Y, mask_idx=mask_idx, vol_shape=vol_shape, cov=cov,
                         continuous=continuous, seed=base_seed + s, **conv_kwargs)
            for s in range(n_bag)]
    binaries = [focal_mass_mask(m) for m in maps]
    mask, fraction = consensus_substrate(binaries, agreement=agreement)
    return BaggedSubstrate(consensus_mask=mask, consensus_fraction=fraction,
                           per_seed_binaries=binaries, per_seed_maps=maps)
