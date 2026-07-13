"""Shared helpers for the LSM baselines: nuisance residualization + BH-FDR."""

from __future__ import annotations

import numpy as np


def _residualize(y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """OLS residual of ``y`` on the design ``[1, cov]`` (nuisance regression).

    This is the *partialing-out* covariate control (DeMarco & Turkeltaub 2018 / svrlsmgui
    standard): the behaviour is regressed on the covariates before the test/fit, so the
    lesion map is estimated against the confound-adjusted deficit.
    """
    y64 = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(y64)), np.asarray(cov, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(design, y64, rcond=None)
    return y64 - design @ coef


def _fdr_bh(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns a boolean mask (same shape as ``p_values``) where ``True`` indicates rejection
    of the null at level ``q``.

    Parameters
    ----------
    p_values:
        ``(m,)`` array of per-feature p-values in ``[0, 1]``.
    q:
        FDR level (default 0.05).
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = np.arange(1, m + 1) / m * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    max_k = int(np.max(np.where(below)[0]))
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx[: max_k + 1]] = True
    return rejected


__all__ = ["_residualize", "_fdr_bh"]
