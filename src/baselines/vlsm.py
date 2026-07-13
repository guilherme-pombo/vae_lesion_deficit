"""VLSM — voxel/parcel lesion-symptom mapping via a vectorised Welch t-test.

Bates et al. (2003), Rorden & Karnath (2004). Optional nuisance covariates use the
partialing-out control (residualize the behaviour on ``[1, cov]``) from ``_common``.
"""

from __future__ import annotations

import numpy as np

from ._common import _fdr_bh, _residualize


def vlsm_stat(
    X: np.ndarray,
    y: np.ndarray,
    *,
    delta: float,
    cov: np.ndarray | None = None,
) -> np.ndarray:
    """Absolute Welch t-statistic per feature (continuous VLSM).

    Binarises each patient's feature overlap at ``delta`` (present = X > delta), then
    computes a fully vectorised Welch t-test between the ``y`` values of patients with the
    feature present vs absent, for every feature simultaneously.

    Parameters
    ----------
    X:
        ``(N, K)`` continuous overlap matrix (values in [0, 1]).
    y:
        ``(N,)`` continuous or binary deficit scores.
    delta:
        Per-feature damage threshold; a feature is "present" for a patient when ``X[i, k] > delta``.
    cov:
        Optional ``(N, C)`` nuisance covariates; ``y`` is residualized on ``[1, cov]`` first.

    Returns
    -------
    stats:
        ``(K,)`` absolute Welch t-statistics; features with < 2 patients in either group get 0.
    """
    if cov is not None:
        y = _residualize(y, cov)
    N, K = X.shape
    lesion_float = (X > delta).astype(np.float64)  # (N, K)

    n_hit = lesion_float.sum(axis=0)  # (K,)
    n_miss = N - n_hit  # (K,)
    valid = (n_hit >= 2) & (n_miss >= 2)  # (K,) bool

    labels_f = y.astype(np.float64)  # (N,)
    labels_sq = labels_f ** 2

    hit_sum = lesion_float.T @ labels_f  # (K,)
    hit_sq_sum = lesion_float.T @ labels_sq  # (K,)
    total_sum = labels_f.sum()
    total_sq_sum = labels_sq.sum()
    miss_sum = total_sum - hit_sum
    miss_sq_sum = total_sq_sum - hit_sq_sum

    with np.errstate(divide="ignore", invalid="ignore"):
        hit_mean = hit_sum / n_hit
        miss_mean = miss_sum / n_miss
        hit_var = (hit_sq_sum - hit_sum ** 2 / n_hit) / (n_hit - 1)  # Bessel-corrected
        miss_var = (miss_sq_sum - miss_sum ** 2 / n_miss) / (n_miss - 1)
        se = np.sqrt(hit_var / n_hit + miss_var / n_miss)
        t_stat = (hit_mean - miss_mean) / se

    stats = np.zeros(K, dtype=np.float64)
    stats[valid] = np.abs(np.nan_to_num(t_stat[valid], nan=0.0))
    return stats


def vlsm_binary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    delta: float,
    n_perm: int = 1000,
    q: float = 0.05,
    seed: int = 0,
    cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """VLSM binary decision via label-permutation p-values + Benjamini-Hochberg FDR.

    Shuffles ``y`` ``n_perm`` times, counts how often each feature's permuted statistic meets
    or exceeds the observed one, and rejects at FDR ``q``. With ``cov`` the deficit is
    residualized once and the permutation null is run on the residual. Returns
    ``(binary_mask (K,) bool, observed_stat (K,))``.
    """
    rng = np.random.default_rng(seed)
    K = X.shape[1]
    y_eff = _residualize(y, cov) if cov is not None else y
    observed = vlsm_stat(X, y_eff, delta=delta)
    perm_exceeds = np.zeros(K, dtype=np.int64)
    for _ in range(n_perm):
        perm_stat = vlsm_stat(X, y_eff[rng.permutation(len(y_eff))], delta=delta)
        perm_exceeds += (perm_stat >= observed).astype(np.int64)
    p = (perm_exceeds + 1) / (n_perm + 1)
    return _fdr_bh(p, q=q), observed


__all__ = ["vlsm_stat", "vlsm_binary"]
