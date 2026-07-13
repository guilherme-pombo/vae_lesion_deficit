"""SVR-LSM — multivariate lesion-symptom mapping (Zhang et al. 2014).

dTLVC (L2-normalise each patient's lesion vector) → z-score the deficit → RBF-kernel
epsilon-SVR → back-project the dual coefficients to feature space. Covariate control defaults
to partialing-out (``cov_mode="partial"``, DeMarco & Turkeltaub 2018) — appending covariates as
features (``"features"``) mis-scales the kernel and is kept only for comparison.
"""

from __future__ import annotations

import numpy as np

from ._common import _fdr_bh, _residualize


def _resolve_rbf_gamma(x_norm: np.ndarray, gamma: float | str) -> float:
    """Resolve the RBF ``gamma``; ``"scale"`` → ``1/(n_features * X.var())`` (sklearn-style)."""
    if isinstance(gamma, str):
        if gamma != "scale":
            raise ValueError(f"unknown gamma mode {gamma!r}; use 'scale' or a float")
        return 1.0 / (x_norm.shape[1] * float(x_norm.var()) + 1e-12)
    return float(gamma)


def _augment_with_cov(x_norm: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    """Append z-scored ``cov`` columns as SVR nuisance features (legacy ``cov_mode='features'``)."""
    if cov is None:
        return x_norm
    cov_f = np.asarray(cov, dtype=np.float64)
    cov_z = (cov_f - cov_f.mean(axis=0)) / (cov_f.std(axis=0) + 1e-8)
    return np.hstack([x_norm, cov_z])


def _svr_cov_features(
    x_norm: np.ndarray, y: np.ndarray, cov: np.ndarray | None, cov_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the chosen covariate control, returning ``(features, target)`` for the SVR.

    ``"partial"`` (default) residualizes the behaviour on ``[1, cov]`` and leaves the lesion
    features untouched. ``"features"`` (legacy) appends standardized covariate columns as
    nuisance features — mis-scaled (the few covariate columns dominate the unit-norm dTLVC
    vector), so it barely perturbs the back-projected map and neither changes the prediction
    nor controls the confound.
    """
    if cov is None:
        return x_norm, np.asarray(y, dtype=np.float64)
    if cov_mode == "partial":
        return x_norm, _residualize(y, cov)
    if cov_mode == "features":
        return _augment_with_cov(x_norm, cov), np.asarray(y, dtype=np.float64)
    raise ValueError(f"unknown cov_mode {cov_mode!r}; use 'partial' or 'features'")


def svr_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    gamma: float | str = "scale",
    cov: np.ndarray | None = None,
    cov_mode: str = "partial",
) -> np.ndarray:
    """Absolute back-projected SVR weights per feature (observed-only; no permutation).

    ``gamma='scale'`` sets the RBF width from the data (``1/(K·var)``); ``C`` is the SVR
    regularisation. When ``cov`` is given, ``cov_mode`` selects the covariate control (see
    :func:`_svr_cov_features`). Returns the ``(K,)`` absolute weight per feature.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVR

    K_feat = X.shape[1]
    X_f = X.astype(np.float64)
    X_norm = X_f / (np.linalg.norm(X_f, axis=-1, keepdims=True) + 1e-5)
    feats, y_f = _svr_cov_features(X_norm, y, cov, cov_mode)
    y_z = (y_f - y_f.mean()) / (y_f.std() + 1e-8)
    g = _resolve_rbf_gamma(feats, gamma)
    K_mat = rbf_kernel(feats, gamma=g)
    model = SVR(kernel="precomputed", C=C).fit(K_mat, y_z)
    backproj = (model.dual_coef_ @ feats[model.support_]).ravel()
    return np.abs(backproj[:K_feat])  # feature columns only


def svr_binary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 1000,
    q: float = 0.05,
    C: float = 1.0,
    gamma: float | str = "scale",
    seed: int = 0,
    cov: np.ndarray | None = None,
    correction: str = "fwer_maxstat",
    cov_mode: str = "partial",
) -> tuple[np.ndarray, np.ndarray]:
    """SVR-LSM binary decision via permutation testing on back-projected weights.

    The kernel is precomputed once and reused; each of ``n_perm`` label permutations refits the
    SVR and records the absolute back-projected weights. ``cov``/``cov_mode`` control covariates
    (see :func:`_svr_cov_features`). ``correction`` selects the multiple-comparison control:

    * ``"fwer_maxstat"`` (default) — Westfall-Young max-statistic FWER (calibrated for
      low-dimensional parcel space; collapses to near-empty masks at ~10³–10⁴ voxels).
    * ``"fdr"`` — voxelwise permutation p-values + Benjamini-Hochberg FDR (Zhang 2014 voxelwise
      p-maps; use this in voxel space).

    Returns ``(binary_mask (K,) bool, observed_weights (K,))``.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVR

    if correction not in ("fwer_maxstat", "fdr"):
        raise ValueError(f"unknown correction {correction!r}; use 'fwer_maxstat' or 'fdr'")

    rng = np.random.default_rng(seed)
    X_f = X.astype(np.float64)
    K = X_f.shape[1]
    X_norm = X_f / (np.linalg.norm(X_f, axis=-1, keepdims=True) + 1e-5)
    feats, y_f = _svr_cov_features(X_norm, y, cov, cov_mode)
    y_z = (y_f - y_f.mean()) / (y_f.std() + 1e-8)
    g = _resolve_rbf_gamma(feats, gamma)
    K_mat = rbf_kernel(feats, gamma=g)

    def _weights(labels: np.ndarray) -> np.ndarray:
        m = SVR(kernel="precomputed", C=C).fit(K_mat, labels)
        w = np.abs((m.dual_coef_ @ feats[m.support_]).ravel()[:K])
        if w.shape[0] != K:  # defensive: support collapse
            full = np.zeros(K, dtype=np.float64)
            full[: w.shape[0]] = w
            return full
        return w

    observed = _weights(y_z)
    max_null = np.empty(n_perm, dtype=np.float64)
    exceeds = np.zeros(K, dtype=np.int64)  # per-feature exceed counts (for FDR)
    for i in range(n_perm):
        perm_w = _weights(y_z[rng.permutation(len(y_z))])
        max_null[i] = float(perm_w.max())
        exceeds += (perm_w >= observed).astype(np.int64)

    if correction == "fdr":
        binary_mask = _fdr_bh((exceeds + 1) / (n_perm + 1), q=q)
    else:
        binary_mask = observed > float(np.quantile(max_null, 1.0 - q))
    return binary_mask, observed


__all__ = ["svr_weights", "svr_binary", "_resolve_rbf_gamma"]
