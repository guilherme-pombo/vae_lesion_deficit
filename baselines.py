"""Parcel/voxel-space VLSM and SVR-LSM baselines for synthetic substrate recovery.

Observed-only continuous stat maps in a feature space of dimension ``K`` (parcels or
coverage voxels) that can be ranked and scored with the same ``dice`` / ``dice_at_k``
metrics used elsewhere. Both methods accept optional nuisance covariates (``cov``) with
the DeMarco & Turkeltaub (2018) partialing-out control as the default.

References
----------
- VLSM: Bates et al. (2003), Rorden & Karnath (2004). Vectorised Welch t-test.
- SVR-LSM: Zhang et al. (2014). dTLVC normalisation + RBF SVR + back-projection.
- Covariate control: DeMarco & Turkeltaub (2018) partialing-out (svrlsmgui standard).
"""

from __future__ import annotations

import numpy as np


def _residualize(y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """OLS residual of ``y`` on the design ``[1, cov]`` (nuisance regression)."""
    y64 = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(y64)), np.asarray(cov, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(design, y64, rcond=None)
    return y64 - design @ coef


def _augment_with_cov(x_norm: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    """Append z-scored ``cov`` columns to ``x_norm`` as SVR nuisance features.

    Returns ``x_norm`` unchanged when ``cov`` is None. Feature columns stay first,
    so callers back-project onto ``[:K]`` to drop the covariate columns.
    """
    if cov is None:
        return x_norm
    cov_f = np.asarray(cov, dtype=np.float64)
    cov_z = (cov_f - cov_f.mean(axis=0)) / (cov_f.std(axis=0) + 1e-8)
    return np.hstack([x_norm, cov_z])


# ── Welch t-test VLSM ────────────────────────────────────────────────────


def vlsm_stat(
    X: np.ndarray,
    y: np.ndarray,
    *,
    delta: float,
    cov: np.ndarray | None = None,
) -> np.ndarray:
    """Absolute Welch t-statistic per feature (continuous VLSM).

    Binarises each patient's feature overlap at ``delta`` (present = X > delta),
    then computes a fully vectorised Welch t-test between the ``y`` values of
    patients with the feature present vs absent for each feature simultaneously.

    Parameters
    ----------
    X:
        ``(N, K)`` continuous overlap matrix (values in [0, 1]).
    y:
        ``(N,)`` continuous or binary deficit scores.
    delta:
        Per-feature damage threshold; a feature is "present" for a patient
        when ``X[i, k] > delta``.
    cov:
        Optional ``(N, C)`` nuisance covariates. When given, ``y`` is
        residualized on ``[1, cov]`` (OLS) before the Welch t-test.

    Returns
    -------
    stats:
        ``(K,)`` absolute Welch t-statistics. Features with fewer than 2
        patients in either group receive a statistic of 0.
    """
    if cov is not None:
        y = _residualize(y, cov)
    N, K = X.shape
    lesion_bool = X > delta  # (N, K) bool
    lesion_float = lesion_bool.astype(np.float64)  # (N, K)

    n_hit = lesion_float.sum(axis=0)  # (K,)
    n_miss = N - n_hit  # (K,)
    valid = (n_hit >= 2) & (n_miss >= 2)  # (K,) bool

    labels_f = y.astype(np.float64)  # (N,)
    labels_sq = labels_f ** 2

    # Group sums via matrix multiply: (K,) = (N, K)^T @ (N,)
    hit_sum = lesion_float.T @ labels_f  # (K,)
    hit_sq_sum = lesion_float.T @ labels_sq  # (K,)
    total_sum = labels_f.sum()
    total_sq_sum = labels_sq.sum()
    miss_sum = total_sum - hit_sum
    miss_sq_sum = total_sq_sum - hit_sq_sum

    with np.errstate(divide="ignore", invalid="ignore"):
        hit_mean = hit_sum / n_hit
        miss_mean = miss_sum / n_miss
        # Variance with Bessel's correction
        hit_var = (hit_sq_sum - hit_sum ** 2 / n_hit) / (n_hit - 1)
        miss_var = (miss_sq_sum - miss_sum ** 2 / n_miss) / (n_miss - 1)
        se = np.sqrt(hit_var / n_hit + miss_var / n_miss)
        t_stat = (hit_mean - miss_mean) / se

    stats = np.zeros(K, dtype=np.float64)
    stats[valid] = np.abs(np.nan_to_num(t_stat[valid], nan=0.0))
    return stats


# ── SVR-LSM ──────────────────────────────────────────────────────────────


def _resolve_rbf_gamma(x_norm: np.ndarray, gamma: float | str) -> float:
    """Resolve the RBF ``gamma``; ``"scale"`` → ``1/(n_features * X.var())``.

    The string ``"scale"`` mirrors scikit-learn's data-driven default, which is
    calibrated to the feature geometry — essential in parcel space, where Zhang's
    voxel-space constant (``5.0``) mis-scales the kernel.
    """
    if isinstance(gamma, str):
        if gamma != "scale":
            raise ValueError(f"unknown gamma mode {gamma!r}; use 'scale' or a float")
        return 1.0 / (x_norm.shape[1] * float(x_norm.var()) + 1e-12)
    return float(gamma)


def _svr_cov_features(
    x_norm: np.ndarray, y: np.ndarray, cov: np.ndarray | None, cov_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the chosen covariate control, returning ``(features, target)`` for the SVR.

    ``"partial"`` (default, DeMarco & Turkeltaub 2018 / svrlsmgui standard) residualizes the
    behaviour on ``[1, cov]`` before fitting and leaves the lesion features untouched — the
    lesion map is then fit to the confound-adjusted behaviour. ``"features"`` (legacy) appends
    standardized covariate columns as nuisance features; this mis-scales the RBF kernel (the few
    covariate columns dominate the unit-norm dTLVC lesion vector) and barely perturbs the
    back-projected lesion map, so it neither changes the prediction nor controls the confound.
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
    """Absolute back-projected SVR weights per feature (Zhang et al. 2014 pipeline).

    dTLVC (L2-normalise each patient's lesion vector) → z-score the deficit →
    RBF-kernel epsilon-SVR → back-project the dual coefficients to feature space
    (``dual_coef_ @ X[support_]``). ``gamma='scale'`` (default) sets the RBF width
    from the data (``1/(K·var)``); ``C`` is the SVR regularisation. When ``cov`` is given,
    ``cov_mode`` selects covariate control: ``"partial"`` (default) residualizes the deficit on
    ``[1, cov]`` before fitting (DeMarco & Turkeltaub 2018 partialing-out — the standard);
    ``"features"`` appends standardized covariate columns as nuisance features (legacy; back-
    projects the feature columns only). Returns the ``(K,)`` absolute weight per feature.
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


# ── Dice metrics ───────────────────────────────────────────────────────────


def dice_at_k(score_map: np.ndarray, M_binary: np.ndarray, k: int) -> float:
    """Dice@K: binarise ``score_map`` at the top-k and compute Dice vs ``M_binary``.

    Parameters
    ----------
    score_map:
        ``(K,)`` continuous scores (higher = more likely in substrate).
    M_binary:
        ``(K,)`` binary ground-truth substrate mask.
    k:
        Number of top features to select (typically ``int(M_binary.sum())``).

    Returns
    -------
    dice:
        ``2 * |top_k ∩ M| / (k + |M|)`` in ``[0, 1]``.
    """
    pred = np.zeros_like(M_binary, dtype=np.float64)
    pred[np.argsort(-score_map.astype(np.float64))[:k]] = 1.0
    inter = float((pred * M_binary).sum())
    denom = float(pred.sum() + M_binary.sum())
    return 2.0 * inter / denom if denom > 0 else 0.0


def dice(pred_bool: np.ndarray, m_bool: np.ndarray) -> float:
    """Full binary Dice coefficient (penalises over- and under-prediction).

    Returns ``2 * |pred ∩ M| / (|pred| + |M|)``, or 0.0 when both are empty.

    Parameters
    ----------
    pred_bool:
        ``(K,)`` predicted binary mask (any dtype; interpreted as bool).
    m_bool:
        ``(K,)`` ground-truth binary mask (any dtype; interpreted as bool).
    """
    a = pred_bool.astype(bool)
    b = m_bool.astype(bool)
    intersection = float(np.logical_and(a, b).sum())
    total = float(a.sum() + b.sum())
    return 2.0 * intersection / total if total > 0.0 else 0.0


# ── FDR correction ───────────────────────────────────────────────────────


def _fdr_bh(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns a boolean mask (same shape as ``p_values``) where ``True``
    indicates rejection of the null at level ``q``.

    Parameters
    ----------
    p_values:
        ``(m,)`` array of per-feature p-values in ``[0, 1]``.
    q:
        FDR level (default 0.05).

    Returns
    -------
    rejected:
        ``(m,)`` boolean mask.
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


# ── Permutation-based binary decisions ────────────────────────────────────


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
    """VLSM binary decision via permutation + BH-FDR.

    Computes the observed Welch t-statistic at each feature, then estimates
    a per-feature permutation p-value by shuffling ``y`` ``n_perm`` times and
    counting how often the permuted statistic meets or exceeds the observed
    one.  Applies Benjamini-Hochberg FDR at level ``q`` to produce a binary
    rejection mask.

    Parameters
    ----------
    X:
        ``(N, K)`` continuous overlap matrix (values in [0, 1]).
    y:
        ``(N,)`` binary or continuous deficit labels.
    delta:
        Per-feature damage threshold (forwarded to :func:`vlsm_stat`).
    n_perm:
        Number of label permutations.
    q:
        BH-FDR level (default 0.05).
    seed:
        RNG seed for reproducibility.
    cov:
        Optional ``(N, C)`` nuisance covariates. When given, ``y`` is
        residualized once on ``[1, cov]`` and the permutation null is run on
        the residual.

    Returns
    -------
    binary_mask:
        ``(K,)`` boolean array; ``True`` = feature declared significant.
    observed_stat:
        ``(K,)`` observed Welch t-statistics (for inspection / ranking).
    """
    rng = np.random.default_rng(seed)
    K = X.shape[1]

    y_eff = _residualize(y, cov) if cov is not None else y
    observed = vlsm_stat(X, y_eff, delta=delta)
    perm_exceeds = np.zeros(K, dtype=np.int64)

    for _ in range(n_perm):
        y_perm = y_eff[rng.permutation(len(y_eff))]
        perm_stat = vlsm_stat(X, y_perm, delta=delta)
        perm_exceeds += (perm_stat >= observed).astype(np.int64)

    p = (perm_exceeds + 1) / (n_perm + 1)
    binary_mask = _fdr_bh(p, q=q)
    return binary_mask, observed


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

    dTLVC + z-score + RBF-kernel SVR (kernel precomputed once and reused across
    permutations); for each of ``n_perm`` label permutations the SVR is refit and its
    absolute back-projected weights recorded. When ``cov`` is given, ``cov_mode`` selects
    covariate control: ``"partial"`` (default) residualizes the deficit on ``[1, cov]`` before
    fitting (DeMarco & Turkeltaub 2018 partialing-out — the standard; the permutation null
    shuffles the residualized deficit); ``"features"`` appends standardized covariate columns
    as nuisance features and back-projects the feature columns only (legacy).

    ``correction`` selects the multiple-comparison control:

    * ``"fwer_maxstat"`` (default) — Westfall-Young max-statistic FWER: a feature is
      significant when its observed weight exceeds the ``(1 - q)`` quantile of the
      per-permutation *maximum* weight. Calibrated for low-dimensional parcel space
      (K~360); in high-dimensional voxel space (K~10³–10⁴) the max-null is a very high
      bar and this collapses to near-empty masks.
    * ``"fdr"`` — voxelwise permutation p-values ``(exceeds+1)/(n_perm+1)`` with
      Benjamini-Hochberg FDR at ``q``. The standard voxel SVR-LSM thresholding
      (Zhang et al. 2014, voxelwise p-maps); use this in voxel space.

    Returns ``(binary_mask (K,), observed_weights (K,))``.
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
        p = (exceeds + 1) / (n_perm + 1)
        binary_mask = _fdr_bh(p, q=q)
    else:
        threshold = float(np.quantile(max_null, 1.0 - q))
        binary_mask = observed > threshold
    return binary_mask, observed


# ── Calibrated focal mask ──────────────────────────────────────────────────


def focal_mass_mask(scores: np.ndarray) -> np.ndarray:
    """Binary mask of the top-``round(Σp)`` features of a calibrated probability map.

    For a map calibrated so ``mean(p) == r`` (the VAE's learned substrate rate),
    the total mass ``Σp == r·K`` is the expected substrate size — so selecting the
    top-``round(Σp)`` features is the **calibration-implied** focal threshold. Use
    this for diffuse (non-saturated) score maps like the VAE effective-weight readout,
    where an F1 self-sizing rule over-selects. Returns a ``(K,)`` bool mask.
    """
    p = scores.astype(np.float64)
    k = int(round(float(p.sum())))
    if k <= 0:
        return np.zeros(p.shape[0], dtype=bool)
    k = min(k, p.shape[0])
    mask = np.zeros(p.shape[0], dtype=bool)
    mask[np.argsort(-p)[:k]] = True
    return mask


# ── Confound-leakage diagnostics ───────────────────────────────────────────


def age_exposure(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-feature age-exposure ``e_k = corr_i(x[:, k], age)`` (age = ``cov[:, 0]``).

    High ``|e_k|`` flags features whose damage tracks the confounder; the leakage
    diagnostic asks how much a recovered map loads onto these off-target features.
    """
    age = np.asarray(cov, dtype=np.float64)[:, 0]
    age_c = age - age.mean()
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    num = xc.T @ age_c
    den = np.linalg.norm(xc, axis=0) * np.linalg.norm(age_c) + 1e-12
    return num / den


def leakage(
    score_map: np.ndarray, m_binary: np.ndarray, exposure: np.ndarray
) -> float:
    """``corr(|score|, |exposure|)`` over NON-substrate features (0 if degenerate).

    Lower is better: an unbiased map places no extra weight on the age-exposed
    off-target features, so its off-target scores are uncorrelated with exposure.
    """
    off = np.asarray(m_binary) <= 0.5
    s = np.abs(np.asarray(score_map, dtype=np.float64))[off]
    e = np.abs(np.asarray(exposure, dtype=np.float64))[off]
    if len(s) < 2 or s.std() == 0.0 or e.std() == 0.0:
        return 0.0
    return float(np.corrcoef(s, e)[0, 1])


__all__ = [
    "vlsm_stat",
    "vlsm_binary",
    "svr_weights",
    "svr_binary",
    "dice",
    "dice_at_k",
    "focal_mass_mask",
    "age_exposure",
    "leakage",
    "_resolve_rbf_gamma",
    "_fdr_bh",
]
