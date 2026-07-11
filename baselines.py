"""VLSM and SVR-LSM baselines with optional covariate control, plus scoring utilities.

Parcel-space (or any ``(N, K)`` feature space). Each method has a covariate-**blind** mode
(``cov=None``, the classic baseline) and a covariate-**aware** mode:

* VLSM residualizes the deficit on ``[1, cov]`` (nuisance regression) before the Welch t-test.
* SVR-LSM appends standardized covariate columns as nuisance features and back-projects
  only the parcel columns.

References: VLSM -- Bates et al. 2003; SVR-LSM -- Zhang et al. 2014, DeMarco & Turkeltaub 2018.
"""

from __future__ import annotations

import numpy as np


# -- covariate helpers ------------------------------------------------------

def _residualize(y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """OLS residual of ``y`` on the design ``[1, cov]`` (nuisance regression)."""
    y64 = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(y64)), np.asarray(cov, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(design, y64, rcond=None)
    return y64 - design @ coef


def _augment_with_cov(x_norm: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    """Append z-scored ``cov`` columns to ``x_norm`` as SVR nuisance features.

    Returns ``x_norm`` unchanged when ``cov`` is None. Parcel columns stay first, so callers
    back-project onto ``[:K]`` to drop the covariate columns.
    """
    if cov is None:
        return x_norm
    cov_f = np.asarray(cov, dtype=np.float64)
    cov_z = (cov_f - cov_f.mean(axis=0)) / (cov_f.std(axis=0) + 1e-8)
    return np.hstack([x_norm, cov_z])


# -- VLSM -------------------------------------------------------------------

def vlsm_stat(X: np.ndarray, y: np.ndarray, *, delta: float = 0.1,
              cov: np.ndarray | None = None) -> np.ndarray:
    """Absolute Welch t-statistic per parcel (continuous VLSM).

    A parcel is "present" for a patient when ``X > delta`` (default 0.1 for continuous parcel overlaps).
    When ``cov`` is given, ``y`` is residualized on ``[1, cov]`` first. Parcels with < 2
    patients in either group get a statistic of 0.
    """
    if cov is not None:
        y = _residualize(y, cov)
    N, K = X.shape
    lesion = (X > delta).astype(np.float64)
    n_hit = lesion.sum(axis=0)
    n_miss = N - n_hit
    valid = (n_hit >= 2) & (n_miss >= 2)
    yf = y.astype(np.float64)
    ysq = yf ** 2
    hit_sum = lesion.T @ yf
    hit_sq = lesion.T @ ysq
    tot_sum, tot_sq = yf.sum(), ysq.sum()
    miss_sum, miss_sq = tot_sum - hit_sum, tot_sq - hit_sq
    with np.errstate(divide="ignore", invalid="ignore"):
        hm, mm = hit_sum / n_hit, miss_sum / n_miss
        hv = (hit_sq - hit_sum ** 2 / n_hit) / (n_hit - 1)
        mv = (miss_sq - miss_sum ** 2 / n_miss) / (n_miss - 1)
        t = (hm - mm) / np.sqrt(hv / n_hit + mv / n_miss)
    stats = np.zeros(K, dtype=np.float64)
    stats[valid] = np.abs(np.nan_to_num(t[valid], nan=0.0))
    return stats


def _fdr_bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR: boolean rejection mask at level ``q``."""
    m = len(p)
    order = np.argsort(p)
    thresh = np.arange(1, m + 1) / m * q
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    k = int(np.max(np.where(below)[0]))
    rej = np.zeros(m, dtype=bool)
    rej[order[: k + 1]] = True
    return rej


def vlsm_binary(X: np.ndarray, y: np.ndarray, *, delta: float = 0.1, n_perm: int = 500,
                q: float = 0.05, seed: int = 0, cov: np.ndarray | None = None) -> np.ndarray:
    """VLSM significant-parcel mask via label permutation + BH-FDR.

    Residualizes once on ``cov`` (if given), then permutes the residual for the null.
    """
    rng = np.random.default_rng(seed)
    y_eff = _residualize(y, cov) if cov is not None else y
    observed = vlsm_stat(X, y_eff, delta=delta)
    exceed = np.zeros(X.shape[1], dtype=np.int64)
    for _ in range(n_perm):
        perm = vlsm_stat(X, y_eff[rng.permutation(len(y_eff))], delta=delta)
        exceed += (perm >= observed).astype(np.int64)
    return _fdr_bh((exceed + 1) / (n_perm + 1), q=q)


# -- SVR-LSM ----------------------------------------------------------------

def _resolve_rbf_gamma(x_norm: np.ndarray, gamma: float | str) -> float:
    """Resolve the RBF ``gamma``; ``"scale"`` -> ``1/(n_features * X.var())``."""
    if isinstance(gamma, str):
        if gamma != "scale":
            raise ValueError(f"unknown gamma mode {gamma!r}; use 'scale' or a float")
        return 1.0 / (x_norm.shape[1] * float(x_norm.var()) + 1e-12)
    return float(gamma)


def svr_lsm_weights(X: np.ndarray, y: np.ndarray, *, C: float = 1.0,
                    gamma: float | str = "scale", cov: np.ndarray | None = None) -> np.ndarray:
    """Absolute back-projected SVR weights per parcel (Zhang et al. 2014 pipeline).

    dTLVC (L2-normalise each patient's lesion) -> z-score deficit -> RBF SVR -> back-project
    dual coefficients. When ``cov`` is given, standardized covariates are appended as nuisance
    features and only the parcel columns are back-projected. Returns ``(K,)`` absolute weights.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVR

    K = X.shape[1]
    Xf = X.astype(np.float64)
    x_norm = Xf / (np.linalg.norm(Xf, axis=-1, keepdims=True) + 1e-5)
    feats = _augment_with_cov(x_norm, cov)
    yz = (y.astype(np.float64) - y.mean()) / (y.std() + 1e-8)
    g = _resolve_rbf_gamma(feats, gamma)
    model = SVR(kernel="precomputed", C=C).fit(rbf_kernel(feats, gamma=g), yz)
    backproj = (model.dual_coef_ @ feats[model.support_]).ravel()
    return np.abs(backproj[:K])


def svr_lsm_binary(X: np.ndarray, y: np.ndarray, *, n_perm: int = 500, q: float = 0.05,
                   C: float = 1.0, gamma: float | str = "scale", seed: int = 0,
                   cov: np.ndarray | None = None) -> np.ndarray:
    """SVR-LSM significant-parcel mask via max-statistic (Westfall-Young) FWER permutation."""
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVR

    rng = np.random.default_rng(seed)
    K = X.shape[1]
    Xf = X.astype(np.float64)
    x_norm = Xf / (np.linalg.norm(Xf, axis=-1, keepdims=True) + 1e-5)
    feats = _augment_with_cov(x_norm, cov)
    yz = (y.astype(np.float64) - y.mean()) / (y.std() + 1e-8)
    g = _resolve_rbf_gamma(feats, gamma)
    K_mat = rbf_kernel(feats, gamma=g)
    base = SVR(kernel="precomputed", C=C).fit(K_mat, yz)
    observed = np.abs((base.dual_coef_ @ feats[base.support_]).ravel()[:K])
    max_null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        pm = SVR(kernel="precomputed", C=C).fit(K_mat, yz[rng.permutation(len(yz))])
        w = np.abs((pm.dual_coef_ @ feats[pm.support_]).ravel()[:K])
        max_null[i] = float(w.max()) if w.size else 0.0
    return observed > float(np.quantile(max_null, 1.0 - q))


# -- scoring / thresholding -------------------------------------------------

def dice(pred: np.ndarray, m: np.ndarray) -> float:
    """Binary Dice ``2|pred and M| / (|pred| + |M|)``; 0 if both empty."""
    a, b = np.asarray(pred, dtype=bool), np.asarray(m, dtype=bool)
    inter = float(np.logical_and(a, b).sum())
    tot = float(a.sum() + b.sum())
    return 2.0 * inter / tot if tot > 0 else 0.0


def dice_at_k(score_map: np.ndarray, m: np.ndarray, k: int | None = None) -> float:
    """Dice of the top-``k`` voxels of ``score_map`` vs ``M`` (``k`` defaults to ``|M|``).

    The standard rank-based LSM recovery metric: it scores how well a continuous stat map
    *ranks* substrate parcels, independent of each method's self-thresholding (which is
    ill-calibrated at large voxel counts). Higher = better.
    """
    m = np.asarray(m)
    k = int(m.sum()) if k is None else int(k)
    if k <= 0:
        return 0.0
    pred = np.zeros(m.shape[0], dtype=bool)
    pred[np.argsort(-np.asarray(score_map, dtype=np.float64))[:k]] = True
    return dice(pred, m)


def focality(mask: np.ndarray, m: np.ndarray) -> dict:
    """``{'precision','recall','n'}`` of a boolean mask vs the GT mask."""
    mask, mb = np.asarray(mask, dtype=bool), np.asarray(m, dtype=bool)
    n = int(mask.sum())
    tp = int((mask & mb).sum())
    return {"precision": tp / n if n else 0.0,
            "recall": tp / int(mb.sum()) if mb.sum() else 0.0, "n": float(n)}


def focal_mass_mask(scores: np.ndarray) -> np.ndarray:
    """Top-``round(sum(scores))`` voxels of a calibrated probability/weight map.

    For the VAE effective-weight readout (scaled so total mass == expected substrate size),
    this self-sizes the mask instead of flooding. Returns a ``(K,)`` bool mask.
    """
    p = np.asarray(scores, dtype=np.float64)
    k = int(round(float(p.sum())))
    if k <= 0:
        return np.zeros(p.shape[0], dtype=bool)
    k = min(k, p.shape[0])
    out = np.zeros(p.shape[0], dtype=bool)
    out[np.argsort(-p)[:k]] = True
    return out


def age_exposure(X: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-parcel age-exposure ``e_k = corr_i(X[:, k], age)`` (age = ``cov[:, 0]``)."""
    age = np.asarray(cov, dtype=np.float64)[:, 0]
    age_c = age - age.mean()
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    den = np.linalg.norm(Xc, axis=0) * np.linalg.norm(age_c) + 1e-12
    return (Xc.T @ age_c) / den


def leakage(score_map: np.ndarray, m: np.ndarray, exposure: np.ndarray) -> float:
    """``corr(|score|, |age-exposure|)`` over NON-substrate parcels (lower = less bias).

    Read jointly with Dice: a near-empty mask has trivially low leakage.
    """
    off = np.asarray(m) <= 0.5
    s = np.abs(np.asarray(score_map, dtype=np.float64))[off]
    e = np.abs(np.asarray(exposure, dtype=np.float64))[off]
    if len(s) < 2 or s.std() == 0.0 or e.std() == 0.0:
        return 0.0
    return float(np.corrcoef(s, e)[0, 1])


__all__ = [
    "vlsm_stat", "vlsm_binary", "svr_lsm_weights", "svr_lsm_binary",
    "dice", "dice_at_k", "focality", "focal_mass_mask", "age_exposure", "leakage",
]
