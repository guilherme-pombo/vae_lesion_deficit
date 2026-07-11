"""Covariate-conditioned deficit generation for lesion-deficit substrate recovery.

The simulated deficit depends on the substrate AND on a covariate (age) that both
modulates deficit severity (main effect + interaction with substrate load) and is
*confounded* with lesion volume (older patients tend to have larger lesions -- the
canonical lesion-symptom-mapping confound). Fully deterministic given ``seed``.

Everything operates on flattened in-brain parcel vectors ``X`` of shape ``(N, K)`` (K =
number of parcels) and a binary substrate ``M`` of shape ``(K,)`` -- so the same
API drives parcels here and voxels elsewhere.
"""

from __future__ import annotations

import numpy as np

RULES = ("coverage", "bool_OR", "hub_spoke", "bool_AND")
DEFAULT_TARGET_RATE = 0.12


def _zscore(v: np.ndarray) -> np.ndarray:
    """Z-score a vector; returns zeros if the input has no variance."""
    v = np.asarray(v, dtype=np.float64)
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 0.0 else np.zeros_like(v)


def split2(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split substrate parcels into two spatial clusters (median split on the widest axis).

    ``coords`` is ``(m, 3)`` parcel centroids of the substrate parcels. Falls back to an
    index split if the median split is degenerate.
    """
    ax = int(coords.var(axis=0).argmax())
    med = float(np.median(coords[:, ax]))
    a = coords[:, ax] <= med
    if a.all() or (~a).all():
        a = np.arange(len(coords)) < len(coords) // 2
    return a, ~a


def propensity(rule: str, Xm: np.ndarray, coords_m: np.ndarray) -> np.ndarray:
    """Continuous per-patient substrate propensity for ``rule`` from overlap ``Xm (N, m)``.

    ``Xm`` is each patient's lesion overlap with the substrate parcels; ``coords_m`` the
    ``(m, 3)`` coordinates of those voxels (used by the spatially-split rules).
    """
    if rule == "coverage":
        return Xm.sum(axis=1)
    if rule in ("bool_OR", "bool_AND"):
        a, b = split2(coords_m)
        la, lb = Xm[:, a].sum(axis=1), Xm[:, b].sum(axis=1)
        return np.maximum(la, lb) if rule == "bool_OR" else la * lb
    if rule == "hub_spoke":
        hub = int(Xm.mean(axis=0).argmax())  # most-exposed parcel = hub
        spoke = np.ones(Xm.shape[1], dtype=bool)
        spoke[hub] = False
        hub_term = Xm[:, hub]
        spoke_term = (
            Xm[:, spoke].mean(axis=1) if spoke.any() else np.zeros(len(Xm), dtype=Xm.dtype)
        )
        return np.maximum(hub_term, spoke_term)
    raise ValueError(f"unknown deficit rule {rule!r}")


def topk_ranked(score: np.ndarray, n_pos: int) -> np.ndarray:
    """y=1 for the ``n_pos`` highest-scoring patients by rank (negatives allowed).

    Unlike a ``score>0`` gate, a covariate-driven positive may have zero substrate
    overlap -- exactly the confounding case we want to allow.
    """
    n = int(min(max(n_pos, 0), len(score)))
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")[:n]
    y = np.zeros(len(score), dtype=np.float32)
    y[order] = 1.0
    return y


def build_covariates(
    X: np.ndarray, *, n_covariates: int = 2, rho: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Construct the observed ``(N, n_covariates)`` covariate matrix (deterministic).

    * col 0 = **age**: ``z(rho * z(lesion_volume) + sqrt(1-rho^2) * noise)``, so
      ``corr(age, volume) ~= rho`` -- the lesion-volume confound expressed through age.
      ``lesion_volume`` is each patient's total lesioned voxel count ``X.sum(1)``.
    * col 1 = **sex**: fixed-seed Bernoulli(0.5), inert (no deficit effect).
    * cols >= 2 = additional inert N(0,1) nuisances.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vol_z = _zscore(np.asarray(X).sum(axis=1))
    noise = rng.standard_normal(n)
    rho = float(np.clip(rho, -1.0, 1.0))
    age = _zscore(rho * vol_z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * noise)
    cols = [age]
    if n_covariates >= 2:
        cols.append((rng.random(n) < 0.5).astype(np.float64))
    for _ in range(max(n_covariates - 2, 0)):
        cols.append(rng.standard_normal(n))
    return np.column_stack(cols[:n_covariates]).astype(np.float32)


def make_covariate_deficits(
    X: np.ndarray,
    M: np.ndarray,
    coords: np.ndarray,
    *,
    n_covariates: int = 2,
    beta_main: float = 0.5,
    beta_int: float = 0.5,
    rho: float = 0.5,
    target_rate: float = DEFAULT_TARGET_RATE,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ``({rule: y (N,) float32}, cov (N, n_covariates) float32)``.

    For each rule, ``severity = g * clip(1 + beta_int*age, 0, None) + beta_main*age`` where
    ``g`` is the z-scored substrate propensity; ``y`` selects the top ``round(target_rate*N)``
    patients by severity (positive rate held fixed).

    Parameters
    ----------
    X : ``(N, K)`` parcel lesion-overlap matrix.
    M : ``(K,)`` binary substrate mask (parcels).
    coords : ``(K, 3)`` parcel centroids for the parcels.
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M)
    m_idx = np.where(M > 0.5)[0]
    Xm = X[:, m_idx]
    coords_m = np.asarray(coords)[m_idx]
    cov = build_covariates(X, n_covariates=n_covariates, rho=rho, seed=seed)
    age = cov[:, 0].astype(np.float64)
    mult = np.clip(1.0 + beta_int * age, 0.0, None)
    n_pos = int(round(target_rate * X.shape[0]))

    deficits: dict[str, np.ndarray] = {}
    for rule in RULES:
        g = _zscore(propensity(rule, Xm, coords_m))
        severity = g * mult + beta_main * age
        deficits[rule] = topk_ranked(severity, n_pos)
    return deficits, cov


__all__ = [
    "RULES",
    "DEFAULT_TARGET_RATE",
    "split2",
    "propensity",
    "topk_ranked",
    "build_covariates",
    "make_covariate_deficits",
]
