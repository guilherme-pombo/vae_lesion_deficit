"""Covariate-conditioned deficit benchmark (cov-v1).

Extends the v3 substrate-recovery benchmark so the deficit depends on the
substrate AND on age. Age carries a MAIN effect (beta_main) and an INTERACTION
with the substrate load (beta_int), and is CONFOUNDED with total lesion volume by
`rho` (older patients tend to have larger lesions). The observed covariate matrix
is constructed here deterministically (fixed-seed generator) so the benchmark is
self-contained and the observed age is exactly the confounded age that drove Y.

Self-contained: the substrate propensity rules and the fixed rule tuple are inlined
from the v3 benchmark (`deficit_benchmarks.py`) so this module depends only on numpy.
"""

from __future__ import annotations

import numpy as np

# ── Inlined from the v3 deficit benchmark (deficit_benchmarks.py) ─────────────

BENCHMARK_RULES = ("coverage", "bool_OR", "hub_spoke", "bool_AND")
DEFAULT_TARGET_RATE = 0.12


def split2(cents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two balanced spatial clusters of substrate parcels.

    Median split on the centroid axis with the largest variance. Falls back to
    an index split if the median split is degenerate (all on one side).
    """
    ax = int(cents.var(axis=0).argmax())
    med = float(np.median(cents[:, ax]))
    a = cents[:, ax] <= med
    if a.all() or (~a).all():
        a = np.arange(len(cents)) < len(cents) // 2
    return a, ~a


def propensity(
    rule: str, Xm: np.ndarray, m_idx: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Continuous per-patient propensity for `rule` from substrate-overlap Xm (N, k)."""
    if rule == "coverage":
        return Xm.sum(axis=1)
    if rule in ("bool_OR", "bool_AND"):
        a, b = split2(centroids[m_idx])
        la, lb = Xm[:, a].sum(axis=1), Xm[:, b].sum(axis=1)
        return np.maximum(la, lb) if rule == "bool_OR" else la * lb
    if rule == "hub_spoke":
        hub = int(Xm.mean(axis=0).argmax())   # most-exposed parcel = hub
        spoke = np.ones(Xm.shape[1], dtype=bool)
        spoke[hub] = False
        hub_term = Xm[:, hub]
        spoke_term = (
            Xm[:, spoke].mean(axis=1) if spoke.any() else np.zeros(len(Xm), dtype=Xm.dtype)
        )
        return np.maximum(hub_term, spoke_term)
    raise ValueError(f"unknown deficit rule {rule!r}")


# ── Covariate benchmark (cov-v1) ──────────────────────────────────────────────

BENCHMARK_COV_VERSION = "cov-v1"


def _zscore(v: np.ndarray) -> np.ndarray:
    """Z-score a vector; returns zeros if the input has no variance."""
    v = np.asarray(v, dtype=np.float64)
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 0.0 else np.zeros_like(v)


def topk_ranked(score: np.ndarray, n_pos: int) -> np.ndarray:
    """Y=1 for the `n_pos` highest-scoring patients by RANK (negatives allowed).

    Unlike `deficit_benchmarks.topk_label`, there is no `score>0` restriction: a
    covariate-driven positive may have zero substrate overlap.
    """
    n = int(min(max(n_pos, 0), len(score)))
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")[:n]
    y = np.zeros(len(score), dtype=np.float32)
    y[order] = 1.0
    return y


def build_covariates(
    X_parcel: np.ndarray,
    *,
    n_covariates: int = 2,
    rho: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Construct the observed `(N, n_covariates)` covariate matrix.

    * col 0 = **age**: `z(rho * z(total_load) + sqrt(1-rho^2) * noise)` so
      `corr(age, load) ~= rho` (the canonical lesion-volume confound via age).
    * col 1 = **sex**: fixed-seed Bernoulli(0.5), inert (no deficit effect).
    * cols >= 2 = additional inert N(0,1) nuisances.

    Deterministic: a fixed seed fully determines the output.
    """
    rng = np.random.default_rng(seed)
    n = X_parcel.shape[0]
    load_z = _zscore(X_parcel.sum(axis=1))
    noise = rng.standard_normal(n)
    rho = float(np.clip(rho, -1.0, 1.0))
    age = _zscore(rho * load_z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * noise)
    cols = [age]
    if n_covariates >= 2:
        cols.append((rng.random(n) < 0.5).astype(np.float64))
    for _ in range(max(n_covariates - 2, 0)):
        cols.append(rng.standard_normal(n))
    return np.column_stack(cols[:n_covariates]).astype(np.float32)


def make_covariate_benchmark_deficits(
    X: np.ndarray,
    M_binary: np.ndarray,
    centroids: np.ndarray,
    *,
    K_parcel: int,
    n_covariates: int = 2,
    beta_main: float = 0.5,
    beta_int: float = 0.5,
    rho: float = 0.5,
    target_rate: float = DEFAULT_TARGET_RATE,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ({rule: Y (N,) float32}, cov (N, n_covariates) float32).

    severity_i = g_i * clip(1 + beta_int*age_i, 0, None) + beta_main*age_i, where
    g_i is the z-scored per-rule substrate propensity. Y selects the top
    `round(target_rate*N)` patients by severity (positive rate held fixed).
    """
    X_parcel = X[:, :K_parcel]
    m_idx = np.where(M_binary[:K_parcel] > 0.5)[0]
    Xm = X_parcel[:, m_idx]
    cov = build_covariates(X_parcel, n_covariates=n_covariates, rho=rho, seed=seed)
    age = cov[:, 0].astype(np.float64)
    mult = np.clip(1.0 + beta_int * age, 0.0, None)
    n_pos = int(round(target_rate * X.shape[0]))

    deficits: dict[str, np.ndarray] = {}
    for rule in BENCHMARK_RULES:
        g = _zscore(propensity(rule, Xm, m_idx, centroids))
        severity = g * mult + beta_main * age
        deficits[rule] = topk_ranked(severity, n_pos)
    return deficits, cov


__all__ = [
    "BENCHMARK_COV_VERSION",
    "BENCHMARK_RULES",
    "DEFAULT_TARGET_RATE",
    "split2",
    "propensity",
    "topk_ranked",
    "build_covariates",
    "make_covariate_benchmark_deficits",
]
