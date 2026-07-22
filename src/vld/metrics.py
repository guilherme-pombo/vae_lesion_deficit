"""Scoring metrics + confound-leakage diagnostics, shared across all methods.

- ``dice`` / ``dice_at_k`` — recovery quality (deployment mask / top-k ranking).
- ``focal_mass_mask`` — the VAEs' calibration-implied self-threshold.
- ``age_exposure`` / ``leakage`` — how much a recovered map loads onto covariate-exposed
  off-target features (lower leakage = less confound bias).
"""

from __future__ import annotations

import numpy as np


def dice(pred_bool: np.ndarray, m_bool: np.ndarray) -> float:
    """Full binary Dice ``2·|pred ∩ M| / (|pred| + |M|)`` (0.0 when both empty)."""
    a = pred_bool.astype(bool)
    b = m_bool.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    total = float(a.sum() + b.sum())
    return 2.0 * inter / total if total > 0.0 else 0.0


def dice_at_k(score_map: np.ndarray, m_binary: np.ndarray, k: int) -> float:
    """Dice of the top-``k`` features of ``score_map`` vs ``m_binary`` (pure ranking quality)."""
    pred = np.zeros_like(m_binary, dtype=np.float64)
    pred[np.argsort(-score_map.astype(np.float64))[:k]] = 1.0
    inter = float((pred * m_binary).sum())
    denom = float(pred.sum() + m_binary.sum())
    return 2.0 * inter / denom if denom > 0 else 0.0


def focal_mass_mask(scores: np.ndarray) -> np.ndarray:
    """Top-``round(Σp)`` features of a calibrated probability/weight map (calibration-implied size).

    For a map calibrated so ``mean(p) == r``, ``Σp == r·K`` is the expected substrate size, so the
    top-``round(Σp)`` features are the self-sizing threshold — the VAEs' deployment mask.
    """
    p = scores.astype(np.float64)
    k = int(round(float(p.sum())))
    if k <= 0:
        return np.zeros(p.shape[0], dtype=bool)
    k = min(k, p.shape[0])
    mask = np.zeros(p.shape[0], dtype=bool)
    mask[np.argsort(-p)[:k]] = True
    return mask


def age_exposure(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-feature age-exposure ``e_k = corr_i(x[:, k], age)`` (age = ``cov[:, 0]``).

    High ``|e_k|`` flags features whose damage tracks the confounder.
    """
    age = np.asarray(cov, dtype=np.float64)[:, 0]
    age_c = age - age.mean()
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    num = xc.T @ age_c
    den = np.linalg.norm(xc, axis=0) * np.linalg.norm(age_c) + 1e-12
    return num / den


def leakage(score_map: np.ndarray, m_binary: np.ndarray, exposure: np.ndarray) -> float:
    """``corr(|score|, |exposure|)`` over NON-substrate features (0 if degenerate; lower = better)."""
    off = np.asarray(m_binary) <= 0.5
    s = np.abs(np.asarray(score_map, dtype=np.float64))[off]
    e = np.abs(np.asarray(exposure, dtype=np.float64))[off]
    if len(s) < 2 or s.std() == 0.0 or e.std() == 0.0:
        return 0.0
    return float(np.corrcoef(s, e)[0, 1])


__all__ = ["dice", "dice_at_k", "focal_mass_mask", "age_exposure", "leakage"]
