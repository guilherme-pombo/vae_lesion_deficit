"""Units-ful causal readout for a recovered substrate map.

The VAE substrate ``E[σ(γ)·w]`` is a predictive *attribution* on an arbitrary scale. These helpers turn
it into interpretable, outcome-unit quantities via a plug-in deficit head fitted on the substrate
overlap ``f_i = mean(X_i · map)`` (plus optional covariates):

  * :func:`marginal_effect_map` — the per-voxel marginal effect ``scale · map`` (Δ outcome per unit
    lesion coverage), a calibrated effect-size map rather than a mass-scaled, unit-less one;
  * :func:`g_computation_ate` — the plug-in g-computation average treatment effect of lesioning a
    region ``R``: ``ATE_R = mean_i[ ŷ(X_i | R:=1) − ŷ(X_i | R:=0) ]`` in outcome units.

The head adjusts for measured covariates, so the ATE is confounder-adjusted for those. It is NOT a
license for a voxel-level *causal* claim on observational lesion data: co-lesioned voxels violate
positivity (shared vascular territory), so estimate at a positivity-respecting granularity
(regions/tracts) and pair with a sensitivity analysis (e.g. E-values) before any causal reading.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def _overlap(substrate_map: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-patient substrate overlap ``f_i = mean_v(X_iv · map_v)`` with the clamped (≥0) map."""
    m = np.clip(np.asarray(substrate_map, np.float64), 0.0, None)
    return (np.asarray(X, np.float64) * m).mean(axis=1)


def _fit_head(f: np.ndarray, y: np.ndarray, cov: np.ndarray | None, continuous: bool):
    """Fit the deficit head ``y ~ [f, cov, cov·f]`` (least-squares / logistic). Returns a callable
    mapping a new overlap ``f_new`` (and covariates) to the predicted outcome in its natural units
    (continuous: the score; binary: probability), plus the fitted overlap coefficient ``scale``."""
    f = np.asarray(f, np.float64).reshape(-1, 1)
    if cov is None:
        design = f
    else:
        cov = np.asarray(cov, np.float64)
        design = np.concatenate([f, cov, cov * f], axis=1)
    reg = (LinearRegression() if continuous else LogisticRegression(max_iter=1000)).fit(design, y)
    scale = float(reg.coef_.ravel()[0])

    def predict(f_new: np.ndarray, cov_new: np.ndarray | None = None) -> np.ndarray:
        fn = np.asarray(f_new, np.float64).reshape(-1, 1)
        d = fn if cov_new is None else np.concatenate([fn, np.asarray(cov_new, np.float64),
                                                       np.asarray(cov_new, np.float64) * fn], axis=1)
        return reg.predict(d) if continuous else reg.predict_proba(d)[:, 1]

    return predict, scale


def marginal_effect_map(substrate_map: np.ndarray, X: np.ndarray, y: np.ndarray, *,
                        cov: np.ndarray | None = None, continuous: bool = False) -> np.ndarray:
    """Units-ful per-voxel marginal effect ``scale · clip(map, 0)`` (Δ outcome per unit coverage),
    where ``scale`` is the fitted overlap coefficient. Same ranking as the raw map, but calibrated to
    outcome units instead of mass-scaled to an arbitrary size."""
    _, scale = _fit_head(_overlap(substrate_map, X), np.asarray(y, np.float64), cov, continuous)
    return scale * np.clip(np.asarray(substrate_map, np.float64), 0.0, None)


def g_computation_ate(substrate_map: np.ndarray, X: np.ndarray, y: np.ndarray, region: np.ndarray, *,
                      cov: np.ndarray | None = None, continuous: bool = False) -> float:
    """Plug-in g-computation ATE of lesioning ``region`` (bool mask over voxels):
    ``mean_i[ ŷ(X_i | region:=1) − ŷ(X_i | region:=0) ]`` in outcome units. Positive ⇒ lesioning the
    region worsens the predicted outcome. Covariates (if given) are held at their observed values."""
    X = np.asarray(X, np.float64)
    region = np.asarray(region, bool)
    predict, _ = _fit_head(_overlap(substrate_map, X), np.asarray(y, np.float64), cov, continuous)
    x1 = X.copy(); x1[:, region] = 1.0
    x0 = X.copy(); x0[:, region] = 0.0
    return float(np.mean(predict(_overlap(substrate_map, x1), cov)
                         - predict(_overlap(substrate_map, x0), cov)))
