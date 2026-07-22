# aphasia_deficits.py — realistic-aphasia synthetic deficits (burden confound, non-linear, fuzzy label)
"""Realistic-aphasia synthetic deficits — a harder testbed than the clean substrate-overlap target.

The clean cov-v1 deficit is a (near-)linear function of substrate overlap, so a single fit recovers it
easily. Real aphasia is harder in ways that matter for debugging James's Vienna results:

* **Lesion-burden confound** — in acute left-MCA stroke, aphasia *presence* is driven heavily by total
  lesion volume, not just perisylvian damage. Deep periventricular WM is the region most reliably
  co-lesioned in large strokes, so a burden-dominated label makes it spuriously predictive — the prime
  suspect for the deep-WM binary peak. ``burden_weight`` dials this in.
* **Non-linear substrate response** — you must damage *enough* of the language network to become
  aphasic; the response saturates. ``nonlinearity="saturating"`` (default) models a thresholded,
  saturating map of overlap; ``"interaction"`` requires damaging two sub-regions (hub disconnection);
  ``"linear"`` recovers the easy case.
* **Fuzzy label + realistic prevalence** — presence is a *noisy* threshold of a graded latent at a
  left-hemi-stroke-like base rate, so the binary label is genuinely low-information vs the graded
  score (``label_noise``, ``prevalence``).

The graded latent is the severity target; the noisy threshold of it is the binary presence target. The
true substrate stays the cortical ``m`` (deep WM is a *confound*, not signal), so a map that lands on
periventricular WM is an artifact to be corrected — which is exactly what we want to test.
"""

from __future__ import annotations

import numpy as np


def _z(v: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array (guarded against zero variance)."""
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-8)


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def _substrate_signal(X: np.ndarray, m: np.ndarray, centroids: np.ndarray, *,
                      nonlinearity: str, gain: float, thresh_q: float) -> np.ndarray:
    """Per-patient substrate signal from lesion–substrate overlap, under the chosen non-linearity."""
    mb = m.astype(bool)
    overlap = X[:, mb].mean(1)
    if nonlinearity == "linear":
        return _z(overlap)
    if nonlinearity == "saturating":
        # need enough of the network damaged: thresholded, saturating (sigmoid) response.
        oz = _z(overlap)
        return _z(_sigmoid(gain * (oz - np.quantile(oz, thresh_q))))
    if nonlinearity == "interaction":
        # aphasia needs BOTH an anterior and a posterior hit (disconnection) — AND-like product.
        x = centroids[mb, 1]  # y-coord (anterior/posterior) of substrate voxels
        ant = mb.copy(); post = mb.copy()
        med = np.median(centroids[mb, 1])
        idx = np.flatnonzero(mb)
        ant[idx[centroids[mb, 1] >= med]] = False   # anterior sub-region
        post[idx[centroids[mb, 1] < med]] = False    # posterior sub-region
        oa, op = X[:, ant].mean(1), X[:, post].mean(1)
        return _z(_z(oa) * _z(op) if ant.any() and post.any() else overlap)
    raise ValueError(f"unknown nonlinearity {nonlinearity!r}")


def make_aphasia_like_deficits(
    X: np.ndarray,
    m: np.ndarray,
    centroids: np.ndarray,
    cov: np.ndarray,
    *,
    burden_weight: float = 2.0,
    substrate_weight: float = 1.0,
    nonlinearity: str = "saturating",
    gain: float = 4.0,
    thresh_q: float = 0.5,
    label_noise: float = 0.5,
    prevalence: float = 0.65,
    beta_main: float = 0.5,
    noise_sd: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build ``(y_binary, y_severity, info)`` — a realistic-aphasia deficit from a lesion matrix.

    ``X`` ``(N, V)`` coverage, ``m`` ``(V,)`` the true (cortical) substrate mask, ``centroids``
    ``(V, 3)`` mm coords, ``cov`` ``(N, d)`` covariates (``cov[:,0]`` = age, the volume-correlated
    confounder from the cov-v1 generator). The graded latent
    ``severity = substrate_weight·f(overlap) + burden_weight·z(volume) + beta_main·age + noise``
    (``f`` the chosen non-linearity); ``y_binary`` is a noisy logistic threshold of it at ``prevalence``.
    Raise ``burden_weight`` to make presence burden-dominated (the hard, James-like regime).

    Returns ``info`` with the latent, the burden vector, and the label's burden/age correlations.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    s = _substrate_signal(X, m, centroids, nonlinearity=nonlinearity, gain=gain, thresh_q=thresh_q)
    burden = _z(X.sum(1))                       # total lesion volume (the confound)
    age = np.asarray(cov[:, 0], dtype=np.float64)
    latent = (substrate_weight * s + burden_weight * burden + beta_main * age
              + noise_sd * rng.standard_normal(n))
    severity = latent.astype(np.float32)

    thr = float(np.quantile(latent, 1.0 - prevalence))
    temp = label_noise * latent.std() + 1e-6    # logistic temperature = noisiness of the threshold
    p = _sigmoid((latent - thr) / temp)
    y_binary = (rng.random(n) < p).astype(np.float32)

    info = {
        "prevalence": float(y_binary.mean()),
        "corr_label_burden": float(np.corrcoef(y_binary, burden)[0, 1]),
        "corr_label_age": float(np.corrcoef(y_binary, age)[0, 1]),
        "corr_sev_burden": float(np.corrcoef(severity, burden)[0, 1]),
        "burden": burden.astype(np.float32),
        "latent": latent.astype(np.float32),
    }
    return y_binary, severity, info


__all__ = ["make_aphasia_like_deficits"]
