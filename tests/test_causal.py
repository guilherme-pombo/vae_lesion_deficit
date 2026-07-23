import numpy as np

from vld.causal import g_computation_ate, marginal_effect_map
from vld.models.conv_vae import fit_conv_vae


def _toy(seed: int = 0):
    """A substrate of the first 5 voxels drives a continuous deficit; map = the ideal indicator."""
    rng = np.random.default_rng(seed)
    v = 40
    X = (rng.random((200, v)) < 0.3).astype(np.float64)
    region = np.zeros(v, bool); region[:5] = True
    y = X[:, region].mean(1)                      # deficit driven by the substrate region
    substrate = region.astype(np.float64)          # ideal recovered map
    return X, y, substrate, region


def test_marginal_effect_map_units_and_localisation():
    X, y, substrate, region = _toy()
    eff = marginal_effect_map(substrate, X, y, continuous=True)
    assert eff.shape == (X.shape[1],)
    # calibrated to outcome units (nonzero scale) and concentrated on the true substrate
    assert eff[region].mean() > eff[~region].mean()
    assert np.any(eff != 0.0)


def test_g_computation_ate_localises():
    X, y, substrate, region = _toy()
    ctrl = np.zeros(X.shape[1], bool); ctrl[20:25] = True   # size-matched control, off the substrate
    ate_gt = g_computation_ate(substrate, X, y, region, continuous=True)
    ate_ctrl = g_computation_ate(substrate, X, y, ctrl, continuous=True)
    assert ate_gt > 0.0                       # lesioning the true substrate worsens the outcome
    assert ate_gt > ate_ctrl + 1e-6           # and far more than a control region (map~0 there)


def test_fit_conv_vae_identifiable_smoke():
    rng = np.random.default_rng(0)
    vol_shape = (8, 8, 8)
    v_cov = 40
    mask_idx = np.arange(v_cov)
    X = (rng.random((60, v_cov)) < 0.3).astype(np.float32)
    y = (X[:, 3] > 0).astype(np.float32)
    m = fit_conv_vae(X, y, mask_idx=mask_idx, vol_shape=vol_shape, identifiable=True,
                     epochs=2, num_down=2, batch=32, device="cpu")
    assert np.asarray(m).shape == (v_cov,)
    assert np.isfinite(np.asarray(m)).all()
