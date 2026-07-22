import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bagging import BaggedSubstrate, consensus_substrate, fit_conv_vae_bagged  # noqa: E402


def test_consensus_fraction_and_agreement():
    b1 = np.array([1, 1, 0, 0], bool)
    b2 = np.array([1, 0, 1, 0], bool)
    mask, frac = consensus_substrate([b1, b2], agreement=0.5)
    assert np.allclose(frac, [1.0, 0.5, 0.5, 0.0])
    assert mask.tolist() == [True, True, True, False]

    mask_unanimous, _ = consensus_substrate([b1, b2], agreement=1.0)
    assert mask_unanimous.tolist() == [True, False, False, False]


def test_agreement_monotonic():
    rng = np.random.default_rng(0)
    bins = [rng.random(200) > 0.5 for _ in range(8)]
    _, frac = consensus_substrate(bins)
    assert (frac >= 0.75).sum() <= (frac >= 0.25).sum()


def test_fit_conv_vae_bagged_smoke():
    rng = np.random.default_rng(0)
    vol_shape = (8, 8, 8)
    v_cov = 40
    mask_idx = np.arange(v_cov)  # first v_cov voxels of the flattened grid
    X = (rng.random((60, v_cov)) < 0.3).astype(np.float32)
    y = (X[:, 3] > 0).astype(np.float32)  # a voxel drives the deficit
    res = fit_conv_vae_bagged(X, y, mask_idx=mask_idx, vol_shape=vol_shape, n_bag=2,
                              epochs=1, batch=16, device="cpu")
    assert isinstance(res, BaggedSubstrate)
    assert res.consensus_mask.shape == (v_cov,)
    assert len(res.per_seed_binaries) == 2 and len(res.per_seed_maps) == 2
    assert res.consensus_fraction.min() >= 0.0 and res.consensus_fraction.max() <= 1.0
