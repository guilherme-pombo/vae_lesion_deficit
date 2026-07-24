import numpy as np
import pytest

from vld.metrics import morans_i, support_auc
from vld.priors import LaplacianInverse, fit_laplacian_substrate, voxel_adjacency, voxel_laplacian

SHAPE = (6, 6, 6)


def _grid_cohort(n=60, seed=0):
    """Small cube of voxels with contiguous blob lesions, so spatial structure is real."""
    mask_idx = np.arange(int(np.prod(SHAPE)))
    coords = np.array(np.unravel_index(mask_idx, SHAPE)).T
    rng = np.random.default_rng(seed)
    centres = coords[rng.integers(0, len(coords), n)]
    dist = np.linalg.norm(coords[None, :, :] - centres[:, None, :], axis=2)
    return (dist <= 1.6).astype(np.float64), mask_idx, coords


def test_adjacency_is_symmetric_and_binary():
    adj = voxel_adjacency(np.arange(int(np.prod(SHAPE))), SHAPE)
    assert (adj != adj.T).nnz == 0
    assert set(np.unique(adj.data)) <= {1.0}
    # interior voxel of a 6-connected grid has exactly 6 neighbours
    degree = np.asarray(adj.sum(axis=1)).ravel().reshape(SHAPE)
    assert degree[2, 2, 2] == 6
    assert degree[0, 0, 0] == 3  # corner


def test_laplacian_quadratic_form_measures_roughness():
    mask_idx = np.arange(int(np.prod(SHAPE)))
    lap = voxel_laplacian(mask_idx, SHAPE)
    constant = np.ones(len(mask_idx))
    assert float(constant @ (lap @ constant)) == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    rough = rng.standard_normal(len(mask_idx))
    smooth = np.linspace(0.0, 1.0, len(mask_idx))
    assert float(rough @ (lap @ rough)) > float(smooth @ (lap @ smooth))


def test_beta_zero_matches_plain_ridge():
    X, mask_idx, _ = _grid_cohort()
    y = X[:, 5] + 0.1 * np.random.default_rng(1).standard_normal(len(X))
    w = fit_laplacian_substrate(X, y, mask_idx=mask_idx, vol_shape=SHAPE, beta=0.0, lam=0.5)
    xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    ridge_lam = 0.5 * float(np.trace(xc @ xc.T)) / len(X)
    expected = xc.T @ np.linalg.solve(xc @ xc.T + ridge_lam * np.eye(len(X)), yc)
    assert np.corrcoef(w, expected)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_row_subset_matches_a_solver_built_on_that_subset():
    """The precompute must be an exact shortcut, not an approximation."""
    X, mask_idx, _ = _grid_cohort()
    y = X[:, 7] + 0.2 * np.random.default_rng(2).standard_normal(len(X))
    rows = np.arange(0, len(X), 2)
    shared = LaplacianInverse(X, mask_idx=mask_idx, vol_shape=SHAPE, beta=20.0)
    subset = LaplacianInverse(X[rows], mask_idx=mask_idx, vol_shape=SHAPE, beta=20.0)
    a = shared.fit(y[rows], rows=rows)
    b = subset.fit(y[rows])
    # the two differ only through the lam scale, which is set from the full vs subset gram
    assert np.corrcoef(a, b)[0, 1] > 0.99


def test_smoothing_increases_spatial_autocorrelation():
    X, mask_idx, _ = _grid_cohort()
    y = X[:, 9] + 0.3 * np.random.default_rng(3).standard_normal(len(X))
    adj = voxel_adjacency(mask_idx, SHAPE)
    rough = fit_laplacian_substrate(X, y, mask_idx=mask_idx, vol_shape=SHAPE, beta=0.0)
    smooth = fit_laplacian_substrate(X, y, mask_idx=mask_idx, vol_shape=SHAPE, beta=100.0)
    assert morans_i(smooth, adj) > morans_i(rough, adj)


def test_recovers_a_planted_blob_better_than_ridge_when_ill_posed():
    """The prior earns its keep only when voxels far outnumber patients (K >> N).

    That is the real lesion-mapping regime. On a small grid where N is comparable to K the problem is
    already well posed, ridge is near-perfect, and smoothing only costs — see
    ``test_prior_is_not_needed_when_well_posed``.
    """
    big = (10, 10, 10)
    mask_idx = np.arange(int(np.prod(big)))
    coords = np.array(np.unravel_index(mask_idx, big)).T
    rng = np.random.default_rng(5)
    centres = coords[rng.integers(0, len(coords), 40)]
    dist = np.linalg.norm(coords[None, :, :] - centres[:, None, :], axis=2)
    X = (dist <= 2.2).astype(np.float64)          # N=40 patients, K=1000 voxels
    truth = (np.linalg.norm(coords - np.array([4, 4, 4]), axis=1) <= 2.0).astype(np.float64)
    y = X @ truth + 2.0 * rng.standard_normal(len(X))
    kw = dict(mask_idx=mask_idx, vol_shape=big)
    ridge = fit_laplacian_substrate(X, y, beta=0.0, **kw)
    lap = fit_laplacian_substrate(X, y, beta=50.0, **kw)
    assert support_auc(lap, truth) > support_auc(ridge, truth)


def test_prior_is_not_needed_when_well_posed():
    """Documents the boundary: with N comparable to K, plain ridge already recovers the blob."""
    X, mask_idx, coords = _grid_cohort(n=120, seed=5)
    truth = (np.linalg.norm(coords - np.array([2, 2, 2]), axis=1) <= 1.5).astype(np.float64)
    y = X @ truth + 2.0 * np.random.default_rng(6).standard_normal(len(X))
    ridge = fit_laplacian_substrate(X, y, mask_idx=mask_idx, vol_shape=SHAPE, beta=0.0)
    assert support_auc(ridge, truth) > 0.95


def test_support_auc_and_morans_i_edge_cases():
    assert support_auc(np.array([1.0, 2.0, 3.0]), np.zeros(3)) == 0.5
    assert support_auc(np.array([0.0, 1.0]), np.array([0, 1])) == 1.0
    assert support_auc(np.array([1.0, 0.0]), np.array([0, 1])) == 0.0
    assert support_auc(np.ones(4), np.array([0, 0, 1, 1])) == 0.5  # all ties
    adj = voxel_adjacency(np.arange(int(np.prod(SHAPE))), SHAPE)
    assert np.isnan(morans_i(np.ones(int(np.prod(SHAPE))), adj))


def test_rejects_bad_arguments():
    X, mask_idx, _ = _grid_cohort()
    with pytest.raises(ValueError):
        LaplacianInverse(X, mask_idx=mask_idx[:-1], vol_shape=SHAPE)
    with pytest.raises(ValueError):
        LaplacianInverse(X, mask_idx=mask_idx, vol_shape=SHAPE, lam=0.0)
    with pytest.raises(ValueError):
        voxel_adjacency(mask_idx, SHAPE, connectivity=18)
