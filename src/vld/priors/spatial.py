"""Spatially-coherent substrate estimation (Laplacian-penalised / graph-Tikhonov regression).

Lesion-deficit mapping is an ill-posed linear inverse problem: with N patients and K ≫ N voxels the data
alone cannot pin the substrate, so the penalty decides the answer. Ordinary ridge penalises ``‖w‖²``,
which treats every voxel as independent of its neighbours — an obviously wrong description of a brain,
and the reason ridge maps come out speckled. Penalising *roughness* instead,

    min_w  ‖y − Xw‖²  +  λ · wᵀ(I + β L) w

with ``L`` the graph Laplacian over the voxel adjacency graph, says "neighbouring voxels should have
similar weights". ``wᵀLw`` is the sum of squared differences across adjacent voxel pairs, so minimising it
favours spatially coherent maps. ``β = 0`` recovers ordinary ridge.

This is the penalty family behind LORETA (Pascual-Marqui et al. 1994) in EEG/MEG source localisation,
which faces the same underdetermination — far more candidate sources than sensors. The name is used here
only as precedent; the estimator itself is plain Laplacian-penalised regression.

Solved in the dual via Woodbury, so cost scales with N rather than K::

    w = M⁻¹Xᵀ (I + X M⁻¹Xᵀ)⁻¹ y ,   M = λ(I + βL)

:class:`LaplacianInverse` precomputes ``M⁻¹Xᵀ`` once. Because column *i* of that matrix is ``M⁻¹x_i``,
any row subset of ``X`` is just a column slice — so cross-validation folds, split halves and bootstrap
resamples all reuse the same factorisation instead of refactorising per fit.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, identity, spmatrix
from scipy.sparse.linalg import factorized

_NEIGHBOUR_OFFSETS: dict[int, tuple[tuple[int, int, int], ...]] = {
    6: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    26: tuple((dz, dy, dx)
              for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dz, dy, dx) > (0, 0, 0)),
}


def voxel_adjacency(mask_idx: np.ndarray, vol_shape: tuple[int, int, int],
                    connectivity: int = 6) -> csr_matrix:
    """Symmetric ``(K, K)`` adjacency of the coverage voxels on the image grid.

    Parameters
    ----------
    mask_idx:
        ``(K,)`` flat indices of the coverage voxels into a ``vol_shape`` volume — the same index array
        used to scatter a ``(N, K)`` lesion matrix back into image space.
    vol_shape:
        ``(D, H, W)`` shape of that volume.
    connectivity:
        ``6`` (face-adjacent, the default) or ``26`` (face + edge + corner).

    Returns
    -------
    adjacency:
        Sparse 0/1 matrix; entry ``(a, b)`` is 1 when coverage voxels ``a`` and ``b`` are neighbours.
    """
    if connectivity not in _NEIGHBOUR_OFFSETS:
        raise ValueError(f"connectivity must be 6 or 26 (got {connectivity})")
    coords = np.array(np.unravel_index(np.asarray(mask_idx), vol_shape)).T
    lookup = {tuple(int(c) for c in row): i for i, row in enumerate(coords)}
    rows: list[int] = []
    cols: list[int] = []
    for i, row in enumerate(coords):
        for offset in _NEIGHBOUR_OFFSETS[connectivity]:
            key = (int(row[0]) + offset[0], int(row[1]) + offset[1], int(row[2]) + offset[2])
            j = lookup.get(key)
            if j is not None:
                rows += [i, j]
                cols += [j, i]
    n = len(coords)
    if not rows:
        return csr_matrix((n, n), dtype=np.float64)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    adjacency.data[:] = 1.0
    return adjacency


def voxel_laplacian(mask_idx: np.ndarray, vol_shape: tuple[int, int, int],
                    connectivity: int = 6) -> csr_matrix:
    """Graph Laplacian ``D − A`` over the coverage voxels (see :func:`voxel_adjacency`).

    ``wᵀLw`` equals the sum of squared differences between adjacent voxel weights, so it measures the
    roughness of a substrate map.
    """
    adjacency = voxel_adjacency(mask_idx, vol_shape, connectivity)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return csr_matrix(diags(degree) - adjacency)


def _partial_out(values: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    """Residualise columns of ``values`` on ``[1, cov]`` (no-op when ``cov`` is None)."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    design = (np.ones((n, 1)) if cov is None
              else np.column_stack([np.ones(n), np.asarray(cov, dtype=np.float64)]))
    basis, _ = np.linalg.qr(design)
    return arr - basis @ (basis.T @ arr)


class LaplacianInverse:
    """Reusable Laplacian-penalised solver for one lesion matrix and one penalty setting.

    Build it once, then call :meth:`fit` for as many deficit vectors and patient subsets as you like —
    the expensive sparse factorisation and the ``M⁻¹Xᵀ`` product are computed a single time.

    Parameters
    ----------
    X:
        ``(N, K)`` lesion matrix (rows = patients, columns = coverage voxels).
    mask_idx, vol_shape:
        Coverage-voxel indices and volume shape, used to build the adjacency graph.
    beta:
        Smoothness weight. ``0`` gives ordinary ridge; larger values enforce more spatial coherence.
        ``100`` was optimal across every condition of the planted-substrate benchmark in
        ``docs/SPATIAL_PRIOR.md`` and is a reasonable default.
    lam:
        Overall penalty scale, in units of the mean gram eigenvalue ``trace(XXᵀ)/N`` — so it is
        comparable across cohorts and voxel-grid sizes.
    cov:
        Optional ``(N, C)`` nuisance covariates. Both ``X`` and the deficit are residualised on
        ``[1, cov]``, matching the partialing-out convention used by the VLSM/SVR baselines.
    connectivity:
        Voxel adjacency, ``6`` or ``26``.

    Examples
    --------
    >>> solver = LaplacianInverse(X, mask_idx=mask_idx, vol_shape=shape, beta=100.0)  # doctest: +SKIP
    >>> substrate = solver.fit(y)                                                     # doctest: +SKIP
    >>> half = solver.fit(y[idx], rows=idx)   # reuses the factorisation               # doctest: +SKIP
    """

    def __init__(self, X: np.ndarray, *, mask_idx: np.ndarray, vol_shape: tuple[int, int, int],
                 beta: float = 100.0, lam: float = 0.5, cov: np.ndarray | None = None,
                 connectivity: int = 6) -> None:
        x64 = np.asarray(X, dtype=np.float64)
        if x64.ndim != 2:
            raise ValueError(f"X must be 2-D (N, K); got shape {x64.shape}")
        if x64.shape[1] != len(np.asarray(mask_idx)):
            raise ValueError(f"X has {x64.shape[1]} columns but mask_idx has "
                             f"{len(np.asarray(mask_idx))} entries")
        if beta < 0.0 or lam <= 0.0:
            raise ValueError("beta must be >= 0 and lam must be > 0")
        self.n_patients, self.n_voxels = x64.shape
        self.beta = float(beta)
        self.lam = float(lam)
        self._cov = None if cov is None else np.asarray(cov, dtype=np.float64)
        self.X = _partial_out(x64, self._cov)
        scale = float(np.trace(self.X @ self.X.T)) / self.n_patients
        penalty: spmatrix = identity(self.n_voxels, format="csc")
        if self.beta > 0.0:
            penalty = penalty + self.beta * voxel_laplacian(mask_idx, vol_shape, connectivity)
        solve = factorized(csc_matrix(self.lam * scale * penalty))
        # column i is M^-1 x_i, so any row subset of X is a column slice of this matrix
        self._m_inv_xt = np.column_stack([solve(self.X[i]) for i in range(self.n_patients)])

    def fit(self, y: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Return the ``(K,)`` substrate map for deficit ``y``, optionally on a subset of patients.

        Parameters
        ----------
        y:
            Deficit vector. Length must match ``rows`` if given, else the full patient count. It is
            residualised on the covariates supplied at construction.
        rows:
            Optional patient indices (a CV fold, split half, or bootstrap resample — repeats allowed).
        """
        idx = np.arange(self.n_patients) if rows is None else np.asarray(rows, dtype=int)
        y64 = _partial_out(np.asarray(y, dtype=np.float64).reshape(-1, 1),
                           None if self._cov is None else self._cov[idx]).ravel()
        if len(y64) != len(idx):
            raise ValueError(f"y has {len(y64)} entries but rows selects {len(idx)} patients")
        block = self._m_inv_xt[:, idx]
        gram = self.X[idx] @ block
        return block @ np.linalg.solve(np.eye(len(idx)) + gram, y64)


def fit_laplacian_substrate(X: np.ndarray, y: np.ndarray, *, mask_idx: np.ndarray,
                            vol_shape: tuple[int, int, int], beta: float = 100.0,
                            lam: float = 0.5, cov: np.ndarray | None = None,
                            connectivity: int = 6) -> np.ndarray:
    """One-shot Laplacian-penalised substrate map — see :class:`LaplacianInverse` for the parameters.

    Use the class directly when fitting more than once on the same ``X`` (sweeps, folds, bootstraps);
    this wrapper rebuilds the factorisation on every call.

    Returns
    -------
    substrate:
        ``(K,)`` weight map over the coverage voxels. Positive weights mean "lesion here predicts a
        larger ``y``"; flip the sign convention of ``y`` if you want damage to score positive.
    """
    solver = LaplacianInverse(X, mask_idx=mask_idx, vol_shape=vol_shape, beta=beta, lam=lam,
                              cov=cov, connectivity=connectivity)
    return solver.fit(y)
