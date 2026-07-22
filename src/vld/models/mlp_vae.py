# mlp_vae.py — two-arm (MLP) lesion-deficit VAE
"""Parcel-space substrate VAE baseline (two-arm).

Faithful parcel-space port of the `vae_lesion_deficit` VAE (Pombo et al.), used
as a per-episode-fit baseline alongside VLSM and SVR-LSM. Two per-parcel arms
decouple the roles that a single map conflates: a MEMBERSHIP arm ``p = sigma(gamma_p)``
(which parcels are in the substrate) and an IMPORTANCE arm ``w`` (how much damage
there drives the deficit). The deficit is predicted from the gated overlap
``mean(X * p * w)``; a lesion-reconstruction arm regularises the latent. The
recovered substrate is the cohort-averaged membership map ``E[p]``.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ParcelLesionDeficitVAE(nn.Module):
    """Two-arm parcel VAE: membership ``p=sigma(gamma_p)`` + importance ``w``.

    The 2-channel inference decoder emits ``gamma_p`` (ch0) and ``w`` (ch1) per
    parcel; the deficit is predicted from ``mean(X * sigma(gamma_p) * w)``.
    ``substrate_map`` reads out the cohort-averaged membership probability ``E[p]``.

    With ``d_cov > 0`` the deficit head gains a covariate pathway (global learnable
    ``beta`` main effect + ``gamma`` interaction): ``logit = overlap_scale *
    overlap * (1 + cov @ gamma) + cov @ beta + overlap_bias``. Covariates enter the
    deficit head ONLY, never the mask-latent encoder, so the ``E[p*w]`` substrate
    readout stays covariate-free (the deficit is recovered controlling for the
    covariates). ``d_cov == 0`` reduces exactly to the covariate-free head.
    """

    def __init__(self, K: int, *, d_cov: int = 0, z_dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.K = K
        self.half = z_dim // 2
        self.enc = nn.Sequential(nn.Linear(K + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        # 2-channel inference decoder: ch0 = membership logit gamma_p, ch1 = importance w
        self.dec = nn.Sequential(nn.Linear(self.half, hidden), nn.GELU(),
                                 nn.Linear(hidden, K * 2))
        self.rdec = nn.Sequential(nn.Linear(self.half, hidden), nn.GELU(),
                                  nn.Linear(hidden, K))
        self.overlap_scale = nn.Parameter(torch.tensor(5.0))
        self.overlap_bias = nn.Parameter(torch.tensor(0.0))
        self.d_cov = int(d_cov)
        if self.d_cov > 0:
            self.beta = nn.Parameter(torch.zeros(self.d_cov))   # covariate main effect
            self.gamma = nn.Parameter(torch.zeros(self.d_cov))  # covariate x substrate interaction

    def _arms(self, mask_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.dec(mask_z).view(-1, self.K, 2)
        return torch.sigmoid(out[:, :, 0]), out[:, :, 1]        # p (membership), w (importance)

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                pos_weight: torch.Tensor | None,
                cov: torch.Tensor | None = None,
                continuous: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, deficit_loss)`` for one full-batch step.

        ``continuous=False`` (default) treats ``Y`` as a binary deficit (BCE head); ``True`` treats
        it as a continuous score, e.g. symptom severity (MSE head on the standardized target). The
        substrate readout ``E[p·w]`` is identical either way — only the deficit likelihood changes.
        """
        h = self.enc(torch.cat([X, Y.view(-1, 1)], dim=1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        mask_z, recon_z = z[:, :self.half], z[:, self.half:]
        kl = torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1)
        p, w = self._arms(mask_z)
        recon = torch.sigmoid(self.rdec(recon_z))
        overlap = (X * p * w).mean(dim=1, keepdim=True)
        if cov is None or self.d_cov == 0:
            pred = self.overlap_scale * overlap + self.overlap_bias
        else:
            scale_i = self.overlap_scale * (1.0 + cov @ self.gamma).view(-1, 1)
            pred = scale_i * overlap + (cov @ self.beta).view(-1, 1) + self.overlap_bias
        if continuous:
            mask_ll = F.mse_loss(pred, Y.view(-1, 1))
        else:
            mask_ll = F.binary_cross_entropy_with_logits(pred, Y.view(-1, 1), pos_weight=pos_weight)
        recon_ll = F.binary_cross_entropy(recon, X, reduction="none").sum(dim=1).mean()
        return mask_ll + recon_ll + kl.mean(), mask_ll

    @torch.no_grad()
    def substrate_map(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """``(K,)`` cohort-averaged membership probability ``E[p]`` in [0, 1]."""
        mask_z = self.mu(self.enc(torch.cat([X, Y.view(-1, 1)], dim=1)))[:, :self.half]
        p, _ = self._arms(mask_z)
        return p.mean(dim=0)

    @torch.no_grad()
    def logit_map(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """``(K,)`` cohort-averaged membership LOGIT ``E[gamma_p]`` (uncentered)."""
        mask_z = self.mu(self.enc(torch.cat([X, Y.view(-1, 1)], dim=1)))[:, :self.half]
        out = self.dec(mask_z).view(-1, self.K, 2)
        return out[:, :, 0].mean(dim=0)

    @torch.no_grad()
    def effective_map(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """``(K,)`` cohort-averaged EFFECTIVE per-parcel weight ``E[sigma(gamma_p)*w]``.

        This is the per-parcel coefficient the deficit head actually uses. The
        bare membership arm ``sigma(gamma_p)`` saturates toward 1 and is
        non-identifiable (the two arms multiply), so the substrate signal lives
        in the product ``p*w``, not in ``p`` alone. Ranking by this recovers the
        substrate robustly across optimisers; ``substrate_map`` (E[p]) does not.
        See VAE_EXPERIMENTS_LOGS.md Rounds 8-10.
        """
        mask_z = self.mu(self.enc(torch.cat([X, Y.view(-1, 1)], dim=1)))[:, :self.half]
        p, w = self._arms(mask_z)
        return (p * w).mean(dim=0)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically-stable elementwise logistic sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _bce_with_logits_np(z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy of logits ``z`` against labels ``y`` (stable)."""
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def _center_to_rate(logits: np.ndarray, rate: float) -> float:
    """Offset ``c`` such that ``mean(sigmoid(logits - c)) == rate`` (bisection).

    ``sigmoid(logits - c)`` is monotone decreasing in ``c``, so a 60-step
    bisection on ``[-50, 50]`` converges to machine precision.
    """
    lo, hi = -50.0, 50.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if _sigmoid_np(logits - c).mean() > rate:
            lo = c
        else:
            hi = c
    return 0.5 * (lo + hi)


def _learn_substrate_rate(
    logits: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    q_lo: float,
    q_hi: float,
    q_n: int,
) -> float:
    """Learn the substrate rate via the calibration-quantile sweep (paper §3.2).

    For each quantile ``t`` in ``linspace(q_lo, q_hi, q_n)``, threshold the cohort
    ``logits`` at their ``t``-quantile, use the surviving (thresholded) logits to
    predict the held-out deficit, and keep the ``t`` with the lowest held-out
    deficit BCE. Returns the implied rate ``1 - t*``. Falls back to the grid
    midpoint rate when the calibration split is degenerate (single-class).
    """
    mid_rate = 1.0 - 0.5 * (q_lo + q_hi)
    if len(y_cal) < 2 or float(y_cal.min()) == float(y_cal.max()):
        return mid_rate
    best_rate, best_bce = mid_rate, float("inf")
    for t in np.linspace(q_lo, q_hi, q_n):
        thr = float(np.quantile(logits, float(t)))
        masked = (logits > thr) * logits
        z = (x_cal * masked).mean(axis=1)
        bce = _bce_with_logits_np(z, y_cal)
        if bce < best_bce:
            best_bce, best_rate = bce, 1.0 - float(t)
    return best_rate


def fit_parcel_vae(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cov: np.ndarray | None = None,
    z_dim: int = 128,
    hidden: int = 256,
    epochs: int = 1000,
    lr: float = 2e-3,
    wd: float = 1e-4,
    balanced: bool = True,
    cal_frac: float = 0.2,
    q_lo: float = 0.95,
    q_hi: float = 0.99,
    q_n: int = 20,
    seed: int = 0,
    readout: str = "pw",
    continuous: bool = False,
) -> np.ndarray:
    """Fit the two-arm parcel VAE on one episode's ``(X, Y)``; return the ``(K,)``
    recovered substrate map.

    X is the ``(N, K)`` parcel-overlap matrix; Y the ``(N,)`` deficit — binary by default, or a
    continuous score (e.g. symptom severity) when ``continuous=True`` (Y is z-scored, the deficit
    head uses MSE, class-balancing is off, and the binary rate-calibration is skipped — the returned
    ``pw`` map is the same effective-weight substrate readout, rank-preserving).

    ``readout`` selects what map is returned (default ``"pw"``):

    * ``"pw"`` — the EFFECTIVE per-parcel weight ``E[sigma(gamma_p)*w]`` (clamped
      to >= 0 and max-normalised). This is the per-parcel coefficient the deficit
      head actually uses. The bare membership arm saturates and is
      non-identifiable, so the substrate signal lives in the product ``p*w``;
      ranking by it recovers the substrate robustly across optimisers (see
      VAE_EXPERIMENTS_LOGS.md, Rounds 8-10). PREFERRED.
    * ``"p"`` — legacy calibrated membership probability. The VAE is fit on a
      ``1 - cal_frac`` split; a quantile grid ``linspace(q_lo, q_hi, q_n)`` on the
      held-out split picks ``t*`` minimising deficit BCE (paper §3.2), and the
      cohort membership logits are sigmoid'd after an offset setting
      ``mean(map) == 1 - t*``. Fragile / optimiser-dependent — kept for
      comparison only.

    ``balanced`` applies a class-balancing ``pos_weight`` to the deficit BCE.
    Full-batch Adamax; the model is taken at the lowest-mask-BCE step (seeded).
    """
    torch.manual_seed(seed)
    Xn = np.asarray(X, dtype=np.float32)
    Yn = np.asarray(Y, dtype=np.float32)
    if continuous:  # regression target (e.g. severity): standardize, no class balancing
        Yn = ((Yn - Yn.mean()) / (Yn.std() + 1e-8)).astype(np.float32)
        balanced = False
    k_parcel = int(Xn.shape[1])
    Cn = None if cov is None else np.asarray(cov, dtype=np.float32)
    d_cov = 0 if Cn is None else int(Cn.shape[1])
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Yn), generator=gen).numpy()
    n_fit = max(int(round((1.0 - cal_frac) * len(Yn))), 1)
    fit_idx, cal_idx = perm[:n_fit], perm[n_fit:]

    model = ParcelLesionDeficitVAE(k_parcel, d_cov=d_cov, z_dim=z_dim, hidden=hidden)
    model.train()
    Xt, Yt = torch.from_numpy(Xn), torch.from_numpy(Yn)
    x_fit, y_fit = Xt[fit_idx], Yt[fit_idx]
    cov_fit = None if Cn is None else torch.from_numpy(Cn)[fit_idx]
    pw: torch.Tensor | None = None
    if balanced:
        npos = float(y_fit.sum())
        if npos > 0:
            pw = torch.tensor((len(y_fit) - npos) / npos)
    opt = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
    best_ll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        opt.zero_grad()
        loss, mask_ll = model(x_fit, y_fit, pw, cov=cov_fit, continuous=continuous)
        loss.backward()
        opt.step()
        ll = float(mask_ll.detach())
        if ll < best_ll:
            best_ll = ll
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    # Score map: effective per-parcel weight E[sigma(gamma_p)*w] (``pw``, default)
    # or the bare membership logit E[gamma_p] (``p``, legacy). Both are calibrated
    # identically via the held-out calibration split (learn the substrate rate that
    # minimises held-out deficit BCE, then centre so mean(map) == rate). The
    # returned map is ranking-identical to the raw score, and its mass is
    # concentrated at the learned rate so the downstream F1-optimal mask self-sizes
    # to ~|M| instead of flooding. ``pw`` is the identifiable, optimiser-robust
    # score (the bare membership saturates; see VAE_EXPERIMENTS_LOGS.md); and it is
    # exactly the per-parcel coefficient the deficit ``mean(X*p*w)`` uses, so the
    # rate-learning BCE probe is faithful.
    if readout == "pw":
        scores = model.effective_map(x_fit, y_fit).cpu().numpy().astype(np.float64)
    else:
        scores = model.logit_map(x_fit, y_fit).cpu().numpy().astype(np.float64)
    if len(cal_idx) > 0 and not continuous:
        rate = _learn_substrate_rate(
            scores, Xn[cal_idx], Yn[cal_idx], q_lo=q_lo, q_hi=q_hi, q_n=q_n
        )
    else:
        # continuous target has no binary rate to learn; use the midpoint-quantile fallback so
        # focal_mass_mask still self-sizes (ranking / dice@k is unaffected by this scaling).
        rate = 1.0 - 0.5 * (q_lo + q_hi)
    if readout == "pw":
        # The effective weight E[p*w] has a heavy positive tail, so sigmoid
        # centring saturates (can't hit the low rate → floods). Instead scale the
        # clamped weights so total mass == rate*K (the expected substrate size);
        # focal_mass_mask then selects ~rate*K ~ |M| parcels. Rank-preserving.
        pos = np.clip(scores, 0.0, None)
        s = float(pos.sum())
        return pos * (rate * scores.shape[0]) / s if s > 0 else pos
    offset = _center_to_rate(scores, rate)
    return _sigmoid_np(scores - offset)
