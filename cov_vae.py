"""Two-arm lesion-deficit VAE with an optional covariate deficit head.

Per-episode-fit substrate-recovery VAE. Two per-parcel arms decouple roles a single map
conflates: a MEMBERSHIP arm ``p = sigma(gamma_p)`` (which parcels are in the substrate) and
an IMPORTANCE arm ``w`` (how much damage there drives the deficit). The deficit is predicted
from the gated overlap ``mean(X * p * w)``; a lesion-reconstruction arm regularises the
latent. The recovered substrate is the cohort-averaged EFFECTIVE weight ``E[p * w]``.

Covariate-aware head (``d_cov > 0``): ``logit = scale * overlap * (1 + C @ gamma) + C @ beta
+ bias`` with global learnable ``beta`` (main effect) and ``gamma`` (interaction). Covariates
enter the deficit head ONLY, never the mask-latent encoder, so the ``E[p*w]`` readout stays
covariate-free -- i.e. the substrate is recovered *controlling for* the covariates.
``cov=None`` / ``d_cov=0`` reduces exactly to the covariate-free VAE.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SubstrateVAE(nn.Module):
    """Two-arm VAE: membership ``p = sigma(gamma_p)`` x importance ``w``, + covariate head."""

    def __init__(self, K: int, *, d_cov: int = 0, z_dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.K = K
        self.half = z_dim // 2
        self.enc = nn.Sequential(nn.Linear(K + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(nn.Linear(self.half, hidden), nn.GELU(),
                                 nn.Linear(hidden, K * 2))       # ch0 = gamma_p, ch1 = w
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

    def forward(self, X: torch.Tensor, Y: torch.Tensor, pos_weight: torch.Tensor | None,
                cov: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, mask_bce)`` for one full-batch step."""
        h = self.enc(torch.cat([X, Y.view(-1, 1)], dim=1))      # NOTE: cov deliberately absent
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        mask_z, recon_z = z[:, :self.half], z[:, self.half:]
        kl = torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1)
        p, w = self._arms(mask_z)
        recon = torch.sigmoid(self.rdec(recon_z))
        overlap = (X * p * w).mean(dim=1, keepdim=True)
        if cov is None or self.d_cov == 0:
            logits = self.overlap_scale * overlap + self.overlap_bias
        else:
            scale_i = self.overlap_scale * (1.0 + cov @ self.gamma).view(-1, 1)
            logits = scale_i * overlap + (cov @ self.beta).view(-1, 1) + self.overlap_bias
        mask_ll = F.binary_cross_entropy_with_logits(logits, Y.view(-1, 1), pos_weight=pos_weight)
        recon_ll = F.binary_cross_entropy(recon, X, reduction="none").sum(dim=1).mean()
        return mask_ll + recon_ll + kl.mean(), mask_ll

    @torch.no_grad()
    def effective_map(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """``(K,)`` cohort-averaged EFFECTIVE per-parcel weight ``E[sigma(gamma_p) * w]``.

        This is the per-parcel coefficient the deficit head actually uses; the bare
        membership arm saturates and is non-identifiable, so the substrate signal lives in
        the product ``p*w``. Covariate-free by construction (encoder never sees ``cov``).
        """
        mask_z = self.mu(self.enc(torch.cat([X, Y.view(-1, 1)], dim=1)))[:, :self.half]
        p, w = self._arms(mask_z)
        return (p * w).mean(dim=0)


def _bce_with_logits_np(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def _learn_substrate_rate(scores: np.ndarray, x_cal: np.ndarray, y_cal: np.ndarray, *,
                          q_lo: float, q_hi: float, q_n: int) -> float:
    """Learn the substrate rate on a held-out split: the quantile threshold on ``scores``
    whose surviving voxels best predict the held-out deficit (min BCE). Returns ``1 - t*``.
    """
    mid = 1.0 - 0.5 * (q_lo + q_hi)
    if len(y_cal) < 2 or float(y_cal.min()) == float(y_cal.max()):
        return mid
    best_rate, best_bce = mid, float("inf")
    for t in np.linspace(q_lo, q_hi, q_n):
        thr = float(np.quantile(scores, float(t)))
        z = (x_cal * ((scores > thr) * scores)).mean(axis=1)
        bce = _bce_with_logits_np(z, y_cal)
        if bce < best_bce:
            best_bce, best_rate = bce, 1.0 - float(t)
    return best_rate


def fit_substrate_vae(X: np.ndarray, y: np.ndarray, *, cov: np.ndarray | None = None,
                      z_dim: int = 128, hidden: int = 256, epochs: int = 1000, lr: float = 2e-3,
                      wd: float = 1e-4, balanced: bool = True, cal_frac: float = 0.2,
                      q_lo: float = 0.95, q_hi: float = 0.99, q_n: int = 20,
                      seed: int = 0) -> np.ndarray:
    """Fit the two-arm VAE on one episode's ``(X, y)``; return the ``(K,)`` recovered map.

    ``X`` is the ``(N, K)`` voxel-overlap matrix, ``y`` the ``(N,)`` binary deficit. When
    ``cov`` is given, the covariate head controls for it (covariates route to the head only).
    The returned map is the effective weight ``E[p*w]`` clamped >=0 and scaled so its total
    mass equals the calibration-learned expected substrate size (so ``focal_mass_mask``
    self-sizes). Full-batch Adamax; model taken at the lowest deficit-BCE step (seeded).
    """
    torch.manual_seed(seed)
    Xn = np.asarray(X, dtype=np.float32)
    yn = np.asarray(y, dtype=np.float32)
    K = int(Xn.shape[1])
    Cn = None if cov is None else np.asarray(cov, dtype=np.float32)
    d_cov = 0 if Cn is None else int(Cn.shape[1])

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(yn), generator=gen).numpy()
    n_fit = max(int(round((1.0 - cal_frac) * len(yn))), 1)
    fit_idx, cal_idx = perm[:n_fit], perm[n_fit:]

    model = SubstrateVAE(K, d_cov=d_cov, z_dim=z_dim, hidden=hidden)
    model.train()
    Xt, yt = torch.from_numpy(Xn), torch.from_numpy(yn)
    x_fit, y_fit = Xt[fit_idx], yt[fit_idx]
    cov_fit = None if Cn is None else torch.from_numpy(Cn)[fit_idx]
    pw: torch.Tensor | None = None
    if balanced:
        npos = float(y_fit.sum())
        if npos > 0:
            pw = torch.tensor((len(y_fit) - npos) / npos)
    opt = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
    best_ll, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        opt.zero_grad()
        loss, mask_ll = model(x_fit, y_fit, pw, cov=cov_fit)
        loss.backward()
        opt.step()
        ll = float(mask_ll.detach())
        if ll < best_ll:
            best_ll, best_state = ll, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    scores = model.effective_map(x_fit, y_fit).cpu().numpy().astype(np.float64)
    if len(cal_idx) > 0:
        rate = _learn_substrate_rate(scores, Xn[cal_idx], yn[cal_idx], q_lo=q_lo, q_hi=q_hi, q_n=q_n)
    else:
        rate = 1.0 - 0.5 * (q_lo + q_hi)
    pos = np.clip(scores, 0.0, None)
    s = float(pos.sum())
    return pos * (rate * K) / s if s > 0 else pos


__all__ = ["SubstrateVAE", "fit_substrate_vae"]
