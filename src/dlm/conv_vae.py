"""3D-convolutional two-arm VAE for voxel-space substrate recovery.

A convolutional counterpart to :class:`ParcelLesionDeficitVAE`, adapting the public
`vae_lesion_deficit` reference (CoordConv, label-as-volume, downsample encoder + upsample
decoders, joint ``P(X, Y)``) while keeping our identifiable two-arm ``p·w`` readout. The
model works on the dense ``vol_shape`` volume (padded so it downsamples cleanly); the
per-voxel lesion matrix ``X (N, V_cov)`` is scattered in via ``mask_idx`` and the deficit is
a mean over the coverage voxels only. The recovered substrate is the cohort-mean effective
weight ``E[σ(γ)·w]`` gathered back at the coverage voxels — the same calibrated, self-sizing
readout the MLP baseline uses (see VAE_EXPERIMENTS_LOGS.md).

Covariate-aware training is deferred (blind substrate recovery only for now); the deficit
head keeps the scalar ``overlap_scale``/``overlap_bias`` so a covariate pathway can be added
later without a rewrite.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dlm.mlp_vae import _learn_substrate_rate


def _pad_to_multiple(shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    """Smallest per-axis size >= ``shape`` divisible by ``factor`` (for clean downsampling)."""
    sizes = [int(math.ceil(s / factor) * factor) for s in shape]
    return (sizes[0], sizes[1], sizes[2])


def _coord_channels(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """``(1, 3, D, H, W)`` normalised [0,1] z/y/x coordinate channels (3D CoordConv)."""
    d, h, w = shape
    zs = torch.linspace(0.0, 1.0, d, device=device).view(1, 1, d, 1, 1).expand(1, 1, d, h, w)
    ys = torch.linspace(0.0, 1.0, h, device=device).view(1, 1, 1, h, 1).expand(1, 1, d, h, w)
    xs = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, 1, 1, w).expand(1, 1, d, h, w)
    return torch.cat([zs, ys, xs], dim=1)


def _norm(kind: str, channels: int) -> nn.Module:
    """Normalisation layer: ``"gn"`` GroupNorm (robust for small per-episode batches) or ``"bn"``."""
    if kind == "gn":
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    if kind == "bn":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"unknown norm {kind!r}; use 'gn' or 'bn'")


class SBlock(nn.Module):
    """``Conv3d(k3) → norm → GELU`` with optional 2× avg-pool down / nearest up (reference block)."""

    def __init__(self, in_planes: int, planes: int, *, norm: str = "gn",
                 downsample: bool = False, upsample: bool = False) -> None:
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        self.conv = nn.Conv3d(in_planes, planes, kernel_size=3, stride=1, padding=1)
        self.norm = _norm(norm, planes)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.act(self.norm(self.conv(x)))
        if self.downsample:
            x = F.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ConvLesionDeficitVAE(nn.Module):
    """3D-conv two-arm VAE. Inference decoder emits ``(γ, w)`` per voxel; deficit uses ``E[X·σ(γ)·w]``.

    ``pad_shape`` is the (already padded) working volume; ``num_down`` downsampling levels take it to
    a ``latent_grid`` on which a dense ``μ/logσ²`` sits. Two decoders mirror the encoder (inference
    → 2 channels ``γ, w``; reconstruction → 1 channel), cropped back to ``pad_shape``.
    """

    def __init__(self, pad_shape: tuple[int, int, int], *, d_cov: int = 0, z_dim: int = 128,
                 start_dims: int = 16, num_down: int = 3, norm: str = "gn") -> None:
        super().__init__()
        self.pad_shape = pad_shape
        self.num_down = num_down
        self.half_z = z_dim // 2
        self.d_cov = int(d_cov)

        enc_layers: list[nn.Module] = []
        in_ch = 5  # X + 3 coords + Y-volume
        ch = start_dims
        chans: list[int] = []
        for _ in range(num_down):
            enc_layers.append(SBlock(in_ch, ch, norm=norm, downsample=True))
            chans.append(ch)
            in_ch = ch
            ch *= 2
        self.encoder = nn.ModuleList(enc_layers)
        self.top_ch = chans[-1]
        self.latent_grid = tuple(s // (2 ** num_down) for s in pad_shape)
        self.dense = self.top_ch * int(np.prod(self.latent_grid))
        self.mu = nn.Linear(self.dense, z_dim)
        self.logvar = nn.Linear(self.dense, z_dim)

        self.inf_fc = nn.Sequential(nn.Linear(self.half_z, self.dense), nn.GELU())
        self.rec_fc = nn.Sequential(nn.Linear(self.half_z, self.dense), nn.GELU())
        inf_up: list[nn.Module] = []
        rec_up: list[nn.Module] = []
        dch = self.top_ch
        for _ in range(num_down):
            nxt = max(dch // 2, start_dims)
            inf_up.append(SBlock(dch, nxt, norm=norm, upsample=True))
            rec_up.append(SBlock(dch, nxt, norm=norm, upsample=True))
            dch = nxt
        self.inf_up = nn.ModuleList(inf_up)
        self.rec_up = nn.ModuleList(rec_up)
        self.inf_head = nn.Conv3d(dch, 2, kernel_size=1)   # (γ, w)
        self.rec_head = nn.Conv3d(dch, 1, kernel_size=1)   # X̂

        self.overlap_scale = nn.Parameter(torch.tensor(5.0))
        self.overlap_bias = nn.Parameter(torch.tensor(0.0))
        if self.d_cov > 0:
            self.beta = nn.Parameter(torch.zeros(self.d_cov))   # covariate main effect
            self.gamma = nn.Parameter(torch.zeros(self.d_cov))  # covariate x substrate interaction

    def encode(self, x5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x5
        for layer in self.encoder:
            h = layer(h)
        h = h.reshape(h.size(0), -1)
        return self.mu(h), self.logvar(h)

    def _decode(self, z_half: torch.Tensor, fc: nn.Module, ups: nn.ModuleList,
                head: nn.Module) -> torch.Tensor:
        h = fc(z_half).view(z_half.size(0), self.top_ch, *self.latent_grid)
        for layer in ups:
            h = layer(h)
        return head(h)

    def _inputs(self, x_vol: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Assemble the 5-channel input: lesion + 3 coord channels + label-as-volume."""
        b = x_vol.size(0)
        coords = _coord_channels(self.pad_shape, x_vol.device).expand(b, 3, *self.pad_shape)
        y_vol = y.view(b, 1, 1, 1, 1).expand(b, 1, *self.pad_shape)
        return torch.cat([x_vol, coords, y_vol], dim=1)

    def forward(self, x_vol: torch.Tensor, cov_vol: torch.Tensor, y: torch.Tensor,
                pos_weight: torch.Tensor | None, *, cov: torch.Tensor | None = None,
                mask_weight: float = 1.0,
                recon_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, mask_bce)`` for a minibatch of scattered lesion volumes.

        ``x_vol`` is ``(B,1,*pad)``, ``cov_vol`` ``(1,1,*pad)`` the *coverage* mask (1 inside the
        analysis voxels), ``y`` ``(B,)``. The deficit is the mean of ``X·σ(γ)·w`` over coverage voxels.
        ``cov`` ``(B, d_cov)`` are the covariates: they enter the deficit HEAD only (main effect
        ``beta`` + substrate interaction ``gamma``), never the conv encoder, so the ``E[σ(γ)·w]``
        substrate readout stays covariate-free (recovered controlling for the covariates).
        ``mask_weight``/``recon_weight`` rebalance the loss — the reconstruction term (summed over
        voxels) otherwise dwarfs the scalar deficit BCE, starving the inference map of substrate
        gradient through the deep conv decoder; raising ``mask_weight`` (or lowering ``recon_weight``)
        lets the deficit organise ``p·w``.
        """
        mu, logvar = self.encode(self._inputs(x_vol, y))
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        kl = torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1)
        inf = self._decode(z[:, :self.half_z], self.inf_fc, self.inf_up, self.inf_head)
        p = torch.sigmoid(inf[:, 0:1])
        w = inf[:, 1:2]
        recon = torch.sigmoid(self._decode(z[:, self.half_z:], self.rec_fc, self.rec_up, self.rec_head))

        denom = cov_vol.sum().clamp_min(1.0)
        overlap = (x_vol * p * w * cov_vol).sum(dim=(-3, -2, -1), keepdim=False).view(-1, 1) / denom
        if cov is None or self.d_cov == 0:
            logits = self.overlap_scale * overlap + self.overlap_bias
        else:
            scale_i = self.overlap_scale * (1.0 + cov @ self.gamma).view(-1, 1)
            logits = scale_i * overlap + (cov @ self.beta).view(-1, 1) + self.overlap_bias
        mask_ll = F.binary_cross_entropy_with_logits(logits, y.view(-1, 1), pos_weight=pos_weight)
        recon_ll = (F.binary_cross_entropy(recon, x_vol, reduction="none") * cov_vol).sum(
            dim=(-4, -3, -2, -1)).mean()
        total = mask_weight * mask_ll + recon_weight * recon_ll + kl.mean()
        return total, mask_ll

    @torch.no_grad()
    def effective_volume(self, x_vol: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cohort-mean effective-weight volume ``E[σ(γ)·w]`` of shape ``pad_shape``."""
        mu, _ = self.encode(self._inputs(x_vol, y))
        inf = self._decode(mu[:, :self.half_z], self.inf_fc, self.inf_up, self.inf_head)
        eff = torch.sigmoid(inf[:, 0:1]) * inf[:, 1:2]
        return eff.mean(dim=0).squeeze(0)


def _scatter_to_volume(X: np.ndarray, mask_idx: np.ndarray,
                       vol_shape: tuple[int, int, int]) -> np.ndarray:
    """``(N, V_cov)`` coverage matrix → ``(N, *vol_shape)`` dense volumes (zeros off-mask)."""
    n = X.shape[0]
    flat = np.zeros((n, int(np.prod(vol_shape))), dtype=np.float32)
    flat[:, mask_idx] = X.astype(np.float32)
    return flat.reshape(n, *vol_shape)


def _pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_coords(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """``(1, 3, D, H, W)`` normalised [0,1] z/y/x coordinate channels (3D CoordConv)."""
    return _coord_channels(shape, device)


def fit_conv_vae(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mask_idx: np.ndarray,
    vol_shape: tuple[int, int, int],
    cov: np.ndarray | None = None,
    z_dim: int = 128,
    start_dims: int = 16,
    num_down: int = 3,
    norm: str = "gn",
    epochs: int = 300,
    batch: int = 64,
    lr: float = 2e-3,
    wd: float = 1e-4,
    mask_weight: float = 1.0,
    recon_weight: float = 1.0,
    balanced: bool = True,
    cal_frac: float = 0.2,
    q_lo: float = 0.95,
    q_hi: float = 0.99,
    q_n: int = 20,
    seed: int = 0,
    device: str | None = None,
) -> np.ndarray:
    """Fit the 3D-conv two-arm VAE on one episode; return the ``(V_cov,)`` recovered substrate map.

    ``X`` is the ``(N, V_cov)`` coverage-voxel lesion matrix, ``Y`` the ``(N,)`` binary deficit,
    ``mask_idx`` the flat indices of the coverage voxels within ``vol_shape``. Optional ``cov``
    ``(N, d_cov)`` covariates enter the deficit head ONLY (main + interaction), leaving the substrate
    readout covariate-free (``cov=None`` reproduces the covariate-blind fit). Returns the cohort-mean
    effective weight ``E[σ(γ)·w]`` at the coverage voxels, clamped ≥0 and mass-scaled to the learned
    substrate rate (so ``focal_mass_mask`` self-sizes to ≈|M|) — same calibration as ``fit_parcel_vae``.
    """
    torch.manual_seed(seed)
    dev = _pick_device(device)
    v_cov = int(X.shape[1])
    pad_shape = _pad_to_multiple(vol_shape, 2 ** num_down)

    Xn = np.asarray(X, dtype=np.float32)
    Yn = np.asarray(Y, dtype=np.float32)
    Cn = None if cov is None else np.asarray(cov, dtype=np.float32)
    d_cov = 0 if Cn is None else int(Cn.shape[1])
    vol = _scatter_to_volume(Xn, mask_idx, vol_shape)  # (N, *vol_shape)
    pw = tuple(p - s for p, s in zip(pad_shape, vol_shape))  # per-axis pad amounts
    vol_p = np.pad(vol, ((0, 0), (0, pw[0]), (0, pw[1]), (0, pw[2])))  # (N, *pad)
    x_all = torch.from_numpy(vol_p).unsqueeze(1).to(dev)  # (N,1,*pad)
    y_all = torch.from_numpy(Yn).to(dev)
    cov_all = None if Cn is None else torch.from_numpy(Cn).to(dev)

    cov_flat = np.zeros(int(np.prod(vol_shape)), dtype=np.float32)
    cov_flat[mask_idx] = 1.0
    cov_p = np.pad(cov_flat.reshape(vol_shape), ((0, pw[0]), (0, pw[1]), (0, pw[2])))
    cov_vol = torch.from_numpy(cov_p).view(1, 1, *pad_shape).to(dev)

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Yn), generator=gen).numpy()
    n_fit = max(int(round((1.0 - cal_frac) * len(Yn))), 2)
    fit_idx, cal_idx = perm[:n_fit], perm[n_fit:]
    x_fit, y_fit = x_all[fit_idx], y_all[fit_idx]
    cov_fit = None if cov_all is None else cov_all[fit_idx]

    pos_weight: torch.Tensor | None = None
    if balanced:
        npos = float(y_fit.sum())
        if npos > 0:
            pos_weight = torch.tensor((len(y_fit) - npos) / npos, device=dev)

    model = ConvLesionDeficitVAE(pad_shape, d_cov=d_cov, z_dim=z_dim, start_dims=start_dims,
                                 num_down=num_down, norm=norm).to(dev)
    opt = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
    best_ll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(fit_idx), generator=gen).numpy()
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            if len(chunk) < 2:  # norm layers need >1 sample
                continue
            opt.zero_grad()
            cb = None if cov_fit is None else cov_fit[chunk]
            loss, _ = model(x_fit[chunk], cov_vol, y_fit[chunk], pos_weight,
                            cov=cb, mask_weight=mask_weight, recon_weight=recon_weight)
            loss.backward()
            opt.step()
        # Eval mask-BCE MINIBATCHED — a full-batch forward on all fit samples blows the
        # activation memory (a 3D conv on the whole cohort is enormous); accumulate instead.
        model.eval()
        with torch.no_grad():
            tot, nb = 0.0, 0
            for start in range(0, len(fit_idx), batch):
                xb, yb = x_fit[start:start + batch], y_fit[start:start + batch]
                if len(xb) < 2:
                    continue
                cb = None if cov_fit is None else cov_fit[start:start + batch]
                _, mb = model(xb, cov_vol, yb, pos_weight, cov=cb)
                tot += float(mb) * len(xb)
                nb += len(xb)
        ll = tot / max(nb, 1)
        if ll < best_ll:
            best_ll = ll
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    # Cohort-mean effective-weight volume, also minibatched (sum over samples / N).
    eff_sum = torch.zeros(pad_shape, device=dev)
    n_seen = 0
    with torch.no_grad():
        for start in range(0, len(fit_idx), batch):
            xb, yb = x_fit[start:start + batch], y_fit[start:start + batch]
            eff_sum = eff_sum + model.effective_volume(xb, yb) * len(xb)
            n_seen += len(xb)
    eff = (eff_sum / max(n_seen, 1)).cpu().numpy()  # (*pad)
    eff = eff[: vol_shape[0], : vol_shape[1], : vol_shape[2]].reshape(-1)[mask_idx]
    scores = eff.astype(np.float64)
    if len(cal_idx) > 0:
        rate = _learn_substrate_rate(scores, Xn[cal_idx], Yn[cal_idx], q_lo=q_lo, q_hi=q_hi, q_n=q_n)
    else:
        rate = 1.0 - 0.5 * (q_lo + q_hi)
    pos = np.clip(scores, 0.0, None)
    s = float(pos.sum())
    return pos * (rate * v_cov) / s if s > 0 else pos


__all__ = ["ConvLesionDeficitVAE", "SBlock", "fit_conv_vae", "add_coords"]
