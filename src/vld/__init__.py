"""vld — lesion-deficit substrate recovery in voxel space.

VLSM and SVR-LSM baselines, two-arm VAEs (MLP and 3D-conv) with a covariate-aware deficit head, and
calibrated bagging. See the notebooks/ tutorials and README for usage.
"""
from vld.baselines import svr_binary, svr_weights, vlsm_binary, vlsm_stat
from vld.causal import g_computation_ate, marginal_effect_map
from vld.data import VoxelData, build_voxel_data
from vld.metrics import dice, dice_at_k, focal_mass_mask, morans_i, support_auc
from vld.priors import (
    LaplacianInverse,
    fit_laplacian_substrate,
    voxel_adjacency,
    voxel_laplacian,
)
from vld.models import (
    BaggedSubstrate,
    ConvLesionDeficitVAE,
    consensus_substrate,
    fit_conv_vae,
    fit_conv_vae_bagged,
    fit_parcel_vae,
)

__version__ = "0.1.0"

__all__ = [
    "fit_conv_vae", "fit_parcel_vae", "ConvLesionDeficitVAE",
    "fit_conv_vae_bagged", "consensus_substrate", "BaggedSubstrate",
    "vlsm_stat", "vlsm_binary", "svr_weights", "svr_binary",
    "VoxelData", "build_voxel_data",
    "dice", "dice_at_k", "focal_mass_mask", "support_auc", "morans_i",
    "LaplacianInverse", "fit_laplacian_substrate", "voxel_adjacency", "voxel_laplacian",
    "g_computation_ate", "marginal_effect_map",
]
