# -*- coding: utf-8 -*-
from pathlib import Path

# create_masks
# paths
source_dir = Path("../kaggle/input/train/overlay")
destination_dir = Path("../kaggle/working/overlay")
copick_config_path = Path("../kaggle/working/copick.config")

# copick
copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"

# scales
particle_scales = {
    "apo-ferritin": 1 / 2,
    "beta-galactosidase": 1 / 2,
    "ribosome": 1 / 3,
    "thyroglobulin": 1 / 3,
    "virus-like-particle": 1 / 3,
}

# dataset
# paths
data_path_name = "/VoxelSpacing10.000/denoised.zarr"
seg_path_name = (
    "/Segmentations/10.000_copickUtils_0_paintedPicks-multilabel.zarr"
)

# eval_loss
particle_radius = {
    "apo-ferritin": 60,
    "beta-amylase": 65,
    "beta-galactosidase": 90,
    "ribosome": 150,
    "thyroglobulin": 130,
    "virus-like-particle": 135,
}
particle_weights = {
    "apo-ferritin": 1,
    "beta-amylase": 0,
    "beta-galactosidase": 2,
    "ribosome": 1,
    "thyroglobulin": 2,
    "virus-like-particle": 1,
}

# train
hydra_config_path = "../configs"
exp_name = ["TS_69_2", "TS_86_3", "TS_73_6"]
wandb_filename = (
    "_convnext-window_32_320_320-pre_ds-r5n5n33n33n33-4tta-bce-30e-all"
)

# infer
infer_particles = [
    "apo-ferritin",
    "beta-galactosidase",
    "ribosome",
    "thyroglobulin",
    "virus-like-particle",
]
