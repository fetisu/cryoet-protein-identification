# -*- coding: utf-8 -*-

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
PATCH_D = 32
PATCH_H = 320
PATCH_W = 320

COOR_LIST = []
for x in range(630 // PATCH_W + 1):
    for y in range(630 // PATCH_H + 1):
        for z in range(184 // PATCH_D + 2):
            COOR_LIST.append(
                [
                    x * PATCH_W - 10 * x,
                    y * PATCH_H - 10 * y,
                    z * PATCH_D - 6 * z,
                ]
            )
