# -*- coding: utf-8 -*-
import numpy as np
import pytorch_lightning as pl
import torch
import zarr
from constants import PATCH_D, PATCH_H, PATCH_W
from threed_models import UNet3D
from torch.utils.data import Dataset


class CZIIDataset(Dataset):
    def __init__(self, coor_list, data_path, obj, usage="sub"):
        self.obj = obj
        self.data_path = data_path
        self.usage = usage
        self.coor_list = coor_list
        self.voxel = np.zeros((184, 630, 630), dtype=np.float32)

        self.voxel[:, :, :] = self.normalize_numpy(
            zarr.open(data_path + obj + "/VoxelSpacing10.000/denoised.zarr")[
                "0"
            ][:]
        )

    def __getitem__(self, index):
        row = self.coor_list[index]
        x = row[0]
        y = row[1]
        z = row[2]

        if z > 184 - PATCH_D:
            z = 184 - PATCH_D
        if x > 630 - PATCH_W:
            x = 630 - PATCH_W
        if y > 630 - PATCH_H:
            y = 630 - PATCH_H
        data = self.voxel[z : z + PATCH_D, y : y + PATCH_H, x : x + PATCH_W]
        data = torch.tensor(data, dtype=torch.float32)

        return (
            data.unsqueeze(0),
            torch.tensor(x),
            torch.tensor(y),
            torch.tensor(z),
        )

    def normalize_numpy(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    def __len__(self):
        return len(self.coor_list)


class CZIIModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UNet3D(in_channels=1, out_channels=5)

    def forward(self, batch):
        preds = self.model(batch)
        return preds
