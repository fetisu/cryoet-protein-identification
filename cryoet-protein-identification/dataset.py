# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd
import torch
import zarr
from rotate_flip import random_flip_3d, random_intensity_shift, rotate_90_3d
from torch.utils.data import Dataset
from utils import data_path_name, seg_path_name


class CZIIDataset(Dataset):
    def __init__(self, data_path, seg_path, df, obj_mapper, usage="train"):
        self.obj = obj_mapper.keys()
        self.obj_mapper = obj_mapper
        self.data_path = data_path
        self.seg_path = seg_path
        self.df = df
        self.usage = usage
        self.coor_df = self.create_coor()
        self.voxel = np.zeros((len(self.obj), 184, 630, 630), dtype=np.float16)
        self.mask = np.zeros((len(self.obj), 6, 184, 630, 630), dtype=np.int8)

        for obj, id in self.obj_mapper.items():
            self.voxel[id, :, :, :] = self.normalize_numpy(
                zarr.open(data_path + obj + data_path_name)["0"][:]
            )
            mask = zarr.open(seg_path + obj + seg_path_name)["0"][:]
            mask_list = []
            for i in range(0, 6):
                mask_list.append((mask == i))
            mask = np.stack(mask_list, axis=0)
            self.mask[id] = mask.astype(np.int8)

        self.depth = 32

    def __getitem__(self, index):
        row = self.coor_df.iloc[index]
        obj = row["experiment"]
        x = row["x"]
        y = row["y"]
        z = row["z"]

        if self.usage == "train":
            x = min(max(0, x + random.randint(-128, 128)), 630 - 256)
            y = min(max(0, y + random.randint(-128, 128)), 630 - 256)
            z = min(max(0, z + random.randint(-16, 16)), 184 - 32)
        if z > 184 - 32:
            z = 184 - 32
        if x > 630 - 256:
            x = 630 - 256
        if y > 630 - 256:
            y = 630 - 256

        data = self.voxel[self.obj_mapper[obj]][
            z : z + self.depth, y : y + 256, x : x + 256
        ]
        mask = self.mask[self.obj_mapper[obj]][
            :, z : z + self.depth, y : y + 256, x : x + 256
        ]
        if self.usage == "train":
            data, mask = rotate_90_3d(data, mask)
            data, mask = random_flip_3d(data, mask)
            data = random_intensity_shift(data)

        id_tensor = torch.tensor(self.obj_mapper[obj])

        data = torch.tensor(data, dtype=torch.float16)
        mask = torch.tensor(mask, dtype=torch.float16)

        return (
            data.unsqueeze(0),
            mask,
            id_tensor,
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

    def create_coor(self):
        coor_dict = {"experiment": [], "x": [], "y": [], "z": []}
        if self.usage == "train":
            for obj in self.obj:
                for x in range(6):
                    for y in range(6):
                        for z in range(8):
                            coor_dict["experiment"].append(obj)
                            coor_dict["x"].append(x * 128 - 23 * x)
                            coor_dict["y"].append(y * 128 - 23 * y)
                            coor_dict["z"].append(z * 32 - 8 * z)
        else:
            for obj in self.obj:
                for x in range(3):
                    for y in range(3):
                        for z in range(8):
                            coor_dict["experiment"].append(obj)
                            coor_dict["x"].append(x * 256 - 69 * x)
                            coor_dict["y"].append(y * 256 - 69 * y)
                            coor_dict["z"].append(z * 32 - 8 * z)

        return pd.DataFrame(coor_dict)

    def __len__(self):
        return len(self.coor_df)
