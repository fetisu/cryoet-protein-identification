# -*- coding: utf-8 -*-

from glob import glob

import torch
from constants import COOR_LIST
from module_dataset import CZIIDataset, CZIIModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def main(config: DictConfig):
    model_path_list = config["model_path_list"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_list = [
        p.split("/")[-1]
        for p in glob(config["data_path"] + "*", recursive=True)
    ]

    dataset = CZIIDataset(COOR_LIST, config["data_path"], exp_list[0])
    data_loader = DataLoader(
        dataset,
        batch_size=config["bs"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    with torch.no_grad():
        for _data, x, y, z in data_loader:
            _data = _data.to(device, non_blocking=True).half()
            _data = torch.cat(
                [
                    _data,
                    torch.rot90(_data, k=1, dims=(3, 4)),
                    torch.rot90(_data, k=2, dims=(3, 4)),
                    torch.rot90(_data, k=3, dims=(3, 4)),
                ],
                dim=0,
            )

    model_list = []
    for model_path in model_path_list:
        model = CZIIModule.load_from_checkpoint(
            model_path, config=config, strict=False
        )
        model.half()
        model.eval()
        for param in model.parameters():
            param.grad = None
        model.to(device)

        model_list.append(model)

    model.to_onnx(config["model_path"], _data, export_params=True)


if __name__ == "__main__":
    main()
