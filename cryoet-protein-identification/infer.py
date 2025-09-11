# -*- coding: utf-8 -*-
import gc
from glob import glob

import cc3d
import hydra
import numpy as np
import pandas as pd
import torch
from constants import (
    COOR_LIST,
    PATCH_D,
    PATCH_H,
    PATCH_W,
    hydra_config_path,
    infer_particles,
    particle_radius,
)

# from eval_loss import score
from joblib import Parallel, delayed
from module_dataset import CZIIDataset, CZIIModule
from omegaconf import DictConfig
from sklearn.cluster import DBSCAN

# from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader


def calc_centroid(pred_temp):
    component = cc3d.connected_components(pred_temp.copy(), connectivity=6)
    stats = cc3d.statistics(component)
    zyx = stats["centroids"][1:]
    return np.ascontiguousarray(zyx[:, ::-1]) * 10.012444


@hydra.main(
    version_base=None, config_path=hydra_config_path, config_name="infer"
)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = torch.tensor(config["thresholds"], device=device).reshape(
        5, 1, 1, 1
    )

    model_path_list = config["model_path_list"]

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

    exp_list = [
        p.split("/")[-1]
        for p in glob(config["data_path"] + "*", recursive=True)
    ]
    pred_df_list = []
    num_model = len(model_list)
    for exp in exp_list:
        dataset = CZIIDataset(COOR_LIST, config["data_path"], exp)
        data_loader = DataLoader(
            dataset,
            batch_size=config["bs"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        pred_temp = torch.zeros(
            (5, 184, 630, 630), device=device, dtype=torch.float16
        )
        wrap_temp = torch.zeros(
            (5, 184, 630, 630), device=device, dtype=torch.float16
        )
        with torch.no_grad():
            for _data, x, y, z in data_loader:
                _data = _data.to(device, non_blocking=True).half()
                shape = _data.shape
                _data = torch.cat(
                    [
                        _data,
                        torch.rot90(_data, k=1, dims=(3, 4)),
                        torch.rot90(_data, k=2, dims=(3, 4)),
                        torch.rot90(_data, k=3, dims=(3, 4)),
                    ],
                    dim=0,
                )
                pred_mask_list = []
                for model in model_list:
                    preds = model.forward(_data)
                    pred_mask_list.append(preds)
                preds = torch.stack(pred_mask_list, dim=0)
                preds = preds.reshape(num_model, 4, shape[0], 5, *shape[2:])
                preds = (
                    torch.stack(
                        [
                            preds[:, 0],
                            torch.rot90(preds[:, 1], k=3, dims=(4, 5)),
                            torch.rot90(preds[:, 2], k=2, dims=(4, 5)),
                            torch.rot90(preds[:, 3], k=1, dims=(4, 5)),
                        ],
                        dim=1,
                    )
                    .mean(1)
                    .sigmoid()
                    .sum(0)
                )
                for _x, _y, _z, _pred in zip(x, y, z, preds):
                    pred_temp[
                        :,
                        _z : _z + PATCH_D,
                        _y : _y + PATCH_H,
                        _x : _x + PATCH_W,
                    ].add_(_pred)
                    wrap_temp[
                        :,
                        _z : _z + PATCH_D,
                        _y : _y + PATCH_H,
                        _x : _x + PATCH_W,
                    ].add_(num_model)
        pred_temp = pred_temp / wrap_temp
        pred_temp = (pred_temp > threshold).detach().cpu().numpy()
        del dataset, data_loader
        gc.collect()
        centroid = []
        centroid = Parallel(n_jobs=5)(
            [delayed(calc_centroid)(p_pred) for p_pred in pred_temp]
        )

        for c, p in zip(centroid, infer_particles):
            centroid_dict = {
                "experiment": exp,
                "particle_type": p,
                "x": c[:, 0],
                "y": c[:, 1],
                "z": c[:, 2],
            }
            pred_df_list.append(pd.DataFrame(centroid_dict))
        del centroid, centroid_dict
        gc.collect()

    pred_df = pd.concat(pred_df_list)
    pred_df["id"] = range(len(pred_df))
    pred_df = pred_df.reset_index(drop=True)

    df = pred_df.copy()

    final = []
    for _pidx, p in enumerate(infer_particles):
        pdf = df[df["particle_type"] == p].reset_index(drop=True)
        p_rad = particle_radius[p]

        grouped = pdf.groupby(["experiment"])

        for _exp, group in grouped:
            group = group.reset_index(drop=True)

            coords = group[["x", "y", "z"]].values
            db = DBSCAN(
                eps=p_rad * 0.5,
                min_samples=2,
                metric="euclidean",
                algorithm="kd_tree",
            ).fit(coords)
            labels = db.labels_

            group["cluster"] = labels

            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue

                cluster_points = group[group["cluster"] == cluster_id]

                avg_x = cluster_points["x"].mean()
                avg_y = cluster_points["y"].mean()
                avg_z = cluster_points["z"].mean()

                group.loc[group["cluster"] == cluster_id, ["x", "y", "z"]] = (
                    avg_x,
                    avg_y,
                    avg_z,
                )
                group = group.drop_duplicates(subset=["x", "y", "z"])

            final.append(group)

    pred_df = pd.concat(final, ignore_index=True)
    pred_df = pred_df.drop(columns=["cluster"])
    pred_df = pred_df.sort_values(
        by=["experiment", "particle_type"]
    ).reset_index(drop=True)
    pred_df["id"] = np.arange(0, len(pred_df))

    pred_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
