# -*- coding: utf-8 -*-
import gc
import json
import os
import random
from glob import glob

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
import wandb
from dataset import CZIIDataset
from eval_loss import BCEDiceLoss, np_find_centroid, np_find_component, score
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from threed_models import UNet3D
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils import exp_name, hydra_config_path
from utils import seed as SEED
from utils import wandb_filename


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = (
        True  # Fix the network according to random seed
    )


class CZIIModule(pl.LightningModule):
    def __init__(self, device, config, df, obj_mapper):
        super().__init__()
        self.config = config
        self.obj_mapper = obj_mapper
        self.inv_obj_mapper = {v: k for k, v in obj_mapper.items()}
        self.pred_temp = np.zeros(
            (len(self.obj_mapper.keys()), 5, 184, 630, 630), dtype=np.float16
        )
        self.mask_temp = np.zeros(
            (len(self.obj_mapper.keys()), 5, 184, 630, 630), dtype=np.float16
        )
        self.wrap_temp = np.zeros(
            (len(self.obj_mapper.keys()), 5, 184, 630, 630), dtype=np.float16
        )
        self.df = df
        self.model = UNet3D(in_channels=1, out_channels=5)
        self.loss_module = BCEDiceLoss(device, lambda_dice=0)

        self.val_step_labels = []

    def forward(self, batch):
        preds = self.model(batch)
        return preds

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), **self.config["model"]["optimizer_params"]
        )
        if self.config["model"]["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                **self.config["model"]["scheduler"]["params"][
                    "CosineAnnealingLR"
                ],
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        elif self.config["model"]["scheduler"]["name"] == "cosine_with_warmup":
            print("cosine with warmup")
            print(
                self.config["model"]["scheduler"]["params"][
                    "cosine_with_warmup"
                ]
            )
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                **self.config["model"]["scheduler"]["params"][
                    "cosine_with_warmup"
                ],
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        else:
            return {"optimizer": optimizer}

    def training_step(self, config, batch, batch_idx):
        volume, mask, _, _, _, _ = batch
        preds = self.model.forward(volume)
        loss = self.loss_module(preds, mask)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=config["train_bs"],
        )
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Add TTA"""
        volume, mask, id_tensor, x, y, z = batch
        # Add TTA
        shape = volume.shape

        volume = torch.cat(
            [
                volume,
                torch.rot90(volume, k=1, dims=(3, 4)),
                torch.rot90(volume, k=2, dims=(3, 4)),
                torch.rot90(volume, k=3, dims=(3, 4)),
            ],
            dim=0,
        )
        preds = self.model.forward(volume)
        preds = preds.reshape(4, shape[0], 5, *shape[2:])
        preds = torch.stack(
            [
                preds[0],
                torch.rot90(preds[1], k=3, dims=(3, 4)),
                torch.rot90(preds[2], k=2, dims=(3, 4)),
                torch.rot90(preds[3], k=1, dims=(3, 4)),
            ],
            dim=0,
        ).mean(0)
        loss = self.loss_module(preds, mask)

        for _id, _x, _y, _z, _pred, _mask in zip(
            id_tensor, x, y, z, preds, mask
        ):
            self.pred_temp[
                _id, :, _z : _z + 32, _y : _y + 256, _x : _x + 256
            ] += (_pred.sigmoid().cpu().numpy().astype(np.float16)[:, :, :, :])
            self.mask_temp[
                _id, :, _z : _z + 32, _y : _y + 256, _x : _x + 256
            ] += (_mask.detach().cpu().numpy().astype(np.float16)[1:, :, :, :])
            self.wrap_temp[
                _id, :, _z : _z + 32, _y : _y + 256, _x : _x + 256
            ] += 1
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        label = self.df
        if np.sum(self.wrap_temp == 0) == 0:
            self.pred_temp = self.pred_temp / self.wrap_temp
            self.mask_temp = self.mask_temp / self.wrap_temp
            max_total_fbeta = 0.0
            max_total_fbeta_thresh = 0
            each_fbeta_dict = {
                "apo-ferritin": [],
                "beta-amylase": [],
                "beta-galactosidase": [],
                "ribosome": [],
                "thyroglobulin": [],
                "virus-like-particle": [],
            }
            threshold_list = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4]
            for th in threshold_list:
                threshold = [th, th, th, th, th]
                pred_df_list = []
                for obj, id in self.obj_mapper.items():
                    pred = self.pred_temp[id]
                    component = np_find_component(pred, threshold)
                    centroid = np_find_centroid(component)
                    particles = [
                        "apo-ferritin",
                        "beta-galactosidase",
                        "ribosome",
                        "thyroglobulin",
                        "virus-like-particle",
                    ]
                    df_list = []
                    for c, p in zip(centroid, particles):
                        centroid_dict = {
                            "experiment": obj,
                            "particle_type": p,
                            "x": c[:, 0],
                            "y": c[:, 1],
                            "z": c[:, 2],
                        }
                        df_list.append(pd.DataFrame(centroid_dict))
                    pred_df = pd.concat(df_list)
                    pred_df_list.append(pred_df)
                pred_df = pd.concat(pred_df_list)
                pred_df["id"] = range(len(pred_df))
                label["id"] = range(len(label))
                total_fbeta, partial_fbeta = score(
                    label.reset_index(drop=True),
                    pred_df.reset_index(drop=True),
                    row_id_column_name="id",
                    distance_multiplier=0.5,
                    beta=4,
                )
                self.log(
                    f"val_fbeta_th-{th}",
                    total_fbeta,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                for k, v in partial_fbeta.items():
                    self.log(
                        f"val_partial_fbeta_{k}_th-{th}",
                        v,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    each_fbeta_dict[k].append(v)
            max_fbeta_dict = {}
            max_fbeta_threshold_list = []
            for k, v in each_fbeta_dict.items():
                # choose max fbeta and its threshold
                max_fbeta_dict[k] = max(v)
                max_fbeta_idx = v.index(max_fbeta_dict[k])
                max_fbeta_threshold_list.append(threshold_list[max_fbeta_idx])
            max_total_fbeta = 0
            for k, v in max_fbeta_dict.items():
                if k == "apo-ferritin":
                    max_total_fbeta += v
                elif k == "beta-galactosidase":
                    max_total_fbeta = max_total_fbeta + v * 2
                elif k == "ribosome":
                    max_total_fbeta += v
                elif k == "thyroglobulin":
                    max_total_fbeta = max_total_fbeta + v * 2
                elif k == "virus-like-particle":
                    max_total_fbeta += v
                else:
                    continue
            max_total_fbeta_thresh = ", ".join(
                [str(t) for t in max_fbeta_threshold_list]
            )
            max_total_fbeta = max_total_fbeta / 7
            self.log(
                "max_fbeta",
                max_total_fbeta,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            print(max_total_fbeta_thresh, max_total_fbeta)

        self.pred_temp[:, :, :, :, :] = 0
        self.mask_temp[:, :, :, :, :] = 0
        self.wrap_temp[:, :, :, :, :] = 0
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)
        return 0

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)


@hydra.main(
    version_base=None, config_path=hydra_config_path, config_name="train"
)
def main(config: DictConfig):
    df_list = []
    pickable_obj_path_list = [
        p + "Picks/"
        for p in glob("../kaggle/input/train/overlay/ExperimentRuns/*/")
    ]
    pickable_obj_path_list
    for pickable_obj_path in pickable_obj_path_list:
        path_list = glob(pickable_obj_path + "*.json")
        for path in path_list:
            picks = json.load(open(path))
            pickable_object_name = picks["pickable_object_name"]
            run_name = picks["run_name"]
            points = picks["points"]
            point_dict = {"x": [], "y": [], "z": []}
            for p in points:
                point_dict["x"].append(p["location"]["x"])
                point_dict["y"].append(p["location"]["y"])
                point_dict["z"].append(p["location"]["z"])
            df = pd.DataFrame(point_dict)
            df["experiment"] = run_name
            df["particle_type"] = pickable_object_name
            df_list.append(df)
    df = pd.concat(df_list)
    df = df[["experiment", "particle_type", "x", "y", "z"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    torch.set_float32_matmul_precision("medium")

    df["fold"] = 3
    for i, e in enumerate(exp_name):
        df.loc[df.experiment == e, "fold"] = i

    T_MAX = config["model"]["scheduler"]["params"]["CosineAnnealingLR"][
        "T_max"
    ]
    num_training_steps = config["model"]["scheduler"]["params"][
        "cosine_with_warmup"
    ]["num_training_steps"]
    num_warmup_steps = config["model"]["scheduler"]["params"][
        "cosine_with_warmup"
    ]["num_warmup_steps"]

    for i in [0, 1, 2]:
        train_df = df.loc[df.fold == i]
        valid_df = df.loc[df.fold == i]
        train_df = train_df.drop("fold", axis=1)
        valid_df = valid_df.drop("fold", axis=1)
        train_obj_mapper = {
            exp: i for i, exp in enumerate(train_df.experiment.unique())
        }
        valid_obj_mapper = {
            exp: i for i, exp in enumerate(valid_df.experiment.unique())
        }
        print("train_obj_mapper! ", train_obj_mapper)
        print("valid_obj_mapper! ", valid_obj_mapper)
        dataset_train = CZIIDataset(
            config["data_path"],
            config["mask_path"],
            train_df,
            train_obj_mapper,
            "train",
        )
        dataset_validation = CZIIDataset(
            config["data_path"],
            config["mask_path"],
            valid_df,
            valid_obj_mapper,
            "valid",
        )
        print("dataset_train! ", dataset_train)
        print("dataset_validation! ", dataset_validation)
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=config["train_bs"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        data_loader_validation = DataLoader(
            dataset_validation,
            batch_size=config["valid_bs"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        checkpoint_callback = ModelCheckpoint(
            save_weights_only=True,
            monitor="max_fbeta",
            dirpath=config["output_dir"],
            mode="max",
            filename=wandb_filename,
            save_top_k=config["save_topk"],
            verbose=1,
        )

        progress_bar_callback = TQDMProgressBar(
            refresh_rate=config["progress_bar_refresh_rate"]
        )

        wandb_logger = WandbLogger(
            project="czii",
            name=wandb_filename,
            log_model=False,
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[
                checkpoint_callback,
                progress_bar_callback,
            ],
            check_val_every_n_epoch=1,
            **config["trainer"],
        )

        config["model"]["scheduler"]["params"]["CosineAnnealingLR"][
            "T_max"
        ] = (T_MAX * len(data_loader_train) / config["trainer"]["devices"])
        config["model"]["scheduler"]["params"]["cosine_with_warmup"][
            "num_training_steps"
        ] = int(
            num_training_steps
            * len(data_loader_train)
            / config["trainer"]["devices"]
        )
        config["model"]["scheduler"]["params"]["cosine_with_warmup"][
            "num_warmup_steps"
        ] = int(
            num_warmup_steps
            * len(data_loader_train)
            / config["trainer"]["devices"]
        )
        model = CZIIModule(
            device, config=config, df=valid_df, obj_mapper=valid_obj_mapper
        )
        trainer.fit(model, data_loader_train, data_loader_validation)
        del (
            dataset_train,
            dataset_validation,
            data_loader_train,
            data_loader_validation,
            model.pred_temp,
            model.wrap_temp,
            model.mask_temp,
            model,
        )
        gc.collect()
        wandb.finish()


if __name__ == "__main__":
    main()
