# -*- coding: utf-8 -*-
from pathlib import Path

import fire
from convert_to_onnx import main as run_convert_to_onnx
from generate_masks import main as run_masks_generation
from hydra import compose, initialize_config_dir
from infer import main as run_infer
from train import main as run_train
from vizualize_masks import main as run_masks_vizualization


class CommandCenter:
    def __init__(self):
        config_dir = Path(__file__).resolve().parent / ".." / "configs"
        config_dir = str(config_dir.resolve())
        initialize_config_dir(config_dir)

    @staticmethod
    def train():
        cfg = compose(config_name="train.yaml")
        run_train(cfg)

    @staticmethod
    def infer():
        cfg = compose(config_name="infer.yaml")
        run_infer(cfg)

    @staticmethod
    def generate_masks():
        cfg = compose(config_name="generate_masks.yaml")
        run_masks_generation(cfg)

    @staticmethod
    def vizualize_masks():
        cfg = compose(config_name="vizualize_masks.yaml")
        run_masks_vizualization(cfg)

    @staticmethod
    def convert_to_onnx():
        cfg = compose(config_name="convert_to_onnx.yaml")
        run_convert_to_onnx(cfg)


if __name__ == "__main__":
    fire.Fire(CommandCenter)
