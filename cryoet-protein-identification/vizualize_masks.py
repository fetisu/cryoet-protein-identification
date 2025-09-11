# -*- coding: utf-8 -*-
import copick
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm


def main(config: DictConfig):
    root = copick.from_file(config["copick_config_path"])

    data_dicts = []
    for run in tqdm(root.runs):
        tomogram = (
            run.get_voxel_spacing(config["voxel_size"])
            .get_tomogram(config["tomo_type"])
            .numpy()
        )
        segmentation = run.get_segmentations(
            name=config["copick_segmentation_name"],
            user_id=config["copick_user_name"],
            is_multilabel=True,
        )[0].numpy()
        data_dicts.append({"image": tomogram, "label": segmentation})

    for num, data in enumerate(data_dicts):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.title("Tomogram")
        plt.imshow(data["image"][110], cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 1)
        plt.title("Painted Segmentation from Picks")
        plt.imshow(data["label"][110], cmap="viridis", alpha=0.5)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("generated_mask_" + str(num) + ".png")


if __name__ == "__main__":
    main()
