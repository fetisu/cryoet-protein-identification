# -*- coding: utf-8 -*-
import shutil
from collections import defaultdict

import copick
import copick_utils.writers.write as write
import numpy as np
from copick_utils.segmentation import segmentation_from_picks
from tqdm import tqdm
from utils import (
    copick_config_path,
    copick_segmentation_name,
    copick_user_name,
    destination_dir,
    particle_scales,
    source_dir,
    tomo_type,
)


def walk_through_train_data():
    for root, _dirs, files in source_dir.walk():
        relative_path = root.relative_to(source_dir)
        target_dir = destination_dir / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            if file.startswith("curation_0_"):
                new_filename = file
            else:
                new_filename = f"curation_0_{file}"

            source_file = root / file
            destination_file = destination_dir / relative_path / new_filename

            shutil.copy2(source_file, destination_file)


def generate_masks():
    root = copick.from_file(copick_config_path)
    target_objects = defaultdict(dict)

    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]["label"] = object.label
            target_objects[object.name]["radius"] = object.radius

    for run in tqdm(root.runs):
        print(run)
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()
        target = np.zeros(tomo.shape, dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(
                object_name=pickable_object.name, user_id="curation"
            )
            if len(pick):
                scale = particle_scales[pick[0].pickable_object_name]

                target = segmentation_from_picks.from_picks(
                    pick[0],
                    target,
                    target_objects[pickable_object.name]["radius"] * scale,
                    target_objects[pickable_object.name]["label"],
                )
        write.segmentation(
            run, target, copick_user_name, name=copick_segmentation_name
        )
