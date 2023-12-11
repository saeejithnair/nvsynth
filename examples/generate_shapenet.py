# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Dataset with online randomized scene generation for Instance Segmentation training.

Use OmniKit to generate a simple scene. At each iteration, the scene is populated by
adding assets from the user-specified classes with randomized pose and colour.
The camera position is also randomized before capturing groundtruth consisting of
an RGB rendered image, Tight 2D Bounding Boxes and Instance Segmentation masks.
"""

import glob
import os
import random
import signal
import sys

import carb
import numpy as np
import torch
from omni.isaac.kit import SimulationApp

# Setup default generation variables
# Value are (min, max) ranges
RANDOM_TRANSLATION_X = (-30.0, 30.0)
RANDOM_TRANSLATION_Z = (-30.0, 30.0)
RANDOM_ROTATION_Y = (0.0, 360.0)
SCALE = 20
CAMERA_DISTANCE = 300
BBOX_AREA_THRESH = 16
RESOLUTION = (1024, 1024)

# Default rendering parameters
RENDER_CONFIG = {
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 12,
    "headless": True,
}


class RandomObjects(torch.utils.data.IterableDataset):
    """Dataset of random ShapeNet objects.
    Objects are randomly chosen from selected categories and are positioned, rotated and coloured
    randomly in an empty room. RGB, BoundingBox2DTight and Instance Segmentation are captured by moving a
    camera aimed at the centre of the scene which is positioned at random at a fixed distance from the centre.

    This dataset is intended for use with ShapeNet but will function with any dataset of USD models
    structured as `root/category/**/*.usd. One note is that this is designed for assets without materials
    attached. This is to avoid requiring to compile MDLs and load textures while training.

    Args:
        categories (tuple of str): Tuple or list of categories. For ShapeNet, these will be the synset IDs.
        max_asset_size (int): Maximum asset file size that will be loaded. This prevents out of memory errors
            due to loading large meshes.
        num_assets_min (int): Minimum number of assets populated in the scene.
        num_assets_max (int): Maximum number of assets populated in the scene.
        split (float): Fraction of the USDs found to use for training.
        train (bool): If true, use the first training split and generate infinite random scenes.
    """

    def __init__(
        self,
        root,
        categories,
        max_asset_size=None,
        num_assets_min=3,
        num_assets_max=5,
        split=0.7,
        train=True,
    ):
        # assert len(categories) > 1
        assert (split > 0) and (split <= 1.0)

        self.kit = SimulationApp(RENDER_CONFIG)
        import omni.replicator.core as rep
        import warp as wp
        import warp.torch
        from omni.isaac.shapenet import utils
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        from foodverse.configs import usd_configs as uc

        self.rep = rep
        self.wp = wp
        self.stage = self.kit.context.get_stage()

        from omni.isaac.core.utils.nucleus import get_assets_root_path

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # If ShapeNet categories are specified with their names, convert to synset ID
        # Remove this if using with a different dataset than ShapeNet
        # category_ids = [utils.LABEL_TO_SYNSET.get(c, c) for c in categories]
        # Split string to list based on space
        self.categories = categories.split()
        self.range_num_assets = (num_assets_min, max(num_assets_min, num_assets_max))
        try:
            self.references = {
                key: value.path
                for (key, value) in uc.MODELS.items()
                if value.type == "food"
            }
            # {key: value for (key, value) in iterable}
            # self.references = self._find_usd_assets(root, category_ids, max_asset_size, split, train)
        except ValueError as err:
            carb.log_error(str(err))
            self.kit.close()
            sys.exit()
        self._setup_world()
        self.cur_idx = 0
        self.exiting = False

        signal.signal(signal.SIGINT, self._handle_exit)

    def _get_textures(self):
        return [
            self.assets_root_path
            + "/Isaac/Samples/DR/Materials/Textures/checkered.png",
            self.assets_root_path
            + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            self.assets_root_path
            + "/Isaac/Samples/DR/Materials/Textures/picture_a.png",
            self.assets_root_path
            + "/Isaac/Samples/DR/Materials/Textures/picture_b.png",
            self.assets_root_path
            + "/Isaac/Samples/DR/Materials/Textures/textured_wall.png",
            self.assets_root_path
            + "/Isaac/Samples/DR/Materials/Textures/checkered_color.png",
        ]

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    def _setup_world(self):
        import omni
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import set_stage_up_axis
        from pxr import UsdGeom

        """Setup lights, walls, floor, ceiling and camera"""
        # Set stage up axis to Y-up
        set_stage_up_axis("y")

        # In a practical setting, the room parameters should attempt to match those of the
        # target domain. Here, we instead opt for simplicity.
        create_prim(
            "/World/Room",
            "Sphere",
            attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]},
        )
        create_prim(
            "/World/Ground",
            "Cylinder",
            position=np.array([0.0, -0.5, 0.0]),
            orientation=euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True),
            attributes={
                "height": 1,
                "radius": 1e4,
                "primvars:displayColor": [(1.0, 1.0, 1.0)],
            },
        )
        create_prim("/World/Asset", "Xform")

        self.camera = self.rep.create.camera()
        self.render_product = self.rep.create.render_product(self.camera, RESOLUTION)
        self.viewport = omni.kit.viewport_legacy.get_default_viewport_window()

        # Setup annotators that will report groundtruth
        self.rgb = self.rep.AnnotatorRegistry.get_annotator("rgb")
        self.bbox_2d_tight = self.rep.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_tight"
        )
        self.instance_seg = self.rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation"
        )
        self.rgb.attach(self.render_product)
        self.bbox_2d_tight.attach(self.render_product)
        self.instance_seg.attach(self.render_product)

        self.kit.update()

        # Setup replicator graph
        self.setup_replicator()

    def _find_usd_assets(self, root, categories, max_asset_size, split, train=True):
        """Look for USD files under root/category for each category specified.
        For each category, generate a list of all USD files found and select
        assets up to split * len(num_assets) if `train=True`, otherwise select the
        remainder.
        """
        references = {}
        for category in categories:
            all_assets = glob.glob(
                os.path.join(root, category, "*/*.usd"), recursive=True
            )
            print(os.path.join(root, category, "*/*.usd"))
            # Filter out large files (which can prevent OOM errors during training)
            if max_asset_size is None:
                assets_filtered = all_assets
            else:
                assets_filtered = []
                for a in all_assets:
                    if os.stat(a).st_size > max_asset_size * 1e6:
                        print(
                            f"{a} skipped as it exceeded the max size {max_asset_size} MB."
                        )
                    else:
                        assets_filtered.append(a)

            num_assets = len(assets_filtered)
            if num_assets == 0:
                raise ValueError(
                    f"No USDs found for category {category} under max size {max_asset_size} MB."
                )

            if train:
                references[category] = assets_filtered[: int(num_assets * split)]
            else:
                references[category] = assets_filtered[int(num_assets * split) :]
        return references

    def _instantiate_category(self, category, references):
        with self.rep.randomizer.instantiate(references, size=1, mode="scene_instance"):
            self.rep.modify.semantics([("class", category)])
            self.rep.modify.pose(
                position=self.rep.distribution.uniform((-40, 5, -40), (40, 5, 40)),
                rotation=self.rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                # scale=self.rep.distribution.uniform(5, 50),
                scale=40,
            )
            # self.rep.randomizer.texture(self._get_textures(), project_uvw=True)

    def setup_replicator(self):
        """Setup the replicator graph with various attributes."""

        # Create two sphere lights
        light1 = self.rep.create.light(
            light_type="sphere", position=(-450, 350, 350), scale=100, intensity=30000.0
        )
        light2 = self.rep.create.light(
            light_type="sphere", position=(450, 350, 350), scale=100, intensity=30000.0
        )

        with self.rep.new_layer():
            with self.rep.trigger.on_frame():
                # Randomize light colors
                with self.rep.create.group([light1, light2]):
                    self.rep.modify.attribute(
                        "color",
                        self.rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)),
                    )

                # Randomize camera position
                with self.camera:
                    self.rep.modify.pose(
                        position=self.rep.distribution.uniform(
                            (100, 0, -100), (100, 100, 100)
                        ),
                        look_at=(0, 0, 0),
                    )

                # Randomize asset positions and textures
                for category, references in self.references.items():
                    self._instantiate_category(category, references)

    def __iter__(self):
        return self

    def __next__(self):
        from omni.isaac.core.utils.stage import is_stage_loading

        # Step - Randomize and render
        self.rep.orchestrator.step()

        # Collect Groundtruth
        gt = {
            "rgb": self.rgb.get_data(device="gpu"),
            "boundingBox2DTight": self.bbox_2d_tight.get_data(device="gpu"),
            "instanceSegmentation": self.instance_seg.get_data(device="gpu"),
        }

        # # RGB
        # # Drop alpha channel
        # image = self.wp.to_torch(gt["rgb"])[..., :3]

        # # Normalize between 0. and 1. and change order to channel-first.
        # image = image.float() / 255.0
        # image = image.permute(2, 0, 1)

        # # Bounding Box
        # gt_bbox = gt["boundingBox2DTight"]["data"]

        # # Create mapping from categories to index
        # bboxes = torch.tensor(gt_bbox[["x_min", "y_min", "x_max", "y_max"]].tolist(), device="cuda")
        # id_to_labels = gt["boundingBox2DTight"]["info"]["idToLabels"]
        # prim_paths = gt["boundingBox2DTight"]["info"]["primPaths"]

        # # For each bounding box, map semantic label to label index
        # cat_to_id = {cat: i + 1 for i, cat in enumerate(self.categories)}
        # semantic_labels_mapping = {int(k): v.get("class", "") for k, v in id_to_labels.items()}
        # semantic_labels = [cat_to_id[semantic_labels_mapping[i]] for i in gt_bbox["semanticId"]]
        # labels = torch.LongTensor(semantic_labels)

        # # Calculate bounding box area for each area
        # areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # # Identify invalid bounding boxes to filter final output
        # valid_areas = (areas > 0.0) * (areas < (image.shape[1] * image.shape[2]))

        # # Instance Segmentation
        # instance_data = self.wp.to_torch(gt["instanceSegmentation"]["data"]).squeeze()
        # path_to_instance_id = {v: int(k) for k, v in gt["instanceSegmentation"]["info"]["idToLabels"].items()}

        # instance_list = [im[0] for im in gt_bbox]
        # masks = torch.zeros((len(instance_list), *instance_data.shape), dtype=bool, device="cuda")

        # # Filter for the mask of each object
        # for i, prim_path in enumerate(prim_paths):
        #     # Merge child instances of prim_path as one instance
        #     for instance in path_to_instance_id:
        #         if prim_path in instance:
        #             masks[i] += torch.isin(instance_data, path_to_instance_id[instance])

        # target = {
        #     "boxes": bboxes[valid_areas],
        #     "labels": labels[valid_areas],
        #     "masks": masks[valid_areas],
        #     "image_id": torch.LongTensor([self.cur_idx]),
        #     "area": areas[valid_areas],
        #     "iscrowd": torch.BoolTensor([False] * len(bboxes[valid_areas])),  # Assume no crowds
        # }

        # self.cur_idx += 1
        # return image, target
        return (0, 0)


if __name__ == "__main__":
    "Typical usage"
    import argparse

    import matplotlib.pyplot as plt

    from foodverse.configs import usd_configs as uc

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        required=False,
        help="List of object classes to use",
    )
    parser.add_argument(
        "--max_asset_size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument(
        "--num_test_images",
        type=int,
        default=10,
        help="number of test images to generate when executing main",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing USDs. If not specified, use {SHAPENET_LOCAL_DIR}_mat as root.",
    )
    args, unknown_args = parser.parse_known_args()

    # # If root is not specified use the environment variable SHAPENET_LOCAL_DIR with the _mat suffix as root
    # if args.root is None:
    #     if "SHAPENET_LOCAL_DIR" in os.environ:
    #         shapenet_local_dir = f"{os.path.abspath(os.environ['SHAPENET_LOCAL_DIR'])}_mat"
    #         if os.path.exists(shapenet_local_dir):
    #             args.root = shapenet_local_dir
    #     if args.root is None:
    #         print(
    #             "root argument not specified and SHAPENET_LOCAL_DIR environment variable was not set or the path did not exist"
    #         )
    #         exit()

    categories = [key for (key, value) in uc.MODELS.items() if value.type == "food"]
    args.categories = " ".join(categories)
    print(args.categories)
    dataset = RandomObjects(
        args.root, args.categories, max_asset_size=args.max_asset_size
    )
    from omni.isaac.shapenet import utils
    from omni.isaac.synthetic_utils import visualization

    # categories = [utils.LABEL_TO_SYNSET.get(c, c) for c in args.categories]
    # Iterate through dataset and visualize the output
    plt.ion()
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.tight_layout()

    image_num = 0
    for image, target in dataset:
        pass
        # for ax in axes:
        #     ax.clear()
        #     ax.axis("off")

        # np_image = image.permute(1, 2, 0).cpu().numpy()
        # axes[0].imshow(np_image)

        # num_instances = len(target["boxes"])
        # colours = visualization.random_colours(num_instances)
        # overlay = np.zeros_like(np_image)
        # for mask, colour in zip(target["masks"].cpu().numpy(), colours):
        #     overlay[mask, :3] = colour

        # axes[1].imshow(overlay)
        # mapping = {i + 1: cat for i, cat in enumerate(categories)}
        # labels = [mapping[label.item()] for label in target["labels"]]
        # visualization.plot_boxes(ax, target["boxes"].tolist(), labels=labels, colours=colours)

        # plt.draw()
        # plt.pause(0.01)
        # fig_name = "domain_randomization_test_image_" + str(image_num) + ".png"
        # plt.savefig(fig_name)
        # image_num += 1
        if dataset.exiting or (image_num >= args.num_test_images):
            break

    # cleanup
    dataset.kit.close()
