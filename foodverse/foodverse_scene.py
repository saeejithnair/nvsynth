import glob
import os
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.graph.core as og
import omni.isaac.core as isaac_core
import omni.isaac.core.utils.bounds as bounds
import omni.replicator.core as rep
from omni.isaac.kit import SimulationApp
from PIL import Image, ImageEnhance
from pxr import Gf, Usd

from foodverse.configs import sim_configs as sc
from foodverse.configs import usd_configs as uc
from foodverse.configs.scene_configs import (
    CartesianPosition,
    FoodItemConfig,
    FoodverseSceneConfig,
    PoseConfig,
    SceneItems,
)
from foodverse.scene import Scene
from foodverse.utils import dataset_utils as du
from foodverse.utils import file_utils as fu
from foodverse.utils import geometry_utils as gu
from foodverse.writer import FoodverseWriter

FOOD_PRIM_PATH = r"\/Replicator\/Ref_Xform.*\/Ref"


class FoodverseScene(Scene):
    def __init__(self, kit: SimulationApp, config: FoodverseSceneConfig) -> None:
        super().__init__(kit)

        self.models = uc.MODELS
        self.food_models = [
            model for model in self.models.values() if model.type == "food"
        ]
        # Sort by uid to ensure that the order is deterministic.
        self.food_models.sort(key=lambda x: x.uid)

        self.cur_scene_idx = config.scene_start_idx
        self.root_output_dir = config.root_output_dir
        self.plate_scale = config.plate_scale
        self.num_cameras = config.num_cameras
        self.num_scenes = config.num_scenes

        # Set the seed for the random number generator.
        # If no seed is provided, use the current time.
        self.cur_time = int(time.time())
        self.seed = self.cur_time if config.seed is None else config.seed

        # Load the scene items config.
        if isinstance(config.scene_items, SceneItems):
            self.scene_items = config.scene_items
        elif isinstance(config.scene_items, Path):
            self.scene_items = SceneItems.from_yaml(config.scene_items)


        # Set up static elements of the world scene.
        self._setup_world()

        self.initialize_orchestrator()

        self.exiting = False

        # Register exit signal handler to capture Ctrl+C.
        signal.signal(signal.SIGINT, self._handle_exit)

        # Set the random seed for the random number generator after everything
        # has been initialized. We mainly want to ensure that the scene
        # generation is randomized. So setting the seed here overrides any
        # previously set seeds (e.g. during color palette generation).
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _handle_exit(self, *args, **kwargs) -> None:
        """Handles exit signal by setting the exiting flag to True.

        Returns:
            None
        """
        print(
            f"Exiting dataset generation after {self.cur_scene_idx} "
            f"scenes generated."
        )
        self.exiting = True

    def create_scene(self) -> None:
        """Loads the room scene and enables physics on the table."""
        scene_cfg = self.models["simple_room"]
        scene = rep.create.from_usd(scene_cfg.usd_path, semantics=[("class", "scene")])

        with scene:
            rep.modify.pose(position=(0, 0, 0))

        with rep.get.prims("table_low$"):
            rep.physics.collider(approximation_shape="none")

        table_prim = self.get_prims("table_low$")

        assert len(table_prim) == 1
        table_prim = table_prim[0]
        self.apply_physics_to_prim(table_prim, approximation_shape="none")

    def load_plate(self, plate_scale: float = sc.PLATE_SCALE) -> None:
        """Loads the plate onto the table and enables static collision on it.

        Args:
            plate_scale: The scale factor for the plate.

        Returns:
            None
        """
        plate_cfg = self.models["plate"]

        # Create a plate prim.
        prim_name = du.compose_prim_name(plate_cfg.label)
        prim_path = du.compose_prim_path(self.scope_name, prim_name)
        plate_prim = isaac_core.utils.prims.create_prim(
            prim_path=prim_path,
            usd_path=plate_cfg.usd_path,
            semantic_label="plate",
            semantic_type="class",
            scale=np.array([plate_scale, plate_scale, plate_scale]),
        )

        # Set plate prim to be a static collider and set material properties.
        self.apply_physics_to_prim(plate_prim, approximation_shape="none")
        self.apply_material_to_prim(plate_prim, sc.PLATE_MATERIAL)

        # Store reference to the plate prim.
        self.plate_prim = plate_prim

        # Compute the bounding box of the plate prim. Since the plate prim is
        # static, we can compute the bounding box once and store it.
        self.plate_bbox = gu.BBox(self.bb_cache.ComputeWorldBound(self.plate_prim))

    def _setup_world(self) -> None:
        """Setup the world for data generation.
        This function is called from the constructor and sets up the
        static attributes in the scene.

        Args:
            None

        Returns:
            None
        """

        # Create bounding box cache for faster retrieval of prim bounds.
        self.bb_cache = bounds.create_bbox_cache()

        # TODO(snair): Check if it's necessary to wrap this in a new layer.
        with rep.new_layer(name="StaticLayer"):
            # Load the room.
            self.create_scene()

            # Load the plate.
            self.load_plate(plate_scale=self.plate_scale)

        # Create container to store loaded food prims.
        self.food_prims = []
        self.food_rigid_prims = []
        self.food_prim_names = []

        # Setup cameras.
        self.render_products = self.create_render_products(
            look_at_prim=self.plate_prim,
            pos=(0, 2, 2),
            num_cameras=self.num_cameras,
            radius=2.0,
        )

        # Setup writer.
        self.output_dir = os.path.join(self.root_output_dir, f"_fv_output_{self.seed}")
        self.writer = self._setup_writer(self.output_dir, self.render_products)
        self.output_semantic_mappings(uc.SEMANTIC_MAPPINGS)

    def output_semantic_mappings(
        self, semantic_mappings: Dict[str, uc.SemanticConfig]
    ) -> None:
        """Outputs the semantic mappings to the output directory.

        Args:
            semantic_mappings: The semantic mappings to output.

        Returns:
            None
        """
        file_basename = "semantic_mappings"
        serialized_semantic_mappings = {}
        for label, semantic_config in semantic_mappings.items():
            serialized_semantic_mappings[label] = {
                "class": semantic_config.class_name,
                "semantic_id": semantic_config.semantic_id,
                "rgba": tuple(semantic_config.rgba.tolist()),
            }

        fu.write_dict_to_json_backend(
            serialized_semantic_mappings, self.backend, file_basename
        )

    def _setup_writer(
        self, output_dir: str, render_products: List[og.Node]
    ) -> FoodverseWriter:
        """Creates and initializes a BasicWriter for data generation.

        Args:
            output_dir: The directory to write the output to.
            render_products: The list of render products (cameras) which collect
                data from various perspectives.

        Returns:
            writer: The initialized writer.
        """
        rep.settings.carb_settings(
            "/omni/replicator/render/write_threads", sc.WRITE_THREADS
        )

        self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        writer = FoodverseWriter(
            output_dir=output_dir,
            backend=self.backend,
            render_products=render_products,
            semantic_types=["class"],
            rgb=True,
            bounding_box_2d_loose=False,
            semantic_segmentation=True,
            instance_segmentation=True,
            instance_id_segmentation=False,
            bounding_box_3d=True,
            occlusion=False,
            camera_params=True,
            distance_to_camera=False,
            normals=False,
            amodal_segmentation=False,
        )

        return writer

    def _setup_replicator(self) -> None:
        """Setup the replicator graph with randomizations.

        Args:
            None

        Returns:
            None
        """
        with rep.new_layer(name="RandomizedLayer"):
            with rep.trigger.on_frame():
                # TODO(snair): Randomize the lighting.
                pass

    def create_temp_assets_dir(self, scene_idx: int) -> str:
        """Creates a temporary directory to store assets in.
        NOTE: This does not use the tempfile module because we want to
        sustain the dir for longer contexts. The caller is responsible
        for cleaning up the directory.

        Args:
            scene_idx: The index of the scene being generated.

        Returns:
            The path to the temporary directory.
        """
        folder_path = "/tmp"
        folder_name = f"assets_{self.cur_time}_{scene_idx}"
        temp_dir_path = fu.create_new_folder(folder_path, folder_name)

        return temp_dir_path

    def augment_food_model_brightness(
        self, food_usd_path: str, temp_assets_dir: str, item_idx: int
    ) -> str:
        """Augments brightness of the food model by changing the texture file.
        To avoid modifying the original USD file, the augmented food model is
        stored in a temporary directory.

        The following directory structure is assumed:
        dataset_root/
        ├── id-26-chicken-leg-133g/
        │   ├── textured_obj.usd
        │   ├── textures/
        │   │   ├── [textured|poly]_0_x2sOzXzQ.jpg
        │   │   ├── textured_0_roughness_x2sOzXzQ.jpg [Optional, unused]
        │   ├── ... [Other unused files (.jpg/.mtl/.obj/.ply/.usda/etc)]
        ├── id-27-chicken-leg-77g/
        │   ├── ...

        Args:
            food_usd_path: Path to the USD for the food model.
            temp_assets_dir: Path to the temporary directory to store assets.
            item_idx: The index of the food item being augmented. Should be
                unique or else the augmented texture file for an item will
                be overwritten.

        Returns:
            The path to the augmented USD file.
        """
        # Ensure that the temporary directory exists.
        assert os.path.exists(
            temp_assets_dir
        ), "Temp assets dir {temp_assets_dir} does not exist."

        # Food model dirname will be something like
        # "/path_to_dataset_root/id-26-chicken-leg-133g".
        food_model_dirname = os.path.dirname(food_usd_path)

        # Food model basename will be something like "id-26-chicken-leg-133g".
        food_model_basename = os.path.basename(food_model_dirname)

        # Create a temporary dir in the temp assets dir to store the augmented
        # food model. e.g. "/tmp/assets_1677740401_0/id-26-chicken-leg-133g_0".
        tmp_food_dir_path = os.path.join(
            temp_assets_dir, f"{food_model_basename}_{item_idx}"
        )

        # Copy the whole food model directory (containing the USD and texture
        # files) to the temporary directory.
        shutil.copytree(food_model_dirname, tmp_food_dir_path)

        # Glob for path to the texture files in the temp food model directory.
        texture_file_paths = glob.glob(
            os.path.join(tmp_food_dir_path, "textures/*.jpg")
        )
        # Filter out the roughness texture files.
        texture_file_paths = list(
            filter(lambda x: "roughness" not in x, texture_file_paths)
        )

        # Expect exactly 1 texture file after filtering. If not, raise an error.
        if len(texture_file_paths) != 1:
            raise ValueError(
                f"Expected to find 1 texture file in {tmp_food_dir_path}, "
                f"but found {len(texture_file_paths)}"
            )

        texture_file_path = texture_file_paths[0]

        # Augment brightness of texture file randomly in the range [1.0, 2.0].
        img = Image.open(texture_file_path).convert("RGB")
        img_enhancer = ImageEnhance.Brightness(img)
        factor = np.random.uniform(low=1.0, high=2.0)
        enhanced_output = img_enhancer.enhance(factor)

        # Overwrite the existing (temp) texture file with the augmented version.
        enhanced_output.save(texture_file_path)

        # Return the path to the USD for the augmented food model.
        tmp_usd_path = os.path.join(tmp_food_dir_path, os.path.basename(food_usd_path))
        return tmp_usd_path

    def generate_plate_scene_random_foods(
        self, size: int, temp_dir_path: str, max_retries: int = 5
    ) -> None:
        """Keeps trying to generate the plate scene with random food items
        up to a maximum number of retries.

        If the plate is empty after a scene generation attempt,
        the temporary directory is removed and we make another attempt
        at generating the scene. Plate might be empty in rare cases where
        the food items collide with each other, knocking them all off the
        plate. This can occur due to a combination of unlucky randomization
        of initial food poses, compounded by the instability of the physics
        simulation. Thus, instead of trying to fix the physics simulation,
        or generate a "perfect random scene", we simply retry reloading the
        plate.

        Args:
            size: The maximum number of food items to load.
            temp_dir_path: The path to the temporary directory to store assets.
            max_retries: The maximum number of retries to reload the plate.

        Returns:
            None.
        """
        for _ in range(max_retries):
            plate_is_empty = self.attempt_scene_generation_random_foods(
                size, temp_dir_path
            )
            if plate_is_empty:
                # If plate is empty, remove the temp folder and try again.
                fu.remove_folder_if_exists(temp_dir_path)
                temp_dir_path = self.create_temp_assets_dir(
                    scene_idx=self.cur_scene_idx
                )
            else:
                # If plate has at least one food item, we're done.
                break

    def remove_all_food_prims(self) -> None:
        """Removes all food prims from the world scene and internal references."""
        # Clear references to existing food rigid body prims.
        self.food_rigid_prims.clear()

        # Remove existing food prims from the scene graph.
        for i in range(len(self.food_prim_names)):
            self.world.scene.remove_object(name=self.food_prim_names[i])

        # Clear references to existing food prims.
        self.food_prims.clear()
        self.food_prim_names.clear()

    def load_food_model_into_scene(
        self,
        food_model: uc.ModelConfig,
        prim_name: str,
        pose: PoseConfig,
        augmented_food_usd_path: Optional[str] = None,
        scale: Optional[float] = None,
        static: bool = False
    ) -> Tuple[Usd.Prim, Optional[isaac_core.prims.RigidPrim]]:
        """Loads a food model into the scene.

        The food model is loaded into the scene as a prim with the given
        name. Physics properties are also set for the food prim, enabling
        it as a rigid body collider.

        Args:
            food_model: The food model to load.
            prim_name: A unique name to assign to the prim that will be loaded
                (e.g. "id_94_chicken_wing_27g_1").
            position_2d: The 2D position (x, y) to load the food model at.
                Height (z) is determined based on plate height and food model
                dimensions.
            augmented_food_usd_path: The path to the USD file for the augmented
                food model. If None, the default USD path from the food model
                config will be used.

        Returns:
            A tuple containing the food prim and the rigid prim wrapper.
        """
        food_usd_path = (
            food_model.usd_path
            if augmented_food_usd_path is None
            else augmented_food_usd_path
        )

        # 0.01 is a qualitatively derived heuristic to make the food mesh
        # sizes look realistic in comparison to the plate size.
        scale = scale * 0.01 if scale else food_model.scale * 0.01
        orientation = pose.orientation.as_list(radians=True) if pose.orientation is not None else [
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi),
        ]

        position = pose.position.as_cartesian()

        prim_path = du.compose_prim_path(self.scope_name, prim_name)
        create_prim_kwargs = {}
        if static:
            assert position.z is not None, "Static food items must have a z position"
            create_prim_kwargs.update({
                "position": Gf.Vec3d(position.x, position.y, self.plate_bbox.z_dim + position.z),
                "orientation": isaac_core.utils.rotations.euler_angles_to_quat(orientation),
                "scale": Gf.Vec3d(scale, scale, scale),
            })

        food_prim = isaac_core.utils.prims.create_prim(
            prim_path=prim_path,
            usd_path=food_usd_path,
            semantic_label=f"{food_model.label}",
            semantic_type="class",
            **create_prim_kwargs,
        )

        # Set physics properties for the food prim.
        self.apply_physics_to_prim(food_prim, approximation_shape="convexDecomposition")
        self.apply_material_to_prim(food_prim, sc.FOOD_MATERIAL)

        if static:
            return (food_prim, None)

        # Wrap the food prim in a rigid prim to enable dynamic collisions.
        food_rigid_prim = isaac_core.prims.RigidPrim(
            prim_path=str(food_prim.GetPrimPath()),
            name=prim_name,
            scale=Gf.Vec3d(scale, scale, scale),
            orientation=isaac_core.utils.rotations.euler_angles_to_quat(orientation),
            linear_velocity=np.array([0, 0, 0]),
            angular_velocity=np.random.normal([0, 0, 0], [0, 0, 0]),
        )
        food_rigid_prim.enable_rigid_body_physics()

        # Add the food prim to the world scene (this is required for the
        # physics handles to propagate correctly on reset).
        self.world.scene.add(food_rigid_prim)

        food_bbox = gu.BBox(self.bb_cache.ComputeWorldBound(food_prim))
        x_pos, y_pos = position.x, position.y

        # Validate that the food model fits within the plate based on its
        # bounding box dimensions and given position.
        if x_pos + food_bbox.x_dim > self.plate_bbox.radius:
            diff = x_pos + food_bbox.x_dim - self.plate_bbox.radius
            x_pos -= diff

        if y_pos + food_bbox.y_dim > self.plate_bbox.radius:
            diff = y_pos + food_bbox.y_dim - self.plate_bbox.radius
            y_pos -= diff

        # The height (z_pos) of the food prim must take into account the random
        # orientation of the food prim. Instead of finding the magnitude of the
        # rotated prim's projection onto the z-axis, we take the lazy approach
        # and simply use the largest dimension of the food prim's bounding box.
        # This height must also be offset by the plate height (which is
        # guaranteed to be oriented upright along the z-axis).
        z_pos = self.plate_bbox.z_dim + food_bbox.largest_dim
        position = Gf.Vec3d(x_pos, y_pos, z_pos)

        # Set the default position of the food prim. This will be the position
        # that the food prim is reset to on each world reset.
        food_rigid_prim.set_default_state(position=position)

        return (food_prim, food_rigid_prim)

    def step_until_settled(
        self,
        rigid_prims: List[isaac_core.prims.RigidPrim],
        item_timeout: float = sc.SIM_ITEM_TIMEOUT_PERIOD_SECS,
        velocity_threshold: float = sc.SIM_ITEM_SETTLE_VELOCITY,
    ) -> List[int]:
        """Keeps stepping the physics simulation until the given rigid prims
        have stopped moving or a timeout has exceeded.

        A prim may fail to settle due to instabilities in the physics
        simulation, or due to extreme collisions that knock the prim off the
        plate (and maybe onto the table or ground).

        Args:
            rigid_prims: The rigid prims to wait for. They should already be
                added to the world scene.

        Returns:
            A list of indices of the rigid prims that failed to settle before
            the timeout exceeded. The indices correspond to the indices of the
            rigid prims in the given list. If the list is empty, all rigid
            prims have settled.
        """
        food_has_settled = False
        indices_in_motion = []
        cur_idx = 0

        # Measure start time.
        start_time = time.time()
        while not food_has_settled:
            self.world.step(render=False)
            vel = np.linalg.norm(rigid_prims[cur_idx].get_linear_velocity())

            if vel < velocity_threshold:
                # If the velocity of food item is below threshold, it has
                # stopped moving. Check the next food item.
                cur_idx += 1
                # Reset start time.
                start_time = time.time()
            elif time.time() - start_time > item_timeout:
                # If we've waited too long for this food item, move on
                # to the next one to prevent the simulation from hanging.
                indices_in_motion.append(cur_idx)
                cur_idx += 1
                # Reset start time.
                start_time = time.time()

            if cur_idx == len(rigid_prims):
                food_has_settled = True

        return indices_in_motion

    def remove_invalid_food_items(self, indices_in_motion: List[int]) -> bool:
        """Removes the food items that failed to settle from the world scene,
        and any items that landed outside the bounds of the plate.

        Args:
            indices_in_motion: The indices of the food prims that failed to
                settle.

        Returns:
            True if there are zero valid food items remaining in the scene.
            False otherwise.
        """
        # Clear bounding box cache, otherwise the initial bounding box
        # pose will be used instead of returning the current pose.
        self.bb_cache.Clear()

        indices_to_remove = indices_in_motion.copy()

        for idx, food_prim in enumerate(self.food_prims):
            if idx in indices_to_remove:
                # If we already decided to remove this food item,
                # don't bother checking if it's outside the plate.
                continue

            # Compute current bounding box of food prim.
            food_bbox = gu.BBox(self.bb_cache.ComputeWorldBound(food_prim))

            # Remove food that is outside of the plate.
            if not (
                gu.point_within_circle(
                    self.plate_bbox.radius, food_bbox.min, radius_scale=1.3
                )
                and gu.point_within_circle(
                    self.plate_bbox.radius, food_bbox.max, radius_scale=1.3
                )
                and gu.point_within_circle(
                    self.plate_bbox.radius, food_bbox.centroid, radius_scale=0.9
                )
            ):
                indices_to_remove.append(idx)

        # Remove all invalid items.
        for idx in sorted(indices_to_remove, reverse=True):
            self.world.scene.remove_object(name=self.food_prim_names[idx])
            self.food_rigid_prims.pop(idx)
            self.food_prims.pop(idx)
            self.food_prim_names.pop(idx)

        num_valid_left = len(self.food_rigid_prims)
        self.bb_cache.Clear()

        return num_valid_left == 0

    def drop_food_onto_plate(self) -> bool:
        """Runs a physics simulation to let the food items settle on the plate.

        Returns:
            True if there are zero valid food items remaining in the scene.
        """
        # Resets world to ensure physics handles get propagated. Also
        # resets the food items to their default positions.
        self.world.reset()

        # Run simulation to let food items settle on plate.
        indices_in_motion = self.step_until_settled(self.food_rigid_prims)

        # Remove food items that failed to settle or ended out of bounds.
        plate_is_empty = self.remove_invalid_food_items(indices_in_motion)

        # Render the scene so that it shows up correctly in the viewports.
        self.world.render()

        return plate_is_empty

    def attempt_scene_generation_random_foods(
        self, size: int, temp_dir_path: str
    ) -> bool:
        """Loads new random food items on the plate with a random pose.

        A physics simulation is run to let the food items settle on the plate
        in a realistic manner.

        Args:
            size: The maximum number of food items to load onto the plate.
                Note that the actual number of food items that appear on the
                plate may be less than this number, due to collisions knocking
                food items off the plate.
            temp_dir_path: The path to the temporary directory to store assets.

        Returns:
            True if the plate is empty after loading the new food items.
        """
        # Clear existing food prims from the scene and our internal references.
        self.remove_all_food_prims()

        # Used for computing the ideal position for each food item. Plate is
        # divided into equal sized sectors.
        delta_theta = 2 * np.pi / size
        theta = 0
        r = self.plate_bbox.radius * 0.5

        # Load food items into the scene.
        for i in range(size):
            # Get a random food item from the dataset.
            food_model = random.choice(self.food_models)

            # A scene can have multiple copies of the same food item. To
            # distinguish between them, we append the index of the food item
            # to get a unique prim name.
            prim_id = i + 1  # Force ID to be 1-indexed.
            prim_name = du.compose_prim_name(model_label=food_model.label, id=prim_id)

            # Augment the food model appearance by changing brightness.
            # This allows us to generate more diverse scenes and simulate
            # food scenes that may be over/under exposed.
            tmp_food_usd_path = self.augment_food_model_brightness(
                food_model.usd_path, temp_dir_path, item_idx=i
            )

            # Compute ideal position for current food item. Even though the
            # scene is randomized, we want to ensure that the food items
            # are placed in a way that minimizes crowding as this will increase
            # the likelihood of extreme collisions. Thus, we cut the plate into
            # sectors and place a food item in the center of each sector.
            x_pos = r * np.cos(theta)
            y_pos = r * np.sin(theta)
            theta += delta_theta

            # Load the food item into the scene.
            pose = PoseConfig(position=CartesianPosition(x=x_pos, y=y_pos))
            food_prim, food_rigid_prim = self.load_food_model_into_scene(
                food_model,
                prim_name,
                pose=pose,
                augmented_food_usd_path=tmp_food_usd_path,
            )

            self.food_prims.append(food_prim)
            self.food_prim_names.append(prim_name)
            self.food_rigid_prims.append(food_rigid_prim)

        # Run simulation to let food items settle on plate.
        plate_is_empty = self.drop_food_onto_plate()

        return plate_is_empty

    def change_prim_visibility(self, prim: Usd.Prim, visible: bool) -> None:
        """Changes the visibility of a prim.

        Args:
            prim: The prim to change visibility for.
            visible: True if the prim should be visible.
        """

        isaac_core.utils.prims.set_prim_visibility(prim, visible)
        self.world.render()

    def change_visibility_assets(
        self, visible: bool, idx: Optional[int] = None
    ) -> None:
        """Changes the visibility of the assets in the scene.

        Args:
            visible: True if the assets should be visible.
            idx: The id of the asset to change visibility for. If None, all
                assets will be changed.
        """
        if idx is None:
            # Change visibility of all assets.
            for food_prim in self.food_prims:
                self.change_prim_visibility(food_prim, visible)
        else:
            # Change visibility of a specific asset.
            food_prim = self.food_prims[idx]
            self.change_prim_visibility(food_prim, visible)

    def collect_amodal_masks(self) -> None:
        """
        Generates amodal masks for all assets in the scene.
        Outputs the masks via the writer.

        Returns:
            None
        """
        # Set all assets to be invisible.
        self.change_visibility_assets(visible=False)
        rep.orchestrator.step()

        for idx, food_prim in enumerate(self.food_prims):
            # Set the current asset to be visible.
            prim_path = str(food_prim.GetPath())
            self.change_prim_visibility(food_prim, visible=True)
            rep.orchestrator.step()
            self.writer.write_amodal(
                self.cur_scene_idx,
                prim_path,
                instance_segmentation=True,
                normals=True,
                distance_to_camera=True,
            )

            # Set the current asset to be invisible.
            self.change_prim_visibility(food_prim, visible=False)
            rep.orchestrator.step()

    def generate_static_procedural_scene(self, capture_placement_every_n_items: int) -> None:
        # Write the camera parameters to dataset folder.
        self.writer.write_camera_params()

        prim_names = []
        num_rules = len(self.scene_items.procedural_items)
        for rule_idx, placement_rule in enumerate(self.scene_items.procedural_items):
            food_items = placement_rule.generate_food_items()
            for i, item in enumerate(food_items):
                model_config = self.models[item.model_label]
                item_idx = rule_idx*num_rules + i
                prim_name = du.compose_prim_name(model_label=item.model_label, id=item_idx)
                scale = item.scale if item.scale is not None else model_config.scale

                # Load the food model into the scene at the specified pose
                food_prim, food_rigid_prim = self.load_food_model_into_scene(
                    food_model=model_config,
                    prim_name=prim_name,
                    pose=item.pose,
                    augmented_food_usd_path=None, # Assuming no augmentation for static scenes
                    scale=scale,
                    static=item.static,
                )
                prim_names.append(prim_name)
                assert food_rigid_prim is None, "Static food items must be static"

                if item_idx % capture_placement_every_n_items == 0:
                    rep.orchestrator.step()
                    self.writer.write_data(prim_names_to_expect=prim_names)



    def generate_persistent_food_items(self, food_items: Optional[List[FoodItemConfig]] = None, capture_falling_every_n_steps: Optional[int] = None) -> None:
        """Generates a static scene with predefined food items.

        Returns:
            None
        """
        # Write the camera parameters to dataset folder.
        self.writer.write_camera_params()

        prim_names = []
        food_rigid_prims = []
        if food_items is None:
            food_items = self.scene_items.food_items

        for i, item in enumerate(food_items):
            model_config = self.models[item.model_label]
            prim_name = du.compose_prim_name(model_label=item.model_label, id=i)
            scale = item.scale if item.scale is not None else model_config.scale

            # Load the food model into the scene at the specified pose
            food_prim, food_rigid_prim = self.load_food_model_into_scene(
                food_model=model_config,
                prim_name=prim_name,
                pose=item.pose,
                augmented_food_usd_path=None, # Assuming no augmentation for static scenes
                scale=scale,
                static=item.static,
            )
            prim_names.append(prim_name)
            if food_rigid_prim is not None:
                food_rigid_prims.append(food_rigid_prim)

        if len(food_rigid_prims) > 0:
            # Run simulation to let food items settle on plate.
            self.world.reset()

            if capture_falling_every_n_steps is None:
                # Run simulation to let food items settle on plate.
                self.step_until_settled(food_rigid_prims)
                # Render the scene so that it shows up correctly in the viewports.
                self.world.render()
            else:
                MAX_STEPS = 150
                for i in range(MAX_STEPS):
                    self.world.step(render=False)
                    if i % capture_falling_every_n_steps == 0:
                        self.world.render()
                        self.writer.write_data(prim_names_to_expect=prim_names)


        rep.orchestrator.step()
        self.writer.write_data(prim_names_to_expect=prim_names)

    def generate_dataset(self, num_scenes: Optional[int] = None):
        """Generates a dataset of food scenes.

        Args:
            num_scenes: The number of scenes to generate.
        """
        if num_scenes is None:
            num_scenes = self.num_scenes

        # Reset the scene counter.
        self.cur_scene_idx = 0

        # Write the camera parameters to dataset folder.
        self.writer.write_camera_params()

        while True:
            temp_dir_path = self.create_temp_assets_dir(scene_idx=self.cur_scene_idx)
            self.generate_plate_scene_random_foods(
                size=sc.MAX_ITEMS_PER_SCENE, temp_dir_path=temp_dir_path
            )

            # Step - Randomize and render
            rep.orchestrator.step()

            try:
                self.writer.write_data(prim_names_to_expect=self.food_prim_names)
                self.collect_amodal_masks()
            except:
                # If an exception occurs, we still want to continue generating
                # scenes.
                pass

            self.cur_scene_idx += 1
            fu.remove_folder_if_exists(temp_dir_path)

            if self.cur_scene_idx >= num_scenes or self.exiting:
                break
