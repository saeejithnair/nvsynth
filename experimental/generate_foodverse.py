"""
Dataset with online randomized scene generation for segmentation training.

Uses OmniKit to generate a simple scene. At each iteration, the scene is populated by adding assets from the Foodverse dataset to the scene. The pose for each food model is randomized and placed onto a plate, which lies on a table. Cameras are placed around the scene to capture the scene from different angles and the lighting is also randomized. The groundtruth consisting of an RGB rendered image, tight 2D bounding boxes, semantic and instance segmentation masks are saved to disk.

"""

from omni.isaac.kit import SimulationApp

import os
import glob
import torch
import random
import re
import numpy as np
import signal
import sys
import carb
import time
import tempfile, shutil
from PIL import Image, ImageEnhance

RENDER_CONFIG = {
    "renderer": "PathTracing", # Can also be "RayTracedLightning"
    # "renderer": "RayTracedLightning", # Can also be "RayTracedLightning"
    #  The number of samples to render per frame, increase for improved quality, used for `PathTracing` only.
    "samples_per_pixel_per_frame": 12,
    "headless": True,
    "multi_gpu": False,
}
FOOD_PRIM_PATH = r"\/Replicator\/Ref_Xform.*\/Ref"

class FoodverseObjects(torch.utils.data.IterableDataset):
    """Dataset of random Foodverse objects.
    Objects are randomly chosen from the Foodverse dataset and dropped onto a plate.
    """
    def __init__(self) -> None:
        super().__init__()

        # SimulationApp helps launch Omniverse Toolkit.
        # This must be executed before any other OV imports.
        self.kit = SimulationApp(RENDER_CONFIG)

        # SimulationContext provides functions that take care of many
        # time-related events such as performing a physics or render step.
        # from omni.isaac.core import SimulationContext
        # self.sim_context = SimulationContext()
        from omni.isaac.core import World
        self.world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)

        import omni.replicator.core as rep
        import warp as wp
        import warp.torch
        import omni.isaac.core as isaac_core
        import omni.usd as omni_usd

        self.rep = rep
        self.wp = wp
        self.isaac_core = isaac_core
        self.stage = self.kit.context.get_stage()
        self.scope_name = "/MyScope"
        self.omni_usd = omni_usd

        # Need to bind these imports to the class instance because they can
        # only be imported after SimulationApp has been launched.
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdShade
        self.pxr_usd_physics = UsdPhysics
        self.pxr_physx_schema = PhysxSchema
        self.pxr_gf = Gf
        self.pxr_usd_geom = UsdGeom
        self.pxr_usd_shade = UsdShade

        from omni.isaac.core.utils.nucleus import get_assets_root_path

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        # TODO(snair): Add support for loading USD models, this corresponds to categories in ShapeNet script.
        from foodverse.configs import usd_configs as uc
        self.models = uc.MODELS
        self.food_models = [model for model in self.models.values() if model.type is "food"]
        # Sort by uid to ensure that the order is deterministic.
        self.food_models.sort(key=lambda x: x.uid)

        self.plate_models = [model for model in self.models.values() if model.type is "plate"]

        scope = self.pxr_usd_geom.Scope.Define(self.stage, self.scope_name)
        
        self.num_scenes_generated = 0

        self.cur_time = int(time.time())
        # self.cur_time = 1677740401
        random.seed(self.cur_time)
        np.random.seed(self.cur_time)

        self.cur_food_model = 0

        # Set up static elements of the world scene.
        self._setup_world()
        
        # Used to determine if data generation should be stopped based on interrupt signal.
        self.exiting = False

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print(f"Exiting dataset generation after {self.num_scenes_generated} scenes generated.")
        self.exiting = True

    def cleanup(self):
        self.rep.BackendDispatch.wait_until_done()
        self.rep.orchestrator.stop()
        self.kit.update()
        self.kit.close()

    def create_scene(self):
        scene_cfg = self.models["simple_room"]
        scene = self.rep.create.from_usd(
                    scene_cfg.usd_path,
                    semantics=[('class', 'scene')]
        )

        with scene:
            self.rep.modify.pose(position=(0,0,0))

        with self.rep.get.prims("table_low$"):
            self.rep.physics.collider(approximation_shape="none")

        table_prim = self.get_prims("table_low$")

        assert len(table_prim) == 1
        table_prim = table_prim[0]
        self.apply_physics_to_prim(table_prim, approximation_shape="none")
    
    def generate_plate_plane(self, plate_prim):
        """Creates an invisible plane that is used to drop objects onto the
        plate.
        
        """
        plate_bb_cache = self.isaac_core.utils.bounds.create_bbox_cache()
        # Compute the oriented bounding box of the plate.
        bbox3d_gf = plate_bb_cache.ComputeLocalBound(plate_prim)
        
        # Get the transformation matrix for the plate in world frame.
        plate_tf_gf = self.omni_usd.get_world_transform_matrix(plate_prim)

        # Calculate the bounds of the plate prim.
        bbox3d_gf.Transform(plate_tf_gf)
        # Range size is a 3D vector containing {GfRange3d::_max - GfRange3d::_min}.
        range_size = bbox3d_gf.GetRange().GetSize()

        # Get the quaternion of the plate prim in xyzw format from USD.
        plate_quat_gf = plate_tf_gf.ExtractRotation().GetQuaternion()
        plate_quat_xyzw = (plate_quat_gf.GetReal(), *plate_quat_gf.GetImaginary())

        # Create a plane slightly above the plate and slightly smaller such that
        # only the flat part of the plate is covered (heuristically estimated).
        plane_scale = (range_size[0] * 0.8, range_size[1] * 0.8, 1)
        # TODO(snair): range_size[2] would put it above the plate, we need it to be above the plate bottom
        plane_pos_gf = plate_tf_gf.ExtractTranslation() + self.pxr_gf.Vec3d(0, 0, range_size[2])
        plane_rot_euler_deg = self.isaac_core.utils.rotations.quat_to_euler_angles(
                                    np.array(plate_quat_xyzw), degrees=True)
        
        scatter_plane = self.rep.create.plane(
                                scale=plane_scale, position=plane_pos_gf,
                                rotation=plane_rot_euler_deg, visible=True)
        

    def generate_tableware(self):
        plate_cfg = self.models["plate"]
        
        # Create a plate prim.
        plate_prim = self.isaac_core.utils.prims.create_prim(
            prim_path=f"{self.scope_name}/plate",
            usd_path=plate_cfg.usd_path,
            semantic_label="plate",
            semantic_type="class",
            scale=np.array([1.3,1.3,1.3])
        )

        # # Create a rigid prim for the plate (ideally plate would be loaded
        # # as a static collider object, not a rigid prim but then physics
        # # doesn't work as expected. TODO(snair): Fix this.)
        # plate_rigid_prim = self.isaac_core.prims.RigidPrim(
        #     prim_path=str(plate_prim.GetPrimPath()),
        #     name="plate",
        #     position=self.pxr_gf.Vec3d(0,0,0),
        #     linear_velocity=np.array([0,0,0]),
        #     angular_velocity=np.array([0,0,0]),
        #     mass=1000,
        # )

        # # Enable rigid body physics for the plate. This turns it into a dynamic
        # # collider object that can be moved around or "hold" food meshes.
        # plate_rigid_prim.enable_rigid_body_physics()
        self.apply_physics_to_prim(plate_prim, approximation_shape="none")
        plate_material_properties = {
            "static_friction": 0.7,
            "dynamic_friction": 0.5,
            "restitution": 0.2
        }
        self.apply_material_to_prim(plate_prim, plate_material_properties)

        # self.world.scene.add(plate_rigid_prim)

        self.plate_prim = plate_prim
        # self.plate_rigid_prim = plate_rigid_prim

        plate_bbox3d_gf = self.bb_cache.ComputeWorldBound(self.plate_prim)
        plate_range = plate_bbox3d_gf.GetRange()
        plate_range_size = plate_range.GetSize()
        self.plate_radius = min(plate_range_size[0], plate_range_size[1]) / 2.0
        self.plate_bbox_height = plate_range_size[2]

    def _setup_world(self):
        """Setup the world for data generation.

        This function is called from the constructor and sets up the static attributes in the scene.
        """
        from pxr import UsdGeom
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import set_stage_up_axis
        import omni

        # Create bounding box cache for faster retrieval of prim bounds.
        self.bb_cache = self.isaac_core.utils.bounds.create_bbox_cache()

        # TODO(snair): Check if it's necessary to wrap this in a new layer.
        with self.rep.new_layer(name="StaticLayer"):
            # Load the room.
            self.create_scene()

            # Load the plate.
            self.generate_tableware()
        
        
        # Create container to store loaded food prims.
        self.food_rigid_prims = []
        self.food_prim_names = []

        # Load the food models.
        # TODO(snair): Don't hardcode size.
        self.times_reloaded = 0
        # self.reload_new_foods(size=5)

        # Setup cameras.
        render_products = self.make_views(
                look_at_prim=self.plate_prim,
                pos=(0,2,2),
                theta=360,
                ntheta=4,
                phi=90,
                nphi=3
            )

        # Setup annotators that will report groundtruth.
        # self.rgb = self.rep.AnnotatorRegistry.get_annotator("rgb")
        # self.bbox_2d_tight = self.rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        # self.instance_seg = self.rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        # self.semantic_seg = self.rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        # self.rgb.attach(render_products)
        # self.bbox_2d_tight.attach(render_products)
        # self.instance_seg.attach(render_products)
        # self.semantic_seg.attach(render_products)
        
        # # TODO(snair): Document when exactly update() should be called.
        # self.kit.update()

        # # Setup replicator graph.
        # self.setup_replicator()

        writer = self.rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=f"/pub3/smnair/foodverse/output/_fv_output_{self.cur_time}",
            semantic_types=["class"],
            rgb=True,
            bounding_box_2d_tight=True,
            bounding_box_2d_loose=True,
            semantic_segmentation=True,
            instance_segmentation=True,
            bounding_box_3d=True,
            occlusion=True,
            normals=True,
            distance_to_camera=True,
            distance_to_image_plane=True,
        )

        writer.attach(render_products)


    def setup_replicator(self):
        """Setup the replicator graph with various attributes.
        
        This function is called from _setup_world().
        """

        # Create lights.
        # TODO(snair): Create sphere lights
        
        with self.rep.new_layer(name="RandomizedLayer"):
            # TODO(snair): What should we set num_frames to? When we set it to 3, we get this 
            # weird result where frames (0,1) identical, frame 2 different, frame (3,4) identical, etc
            with self.rep.trigger.on_frame():
                # Randomize the lighting.

                pass
        
    def range_contains_2d_circle(self, range, point, range_scale=1.0):
        food_radius = np.sqrt(point[0]**2 + point[1]**2)
        range_size = range.GetSize()
        range_radius = min(range_size[0], range_size[1]) / 2.0

        return food_radius <= (range_radius * range_scale)
    
    def plate_contains_point_2d(self, plate_radius, point, radius_scale=1.0):
        """Checks if the point is within the plate, given some error (scale)
        bounds. Assumes that the plate is centered at the origin.
        """
        food_radius = np.sqrt(point[0]**2 + point[1]**2)

        return food_radius <= (plate_radius * radius_scale)
    
    def create_new_folder(self, folder_path, folder_name):
        """Creates a new folder if it doesn't already exist.
        """

        folder_path = os.path.join(folder_path, folder_name)
        
        # If folder exists, remove existing folder.
        self.remove_folder_if_exists(folder_path)
        
        # Create new folder.
        os.makedirs(folder_path)
        return folder_path
    
    def remove_folder_if_exists(self, folder_path):
        """Removes the folder at the given path if it exists.
        """
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    
    def create_temp_assets_dir(self, scene_idx):
        """Creates a temporary directory to store assets in.
        """
        folder_path = f"/tmp"
        folder_name = f"assets_{self.cur_time}_{scene_idx}"
        temp_dir_path = self.create_new_folder(folder_path, folder_name)

        return temp_dir_path
    
    def augment_food_model(self, food_usd_path, temp_assets_dir, item_idx):
        """Augments the food model by changing the texture file.
        Returns the path to the USD for the augmented food model.
        The augmented food model is stored in a temporary directory.

        Args:
            food_usd_path (str): Path to the USD for the food model.
            temp_assets_dir (str): Path to the temporary directory to store assets in.
        """
        # Ensure that the temporary directory exists.
        assert os.path.exists(temp_assets_dir), f"Temp assets dir {temp_assets_dir} does not exist."

        # Copy the food model to a temporary directory.
        food_model_dirname = os.path.dirname(food_usd_path)

        # Food model basename will be something like "id-26-chicken-leg-133g".
        food_model_basename = os.path.basename(food_model_dirname)
        tmp_food_dir_path = os.path.join(temp_assets_dir, f"{food_model_basename}_{item_idx}")

        # Copy the whole food model directory (containing the USD and texture
        # files) to the temporary directory.
        shutil.copytree(food_model_dirname, tmp_food_dir_path)

        # Glob for path to the texture file in the temp food model directory.
        texture_file_paths = glob.glob(os.path.join(tmp_food_dir_path, "textures/*.jpg"))
        texture_file_paths = list(filter(lambda x: "roughness" not in x, texture_file_paths))
        if len(texture_file_paths) != 1:
            raise ValueError(
                f"Expected to find 1 texture file in {tmp_food_dir_path}, but found {len(texture_file_paths)}")
        
        texture_file_path = texture_file_paths[0]

        # Augment brightness of the texture file.
        img = Image.open(texture_file_path).convert("RGB")
        img_enhancer = ImageEnhance.Brightness(img)
        factor = np.random.uniform(low=1.0, high=2.0)
        enhanced_output = img_enhancer.enhance(factor)

        # Overwrite the previous texture file with the augmented version.
        enhanced_output.save(texture_file_path)

        # Return the path to the USD for the augmented food model.
        tmp_usd_path = os.path.join(tmp_food_dir_path, os.path.basename(food_usd_path))
        return tmp_usd_path

    def load_new_foods(self, size, temp_dir_path):
        """Reloads new food models into the scene.
        """
        # Clear all existing food prims from the scene and our internal references.
        self.food_rigid_prims.clear()
        for i in range(len(self.food_prim_names)):
            self.world.scene.remove_object(name=self.food_prim_names[i])
        self.food_prim_names.clear()
        
        food_prims = []
        delta_theta = 2*np.pi/size
        theta = 0
        r = self.plate_radius*0.5

        for i in range(size):
            # Generate random food item.
            food_model = random.choice(self.food_models)
            # food_model = self.food_models[self.cur_food_model]
            # self.cur_food_model += 1

            tmp_food_usd_path = self.augment_food_model(food_model.usd_path, temp_dir_path, item_idx=i)
            food_name = f"{food_model.label}_{i}"
            food_prim = self.isaac_core.utils.prims.create_prim(
                prim_path=f"{self.scope_name}/{food_name}",
                usd_path=tmp_food_usd_path,
                semantic_label=f"{food_model.label}",
                semantic_type="class",
            )
            food_prims.append(food_prim)
            self.food_prim_names.append(food_name)
            
            # 0.01 is qualitatively derived heuristic to make the food mesh
            # sizes look realistic in comparison to the plate size.
            scale = food_model.scale * 0.01
            food_rigid_prim = self.isaac_core.prims.RigidPrim(
                prim_path=str(food_prim.GetPrimPath()),
                name=food_name,
                scale=self.pxr_gf.Vec3d(scale, scale, scale),
                orientation=self.isaac_core.utils.rotations.euler_angles_to_quat(
                    [random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi)]),
                # mass=1,
                linear_velocity=np.array([0,0,0]),
                angular_velocity=np.random.normal([0,0,0], [0,0,0])
            )
            food_rigid_prim.enable_rigid_body_physics()
            self.world.scene.add(food_rigid_prim)

            self.apply_physics_to_prim(food_prim, approximation_shape="convexDecomposition")

            food_material_properties = {
                "static_friction": 0.9,
                "dynamic_friction": 0.9,
                "restitution": 0.001
            }
            self.apply_material_to_prim(food_prim, food_material_properties)

            food_bbox3d_gf = self.bb_cache.ComputeWorldBound(food_prim)
            food_range = food_bbox3d_gf.GetRange()
            food_range_size = food_range.GetSize()
            food_height = max(food_range_size[0], food_range_size[1],
                              food_range_size[2])

            x_pos = r*np.cos(theta)
            # Check if x_pos + food_range_size[0] is within the plate radius.
            if x_pos + food_range_size[0] > self.plate_radius:
                diff = x_pos + food_range_size[0] - self.plate_radius
                x_pos -= diff

            y_pos = r*np.sin(theta)
            # Check if y_pos + food_range_size[1] is within the plate radius.
            if y_pos + food_range_size[1] > self.plate_radius:
                diff = y_pos + food_range_size[1] - self.plate_radius
                y_pos -= diff
            
            position=self.pxr_gf.Vec3d(x_pos,
                                       y_pos,
                                       self.plate_bbox_height + food_height)
            theta += delta_theta
            
            food_rigid_prim.set_default_state(position=position)
            self.food_rigid_prims.append(food_rigid_prim)

        # prims = self.get_prims(path_pattern=FOOD_PRIM_PATH,
        #                 semantics=[("class", "food")])
        # prims = [p.GetParent() for p in prims]
        # for prim in prims:
        #     self.apply_physics_to_prim(prim)
        #     # self.world.scene.add(prim)
        #     # world.scene().add(prim)

        # self._run_replicator(num_steps=200)

        # TODO(snair): size shouldn't be set to size, remember we're passing
        # in multiple items, we really need to instantiate size*num_items models
        # food_objects =  self.rep.randomizer.instantiate(foods, size=num_total, mode="point_instance")
        # food_objects = self.rep.create.group(foods)

        self.world.reset()
        # for i in range(3000):
        #     self.world.step(render=False)

        food_has_settled = False
        cur_idx = 0
        counter = 0
        
        # Measure start time.
        start_time = time.time()
        MAX_TIME_TO_WAIT_SECS = 30
        indices_to_remove = []
        while not food_has_settled:
            self.world.step(render=False)
            counter+=1
            vel = np.linalg.norm(
                    self.food_rigid_prims[cur_idx].get_linear_velocity())

            if vel < 0.001:
                # If the velocity of food item is below threshold, it has
                # stopped moving. Check the next food item.
                cur_idx += 1
                # Reset start time.
                start_time = time.time()
            elif time.time() - start_time > MAX_TIME_TO_WAIT_SECS:
                # If we've waited too long for this food item, move on 
                # to the next one to prevent the simulation from hanging.
                indices_to_remove.append(cur_idx)
                cur_idx += 1
                # Reset start time.
                start_time = time.time()
            
            if cur_idx == len(self.food_rigid_prims):
                food_has_settled = True

        self.bb_cache.Clear()
        for idx, food_prim in enumerate(food_prims):
            if idx in indices_to_remove:
                # If we already decided to remove this food item,
                # don't bother checking if it's outside the plate.
                continue

            food_bbox3d_gf = self.bb_cache.ComputeWorldBound(food_prim)
            food_range = food_bbox3d_gf.GetRange()
            food_range_min = food_range.GetMin()
            food_range_max = food_range.GetMax()
            food_centroid = food_bbox3d_gf.ComputeCentroid()

            # Remove food that is outside of the plate.
            if not (self.plate_contains_point_2d(
                    self.plate_radius, food_range_min, radius_scale=1.3) and
                self.plate_contains_point_2d(
                    self.plate_radius, food_range_max, radius_scale=1.3) and
                self.plate_contains_point_2d(
                    self.plate_radius, food_centroid, radius_scale=0.9)):
                indices_to_remove.append(idx)

        for idx in sorted(indices_to_remove, reverse=True):
            self.world.scene.remove_object(name=self.food_prim_names[idx])
            self.food_rigid_prims.pop(idx)
            self.food_prim_names.pop(idx)
        
        plate_is_empty = False
        if len(indices_to_remove) == size:
            plate_is_empty = True
        
        self.bb_cache.Clear()
        # Unnecessary, gc should handle this. But just to be safe.
        food_prims.clear()

        return plate_is_empty
    
    def reload_new_foods(self, size, temp_dir_path, max_retries=5):
        for i in range(max_retries):
            plate_is_empty = self.load_new_foods(size, temp_dir_path)
            if plate_is_empty:
                # If plate is empty, remove the temp folder and try again.
                self.remove_folder_if_exists(temp_dir_path)
                temp_dir_path = self.create_temp_assets_dir(
                                    scene_idx=self.num_scenes_generated)
            else:
                # If plate has at least one food item, we're done.
                break

                

    def __iter__(self):
        """Returns an iterator over the dataset.
        """
        return self
    
    def __next__(self):
        """Returns the next item in the dataset.
        """    
        temp_dir_path = self.create_temp_assets_dir(scene_idx=self.num_scenes_generated)
        self.reload_new_foods(size=7, temp_dir_path=temp_dir_path)

        # Step - Randomize and render
        self.rep.orchestrator.step()

        self.num_scenes_generated += 1
        self.remove_folder_if_exists(temp_dir_path)
        image, target = -1, -1
        return image, target

    def apply_physics_to_prim(self, prim, approximation_shape=None):
        # https://forums.developer.nvidia.com/t/unexpected-collision-behaviour-with-replicator-python-api/233575/9
        self.pxr_usd_physics.CollisionAPI.Apply(prim)
        self.pxr_physx_schema.PhysxCollisionAPI.Apply(prim)
        mesh_collision_api = self.pxr_usd_physics.MeshCollisionAPI.Apply(prim)
        
        if approximation_shape:
            mesh_collision_api.GetApproximationAttr().Set(approximation_shape)
        
        # The Collision Offset defines a small distance from the surface
        # of the collision geometry at which contacts start being generated.
        # The default value is -inf which means that the application tries to
        # determine a suitable value based on scene gravity, simulation frame
        # rate and object size. Increase this parameter if fast-moving objects
        # are missing collisions. Increasing the offset too much incurs a
        # performance penalty since more contacts are generated between
        # objects that need to be processed at each simulation step.    
        prim.GetAttribute("physxCollision:contactOffset").Set(0.1)

        # The Rest Offset defines a small distance from the surface
        # of the collision geometry at which the effective contact with
        # the shape takes place. It can be both positive and negative, and
        # may be useful in cases where the visualization mesh is slightly
        # smaller than the collision geometry: setting an appropriate
        # negative rest offset results in the contact occurring at the
        # visually correct distance.
        prim.GetAttribute("physxCollision:restOffset").Set(0.1)

    def apply_material_to_prim(self, prim, material_properties):
        prim_path = prim.GetPath()
        material_path = prim_path.AppendChild("PhysicsMaterial")
        self.pxr_usd_shade.Material.Define(self.stage, material_path)

        material = self.pxr_usd_physics.MaterialAPI.Apply(self.stage.GetPrimAtPath(material_path))
        material.CreateStaticFrictionAttr().Set(material_properties["static_friction"])
        material.CreateDynamicFrictionAttr().Set(material_properties["dynamic_friction"])
        material.CreateRestitutionAttr().Set(material_properties["restitution"])

    def apply_physics_to_prims(self, prims):
        for prim in prims:
            self.apply_physics_to_prim(prim)

    def apply_physics_to_prim_paths(self, prim_paths):
        prims = self.rep.utils.find_prims(prim_paths, mode='prims')
        for prim in prims:
            self.apply_physics_to_prim(prim)

    def get_prims(
        self,
        path_pattern=None,
        path_pattern_exclusion=None,
        prim_types=None,
        prim_types_exclusion=None,
        semantics=None,
        semantics_exclusion=None,
        cache_prims=True # ignored
    ):
        # Code modified from:
        # isaac_sim/exts/omni.replicator.core-1.4.3+lx64.r.cp37/omni/replicator/core/ogn/python/_impl/nodes/OgnGetPrims.py
        # rep.get.prims returns ReplicatorItems, which aren't very useful
        # this function does the same as rep.get.prims but
        # returns the prim paths instead of Replicator Items.

        stage = self.kit.context.get_stage()
        gathered_prims = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            prim_type = str(prim.GetTypeName()).lower()
            prim_semantics = self.rep.utils._parse_semantics(prim)

            if path_pattern:
                if not re.search(path_pattern, prim_path):
                    continue
            if path_pattern_exclusion:
                if re.search(path_pattern_exclusion, prim_path):
                    continue
            if prim_types:
                if prim_type not in prim_types:
                    continue
            if prim_types_exclusion:
                if prim_type in prim_types_exclusion:
                    continue
            if semantics:
                if not any([prim_semantic in semantics 
                                for prim_semantic in prim_semantics]):
                    continue
            if semantics_exclusion:
                if any([prim_semantic in semantics_exclusion 
                            for prim_semantic in prim_semantics]):
                    continue

            gathered_prims.append(prim)
        return gathered_prims
    
    def load_mesh(
            self,
            semantics,
            usd_path,
            num=1,
            scale_min=1,
            scale_max=None,
            position_mu=[0,0,0],
            position_std=None,
            rotation_min=[0,0,0],
            rotation_max=None,
            velocity_mu=[0,0,0],
            velocity_std=None,
            angular_velocity_mu=[0,0,0],
            angular_velocity_std=None,
            static_friction=None,
            dynamic_friction=None,
            restitution=None,
            mass=None):
        
        # Load mesh into simulation.
        mesh = self.rep.create.from_usd(usd_path, count=num,
                                        semantics=semantics)
        
        # Compute initial pose and velocity for mesh.
        position = (self.rep.distribution.normal(position_mu, position_std)
                        if position_std else position_mu)
        rotation = (self.rep.distribution.uniform(rotation_min, rotation_max)
                            if rotation_max else rotation_min)
        scale = (self.rep.distribution.uniform(scale_min, scale_max)
                        if scale_max else scale_min)
        
        velocity = (self.rep.distribution.normal(velocity_mu, velocity_std)
                            if velocity_std else velocity_mu)
        angular_velocity = (self.rep.distribution.normal(
            angular_velocity_mu, angular_velocity_std)
                if angular_velocity_std else angular_velocity_mu)
        
        with mesh:
            self.rep.physics.collider(approximation_shape='convexHull')
            self.rep.modify.pose(
                position=position, 
                rotation=rotation,
                scale=scale
            )
            self.rep.physics.rigid_body(
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            self.rep.physics.physics_material(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution
            )
            self.rep.physics.mass(
                mass=mass,
            )

        return mesh
    
    def make_views(self, look_at_prim, pos, theta, ntheta, phi, nphi):
        """
        Generates camera poses on the surface of a sphere.
        Starting at pos, sweeps theta degrees at ntheta uniformly-spaced points.
        For each point, also sweeps phi degress at nphi uniformly-spaced points.

        pos: initial position of the camera
        theta: max rotation about z (yaw)
        ntheta: number of angles to split theta into
        phi: max rotation about x (pitch)
        nphi: number of angles to split phi into

        returns: list of render products to call writer.attach(...) on
        """
        assert theta <= 360, 'Max theta should not exceed 360 deg'
        assert phi <= 180, 'Max phi should not exceed 180 deg'

        # Cartesian to spherical
        x0,y0,z0 = pos
        r = np.sqrt(x0**2 + y0**2 + z0**2)
        th0 = np.arctan2(y0, x0)
        ph0 = np.arctan2(np.sqrt(x0**2 + y0**2), z0)

        # Generate
        if theta == 360:
            theta -= 360 / ntheta
        if phi == 180:
            phi -= 180 / nphi

        th, ph = np.meshgrid(np.linspace(0, theta, ntheta, endpoint=True),
                            np.linspace(0, phi, nphi, endpoint=True))
        print(th)
        print(ph)
        th = th0 + np.deg2rad(th.flatten())
        ph = ph0 - np.deg2rad(ph.flatten()) # Subtract so +ve = up
        
        # Spherical to cartesian
        x = r * np.sin(ph) * np.cos(th)
        y = r * np.sin(ph) * np.sin(th)
        z = r * np.cos(ph)
        pts = np.stack([x,y,z], axis=-1)

        render_products = []
        for pt in pts:
            carb.log_warn(f"(SNAIR) Creating camera at {pt.tolist()}")
            camera = self.rep.create.camera(
                position=pt.tolist(),
                look_at=str(look_at_prim.GetPrimPath()),
                # look_at=look_at_prim,
                focal_length=140)
            
            render_product = self.rep.create.render_product(
                camera, (1200, 1200))
            render_products.append(render_product)
        
        return render_products



if __name__ == "__main__":

    carb.log_warn(f"(SNAIR) Starting dataset creation...")
    dataset = FoodverseObjects()

    image_num = 0
    carb.log_warn(f"(SNAIR) Starting dataset iteration...")
    for image, target in dataset:
        # carb.log_warn(f"(SNAIR) Generated image number: {image_num}")
        image_num += 1

        if image_num > 6000 or dataset.exiting:
            break

    dataset.cleanup()

