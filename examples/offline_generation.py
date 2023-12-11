# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate offline synthetic dataset
"""
import math
import os
import random

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import (
    euler_angles_to_quat,
    lookat_to_quatf,
    quat_to_euler_angles,
)
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from omni.isaac.kit import SimulationApp
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

# Set rendering parameters and create an instance of kit
CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": True,
    "width": 1024,
    "height": 1024,
    "num_frames": 10,
}
simulation_app = SimulationApp(launch_config=CONFIG)

ENV_URL = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
FORKLIFT_URL = "/Isaac/Props/Forklift/forklift.usd"
PALLET_URL = "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd"
CARDBOX_URL = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd"
CONE_URL = "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd"
SCOPE_NAME = "/MyScope"


# Increase subframes if shadows/ghosting appears of moving objects
# See known issues: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator.html#known-issues
rep.settings.carb_settings("/omni/replicator/RTSubframes", 2)


# Helper function to find the assets server
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception(
            "Nucleus server not found, could not access Isaac Sim assets folder"
        )
    return assets_root_path + relative_path


# Randomize boxes materials and their location on the surface of the given prim
def register_scatter_boxes(prim):
    # Calculate the bounds of the prim to create a scatter plane of its size
    bb_cache = create_bbox_cache()
    bbox3d_gf = bb_cache.ComputeLocalBound(prim)
    prim_tf_gf = omni.usd.get_world_transform_matrix(prim)

    # Calculate the bounds of the prim
    bbox3d_gf.Transform(prim_tf_gf)
    range_size = bbox3d_gf.GetRange().GetSize()

    # Get the quaterion of the prim in xyzw format from usd
    prim_quat_gf = prim_tf_gf.ExtractRotation().GetQuaternion()
    prim_quat_xyzw = (prim_quat_gf.GetReal(), *prim_quat_gf.GetImaginary())

    # Create a plane on the pallet to scatter the boxes on
    plane_scale = (range_size[0] * 0.8, range_size[1] * 0.8, 1)
    plane_pos_gf = prim_tf_gf.ExtractTranslation() + Gf.Vec3d(0, 0, range_size[2])
    plane_rot_euler_deg = quat_to_euler_angles(np.array(prim_quat_xyzw), degrees=True)
    scatter_plane = rep.create.plane(
        scale=plane_scale,
        position=plane_pos_gf,
        rotation=plane_rot_euler_deg,
        visible=False,
    )

    cardbox_mats = [
        prefix_with_isaac_asset_server(
            "/Isaac/Environments/Simple_Warehouse/Materials/MI_PaperNotes_01.mdl"
        ),
        prefix_with_isaac_asset_server(
            "/Isaac/Environments/Simple_Warehouse/Materials/MI_CardBoxB_05.mdl"
        ),
    ]

    def scatter_boxes():
        cardboxes = rep.create.from_usd(
            prefix_with_isaac_asset_server(CARDBOX_URL),
            semantics=[("class", "Cardbox")],
            count=5,
        )
        with cardboxes:
            rep.randomizer.scatter_2d(scatter_plane, check_for_collisions=True)
            rep.randomizer.materials(cardbox_mats)
        return cardboxes.node

    rep.randomizer.register(scatter_boxes)


# Randomly place cones from calculated locations around the working area (combined bounds) of the forklift and pallet
def register_cone_placement(forklift_prim, pallet_prim):
    # Helper function to get the combined bounds of the forklift and pallet
    bb_cache = create_bbox_cache()
    combined_range_arr = compute_combined_aabb(
        bb_cache, [forklift_prim.GetPrimPath(), pallet_prim.GetPrimPath()]
    )

    min_x = float(combined_range_arr[0])
    min_y = float(combined_range_arr[1])
    min_z = float(combined_range_arr[2])
    max_x = float(combined_range_arr[3])
    max_y = float(combined_range_arr[4])
    corners = [
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (min_x, max_y, min_z),
        (max_x, max_y, min_z),
    ]

    def place_cones():
        cones = rep.create.from_usd(
            prefix_with_isaac_asset_server(CONE_URL),
            semantics=[("class", "TrafficCone")],
        )
        with cones:
            rep.modify.pose(position=rep.distribution.sequence(corners))
        return cones.node

    rep.randomizer.register(place_cones)


# Randomize lights around the scene
def register_lights_placement(forklift_prim, pallet_prim):
    bb_cache = create_bbox_cache()
    combined_range_arr = compute_combined_aabb(
        bb_cache, [forklift_prim.GetPrimPath(), pallet_prim.GetPrimPath()]
    )
    pos_min = (combined_range_arr[0], combined_range_arr[1], 6)
    pos_max = (combined_range_arr[3], combined_range_arr[4], 7)

    def randomize_lights():
        lights = rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0.2, 0.1, 0.1), (0.9, 0.8, 0.8)),
            intensity=rep.distribution.uniform(500, 2000),
            position=rep.distribution.uniform(pos_min, pos_max),
            scale=rep.distribution.uniform(5, 10),
            count=3,
        )
        return lights.node

    rep.randomizer.register(randomize_lights)


def simulate_falling_objects(prim, num_sim_steps=250, num_boxes=8):
    # Create a simulation ready world
    world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)

    # Choose a random spawn offset relative to the given prim
    prim_tf = omni.usd.get_world_transform_matrix(prim)
    spawn_offset_tf = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(random.uniform(-0.5, 0.5), random.uniform(3, 3.5), 0)
    )
    spawn_pos_gf = (spawn_offset_tf * prim_tf).ExtractTranslation()

    # Spawn pallet prim
    pallet_prim_name = "SimulatedPallet"
    pallet_prim = prims.create_prim(
        prim_path=f"{SCOPE_NAME}/{pallet_prim_name}",
        usd_path=prefix_with_isaac_asset_server(PALLET_URL),
        semantic_label="Pallet",
    )

    # Get the height of the pallet
    bb_cache = create_bbox_cache()
    curr_spawn_height = (
        bb_cache.ComputeLocalBound(pallet_prim).GetRange().GetSize()[2] * 1.1
    )

    # Wrap the pallet prim into a rigid prim to be able to simulate it
    pallet_rigid_prim = RigidPrim(
        prim_path=str(pallet_prim.GetPrimPath()),
        name=pallet_prim_name,
        position=spawn_pos_gf + Gf.Vec3d(0, 0, curr_spawn_height),
    )

    # Make sure physics are enabled on the rigid prim
    pallet_rigid_prim.enable_rigid_body_physics()

    # Register rigid prim with the scene
    world.scene.add(pallet_rigid_prim)

    # Spawn boxes falling on the pallet
    random_assets = [
        prefix_with_isaac_asset_server(CARDBOX_URL),
        prefix_with_isaac_asset_server(CONE_URL),
    ]
    for i in range(num_boxes):
        # Spawn box prim
        cardbox_prim_name = f"SimulatedCardbox_{i}"
        box_prim = prims.create_prim(
            prim_path=f"{SCOPE_NAME}/{cardbox_prim_name}",
            # usd_path=prefix_with_isaac_asset_server(CARDBOX_URL),
            usd_path=random.choice(random_assets),
            semantic_label="Cardbox",
        )

        # Add the height of the box to the current spawn height
        curr_spawn_height += (
            bb_cache.ComputeLocalBound(box_prim).GetRange().GetSize()[2] * 1.1
        )

        # Wrap the cardbox prim into a rigid prim to be able to simulate it
        box_rigid_prim = RigidPrim(
            prim_path=str(box_prim.GetPrimPath()),
            name=cardbox_prim_name,
            position=spawn_pos_gf
            + Gf.Vec3d(
                random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), curr_spawn_height
            ),
            orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
        )

        # Make sure physics are enabled on the rigid prim
        box_rigid_prim.enable_rigid_body_physics()

        # Register rigid prim with the scene
        world.scene.add(box_rigid_prim)

    # Reset world after adding simulated assets for physics handles to be propagated properly
    world.reset()

    # Simulate the world for the given number of steps or until the highest box stops moving
    last_box = world.scene.get_object(f"SimulatedCardbox_{num_boxes - 1}")
    for i in range(num_sim_steps):
        world.step(render=False)
        if last_box and np.linalg.norm(last_box.get_linear_velocity()) < 0.001:
            print(f"Simulation stopped after {i} steps")
            break


# Starts replicator and waits until all data was successfully written
def run_orchestrator():
    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def main():
    # Open the environment in a new stage
    print(f"Loading Stage {ENV_URL}")
    open_stage(prefix_with_isaac_asset_server(ENV_URL))

    # Create a custom scope for newly added prims
    stage = get_current_stage()
    scope = UsdGeom.Scope.Define(stage, SCOPE_NAME)

    # Spawn a new forklift at a random pose
    forklift_prim = prims.create_prim(
        prim_path=f"{SCOPE_NAME}/Forklift",
        position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
        orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
        usd_path=prefix_with_isaac_asset_server(FORKLIFT_URL),
        semantic_label="Forklift",
    )

    # Spawn the pallet in front of the forklift with a random offset on the Y axis
    forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
    pallet_offset_tf = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(0, random.uniform(-1.2, -2.4), 0)
    )
    pallet_pos_gf = (pallet_offset_tf * forklift_tf).ExtractTranslation()
    forklift_quat_gf = forklift_tf.ExtractRotation().GetQuaternion()
    forklift_quat_xyzw = (forklift_quat_gf.GetReal(), *forklift_quat_gf.GetImaginary())

    pallet_prim = prims.create_prim(
        prim_path=f"{SCOPE_NAME}/Pallet",
        position=pallet_pos_gf,
        orientation=forklift_quat_xyzw,
        usd_path=prefix_with_isaac_asset_server(PALLET_URL),
        semantic_label="Pallet",
    )

    # Simulate dropping objects on a pile behind the forklift
    # simulate_falling_objects(forklift_prim)

    # Register randomizers
    register_scatter_boxes(pallet_prim)
    register_cone_placement(forklift_prim, pallet_prim)
    register_lights_placement(forklift_prim, pallet_prim)

    # Spawn a camera in the driver's location looking at the pallet
    foklift_pos_gf = forklift_tf.ExtractTranslation()
    driver_cam_pos_gf = foklift_pos_gf + Gf.Vec3d(0.0, 0.0, 1.9)
    rotate_180_on_y_quat_gf = Gf.Quatf(0, 0, 1, 0)
    look_at_pallet_quat_gf = (
        lookat_to_quatf(driver_cam_pos_gf, pallet_pos_gf, (0, 0, 1))
        * rotate_180_on_y_quat_gf
    )
    look_at_pallet_xyzw = (
        look_at_pallet_quat_gf.GetReal(),
        *look_at_pallet_quat_gf.GetImaginary(),
    )

    driver_cam_prim = prims.create_prim(
        prim_path=f"{SCOPE_NAME}/DriverCamera",
        prim_type="Camera",
        position=driver_cam_pos_gf,
        orientation=look_at_pallet_xyzw,
        attributes={
            "focusDistance": 400,
            "focalLength": 24,
            "clippingRange": (0.1, 10000000),
        },
    )

    # Camera looking at the pallet
    pallet_cam = rep.create.camera()

    # Camera looking at the forklift from a top view with large min clipping to see the scene through the ceiling
    top_view_cam = rep.create.camera(clipping_range=(6.0, 1000000.0))

    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):
        # rep.randomizer.scatter_boxes()
        # rep.randomizer.place_cones()
        simulate_falling_objects(forklift_prim)
        rep.randomizer.randomize_lights()
        pos_min = (pallet_pos_gf[0] - 2, pallet_pos_gf[1] - 2, 2)
        pos_max = (pallet_pos_gf[0] + 2, pallet_pos_gf[1] + 2, 4)
        with pallet_cam:
            rep.modify.pose(
                position=rep.distribution.uniform(pos_min, pos_max),
                look_at=str(pallet_prim.GetPrimPath()),
            )
        with top_view_cam:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (foklift_pos_gf[0], foklift_pos_gf[1], 9),
                    (foklift_pos_gf[0], foklift_pos_gf[1], 11),
                ),
                rotation=rep.distribution.uniform((0, -90, 0), (0, -90, 180)),
            )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    output_directory = os.getcwd() + "/_output_headless_ycb_v2"
    print("Outputting data to ", output_directory)
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        distance_to_image_plane=True,
        bounding_box_3d=True,
        occlusion=True,
        normals=True,
    )

    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    driver_rp = rep.create.render_product(
        str(driver_cam_prim.GetPrimPath()), RESOLUTION
    )
    pallet_rp = rep.create.render_product(pallet_cam, RESOLUTION)
    forklift_rp = rep.create.render_product(top_view_cam, RESOLUTION)
    writer.attach([driver_rp, forklift_rp, pallet_rp])

    run_orchestrator()
    simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
