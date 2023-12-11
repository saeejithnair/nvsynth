import json
import glob
import os
import re
import sys
import yaml
import numpy as np
from collections.abc import Iterable
# os.system('rm -r _output/*')

# Start app
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({'headless': True, 'renderer': "RayTracedLighting"})

import omni
import omni.replicator.core as rep
from pxr import UsdPhysics, PhysxSchema, Gf

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, lookat_to_quatf
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache

rep.settings.carb_settings("/omni/replicator/RTSubframes", 2)

# ** MODIFY THIS TO YOUR ABSOLUTE PATH **
env_path = "/home/smnair/work/nutrition/vip-omni/assets/scene/simple_room/simple_room.usd"
plate_path = "/home/smnair/work/nutrition/vip-omni/assets/tableware/plate/plate.usd"

 
plate_prim_path = "/Replicator/Ref_Xform/Ref/model/mesh"
table_prim_path = "/Replicator/Ref_Xform_01/Ref/table_low_327/table_low"
food_prim_path = r"\/Replicator\/Ref_Xform.*\/Ref"


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


def load_food(usd_path,
              name=None,
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

    semantics = [("class", "food")]
    if name is not None:
        semantics.append(("food_type", name))

    food = rep.create.from_usd(usd_path, count=num, semantics=semantics)
    with food:
        rep.physics.collider(approximation_shape='convexHull')
        rep.modify.pose(
            position=rep.distribution.normal(position_mu, position_std) if position_std else position_mu, 
            rotation=rep.distribution.uniform(rotation_min, rotation_max) if rotation_max else rotation_min,
            scale=rep.distribution.uniform(scale_min, scale_max) if scale_max else scale_min
        )
        rep.physics.rigid_body(
            velocity=rep.distribution.normal(velocity_mu, velocity_std) if velocity_std else velocity_mu,
            angular_velocity=rep.distribution.normal(angular_velocity_mu, angular_velocity_std) if angular_velocity_std else angular_velocity_mu
        )
        rep.physics.physics_material(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution
        )
        rep.physics.mass(
            mass=mass,
        )
    return food


def load_plate():
    plate = rep.create.from_usd(plate_path, semantics=[("class", "plate")])
    with plate:
        rep.modify.pose(position=(0,0,0), scale=100)
        rep.physics.collider(approximation_shape="none")
        rep.physics.rigid_body(
                velocity=0,
                angular_velocity=0
        )
        rep.physics.physics_material(
            static_friction=0.7,
            dynamic_friction=0.5,
            restitution=0.2
        )
        rep.physics.mass(
            mass=1000,
        )
    return plate


def load_env():
    env = rep.create.from_usd(env_path, semantics=[("class", "scene")])
    with env:
        rep.modify.pose(position=(0,0,0), scale=100)
    with rep.get.prims("table_low$"):
        rep.physics.collider(approximation_shape="none")
    return env


def apply_physics_to_prim(prim):
    # https://forums.developer.nvidia.com/t/unexpected-collision-behaviour-with-replicator-python-api/233575/9
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.GetAttribute("physxCollision:contactOffset").Set(0.002)
    prim.GetAttribute("physxCollision:restOffset").Set(0.001)


def apply_physics_to_prim_paths(prim_paths):
    prims = rep.utils.find_prims(prim_paths, mode='prims')
    for prim in prims:
        apply_physics_to_prim(prim)


def get_prims(
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
    # this function does the same as rep.get.prims but returns the prim paths instead of Replicator Items

    stage = omni.usd.get_context().get_stage()
    gathered_prims = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_type = str(prim.GetTypeName()).lower()
        prim_semantics = rep.utils._parse_semantics(prim)

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
            if not any([prim_semantic in semantics for prim_semantic in prim_semantics]):
                continue
        if semantics_exclusion:
            if any([prim_semantic in semantics_exclusion for prim_semantic in prim_semantics]):
                continue

        gathered_prims.append(prim)
    return gathered_prims


def make_views(pos, theta, ntheta, phi, nphi):
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
        camera = rep.create.camera(position=pt.tolist(), look_at=(0,0,0), focal_length=96)
        render_product = rep.create.render_product(camera, (1024, 1024))
        render_products.append(render_product)
    
    return render_products


def add_layer_kwarg(kwargs, layer, name, uniform=False, normal=False):
    if name not in layer:
        return

    assert not (uniform and normal), f'ERROR: Only specify uniform OR normal distirbution for property "{name}"'

    if uniform:
        post1 = '_min'
        post2 = '_max'
    elif normal:
        post1 = '_mu'
        post2 = '_std'
    else:
        post1 = ''

    val = layer[name]
    if isinstance(val, Iterable) and len(val) == 2:
        kwargs[name+post1] = val[0]
        kwargs[name+post2] = val[1]
    else:
        kwargs[name+post1] = val


def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        scene = yaml.safe_load(f)['scene']

    try:
        camera = scene['camera']
    except KeyError:
        print('ERROR: Missing "scene.camera" field in yaml')
        raise

    default_range = {'max': 0, 'steps': 1}
    h_range = camera.get('horizontal_range', default_range)
    v_range = camera.get('vertical_range', default_range)

    orchestrator_kwargs = {
        'num_steps': scene.get('num_steps', 100)
    }

    camera_kwargs = {
        'pos': camera.get('position', [0,200,100]),
        'theta': h_range.get('max', 0),
        'ntheta': h_range.get('steps', 1),
        'phi': v_range.get('max', 0),
        'nphi': v_range.get('steps', 1),
    }

    layer_kwargs = []
    for name, layer in scene.items():
        if 'layer' not in name:
            continue

        assert 'usd_path' in layer, f'ERROR: Missing USD path for "{name}"'

        food_kwargs = {'usd_path': layer['usd_path']}
        add_layer_kwarg(food_kwargs, layer, 'name')
        add_layer_kwarg(food_kwargs, layer, 'scale', uniform=True)
        add_layer_kwarg(food_kwargs, layer, 'num')
        add_layer_kwarg(food_kwargs, layer, 'position', normal=True)
        add_layer_kwarg(food_kwargs, layer, 'rotation', uniform=True)
        add_layer_kwarg(food_kwargs, layer, 'velocity', normal=True)
        add_layer_kwarg(food_kwargs, layer, 'angular_velocity', normal=True)
        add_layer_kwarg(food_kwargs, layer, 'static_friction')
        add_layer_kwarg(food_kwargs, layer, 'dynamic_friction')
        add_layer_kwarg(food_kwargs, layer, 'restitution')
        add_layer_kwarg(food_kwargs, layer, 'mass')

        layer_kwargs.append(food_kwargs)

    assert len(layer_kwargs) > 0, ('ERROR: Missing layers in scene yaml.'
                                   'You can add layers by adding fields containing "layer" to the scene'
                                   'Ex: "scene.layer1", "scene.burger_layer"')

    return orchestrator_kwargs, camera_kwargs, layer_kwargs


def main(yaml_path):
    orchestrator_kwargs, camera_kwargs, layer_kwargs = parse_yaml(yaml_path)

    with rep.new_layer():
        # plate = load_plate()
        env = load_env()

        # Load each food layer
        print('*'*50)
        for i, food_kwargs in enumerate(layer_kwargs):
            print(f'LAYER {i}:')
            print(json.dumps(food_kwargs, indent=2))
            print()
            load_food(**food_kwargs)
        print('*'*50)

        # Get food, plate and table prims and apply physics API to them
        prims = get_prims(path_pattern=food_prim_path,
                          semantics=[("class", "food"), ("class", "plate")])
        prims = [p.GetParent() for p in prims]
        prims.extend(get_prims("table_low$"))

        print('*'*50)
        for prim in prims:
            print("Applying physics to: " + str(prim.GetPath()))
            apply_physics_to_prim(prim)
        print('*'*50)

        render_products = make_views(**camera_kwargs)

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir="/home/smnair/work/nutrition/vip-omni/_output/",
            semantic_types=["class", "food_type"],
            rgb=True,
            bounding_box_2d_tight=True,
            bounding_box_2d_loose=True,
            semantic_segmentation=True,
            instance_segmentation=True,
            instance_id_segmentation=True,
            bounding_box_3d=True,
            occlusion=True,
            normals=True,
        )
        # writer.initialize(
        #     output_dir="/home/smnair/work/nutrition/vip-omni/_output/",
        #     bbox_height_threshold=5,
        #     fully_visible_threshold=0.75,
        #     omit_semantic_type=True
        # )
        writer.attach(render_products)

        run_orchestrator(**orchestrator_kwargs)
    simulation_app.update()
    simulation_app.close()
        

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'ERROR: Missing yaml path to load, expected make_food_scene.py [YAML_PATH]'
    yaml_path = sys.argv[1]
    # yaml_path = "configs/wings.yaml"
    main(yaml_path)
