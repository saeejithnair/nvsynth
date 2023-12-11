#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT
# Date:	26.09.2022

import os
import glob
import pathlib
import random
import numpy as np
import multiprocessing as mp
import copy

import torch
import signal
import trimesh
import h5py
import json
import PIL
import sys

import omni
from pxr import UsdGeom, UsdShade, Gf, Sdf, Semantics, PhysicsSchema, PhysxSchema, PhysicsSchemaTools
from omni.isaac.synthetic_utils import OmniKitHelper, SyntheticDataHelper, DomainRandomization, DataWriter, shapenet, utils
from omni.physx.scripts import utils as physx_utils
from omni.isaac.dynamic_control import _dynamic_control as dc


import dataset_utilities
import CollisionWithScene
import KeypointsInScene

# Print maximum length in console.
np.set_printoptions(threshold=sys.maxsize)

# Set backend for headless rendering in pyrender
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Setup default generation variables
# Value are (min, max) ranges
RANDOM_TRANSLATION_X = (-6.0, 6.0)
RANDOM_TRANSLATION_Y = (-3.0, 3.0)
RANDOM_TRANSLATION_Z = (30.0, 80.0)
RANDOM_ROTATION_X = (0.0, 360.0)
RANDOM_ROTATION_Y = (0.0, 360.0)
RANDOM_ROTATION_Z = (0.0, 360.0)
#CAMERA_DISTANCE = 200.0

# Gripper Parameters ParallelJaw and Suction Cup
GRIPPER_HEIGHT = 11.21 # cm
AREA_CENTER = 0.8 # cm
PREGRASP_PARALLEL_DISTANCE = 3 # cm note: this is added to the predicted gripper width.
GRIPPER_WIDTH_MAX = 8 #cm
SUCTIONCUP_HEIGHT = 11.21 #cm
SUCTIONCUP_APPROACH_DISTANCE = 2 # cm

# Dataset Parameters
#CAM_ANGLE_NICK_START = 0.523599 # 30 deg
#CAM_ANGLE_NICK_END =  0.174533 # 10 deg
CAM_ANGLE_NICK_START = random.uniform(0.3, 0.8) # 0.523599 # 30 deg
CAM_ANGLE_NICK_END =  random.uniform(0.1, 0.3) 

CAM_ANGLE_PHI_START = 0
#CAM_ANGLE_PHI_END = 2*np.pi
CAM_ANGLE_PHI_END = random.uniform(np.pi, 2*np.pi)

#CAM_RADIUS_START = 80.0 #cm
CAM_RADIUS_START = random.uniform(70.0, 80.0) #cm
#CAM_RADIUS_END = 120.0 #cm
CAM_RADIUS_END = random.uniform(100.0, 140.0) #cm

FOCAL_LENGTH =  35#mm

#[DEFAULT] 
STEPS_PHI = 4 # different viewpoints on circle
STEPS_NICK = 2 # different viewpoints on arc
STEPS_RADIUS = 2 # different viewpoints in depth
STEPS_TD = 0

STEP_ANGLE_PHI = (CAM_ANGLE_PHI_END - CAM_ANGLE_PHI_START) / STEPS_PHI
STEP_RADIUS = (CAM_RADIUS_END - CAM_RADIUS_START) / STEPS_RADIUS
STEP_ANGLE_NICK = (CAM_ANGLE_NICK_END - CAM_ANGLE_NICK_START) / STEPS_NICK
# Physics parameters
# https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac/synthetic_utils/docs/index.html#
SIMULATION_CONFIG = {
    "simulation_time" : 7,
    "dt_rendering" : 1/ 60.0,
    'dt_physics' : 1/ 60.0,
    'physics_substeps' : 6,
}

# Default rendering parameters
RENDER_CONFIG = {
    "width": 1200,
    "height": 1200,
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 32,
    "max_bounces": 20,
    "max_specular_transmission_bounces": 20,
    "max_volume_bounces": 10,
    "subdiv_refinement_level": 2,
    "headless": False,
    "experience": f'{os.environ["EXP_PATH"]}/isaac-sim-python.json',
}

def rad2deg(val):
    ret = (val*180)/np.pi
    return ret

def convert_to_franka_6DOF(vec_a, vec_b, contact_pt, width):
    """Convert Contact-Point Pregrasp representation to 6DOF Gripper Pose (4x4)."""
    # get 3rd unit vector
    c_ = np.cross(vec_a, vec_b)
    # rotation matrix
    R_ = [
        [vec_b[0], c_[0], vec_a[0]],
        [vec_b[1], c_[1], vec_a[1]],
        [vec_b[2], c_[2], vec_a[2]]
    ]
    # translation t
    t_ = contact_pt + width/2 * vec_b + (GRIPPER_HEIGHT-AREA_CENTER) * vec_a * (-1)
    # create 4x4 transform matrix of grasp
    pregrasp_transform_ = [
        [R_[0][0], R_[0][1], R_[0][2], t_[0]],
        [R_[1][0], R_[1][1], R_[1][2], t_[1]],
        [R_[2][0], R_[2][1], R_[2][2], t_[2]],
        [0.0, 0.0, 0.0, 1.0]
    ]
    return np.array(pregrasp_transform_)

def generate_camera_config():
    """
    Creates camera config dictionary with given variables.
    STEPS, CAMERA_DISTANCE, FOCAL_LENGTH, CAM_ANGLE_NICK, STEP_ANGLE
    """
    viewpt_counter = 0
    ret_dict = {}
    ret_dict['meta'] = {
        "num_viewpts" : (STEPS_PHI*(STEPS_NICK+1)*(STEPS_RADIUS+1)) + 1 + STEPS_TD
    }
    ret_dict['viewpts'] = {
        str(viewpt_counter) : {
            "radius" : CAM_RADIUS_START,
            "phi" : 0.0,
            "theta" : 0.0,
            "focal_length" : FOCAL_LENGTH,
            "top_down" : False,
        }
    }
    viewpt_counter += 1
    for rad_idx in range(STEPS_RADIUS+1):
        for nick_idx in range(STEPS_NICK+1):
            for phi_idx in range(STEPS_PHI): # ! not plus one here, we end where we start. otherthise we would have same image at phi=0 and phi=2pi
                radius = CAM_RADIUS_START + STEP_RADIUS*rad_idx
                theta = CAM_ANGLE_NICK_START + STEP_ANGLE_NICK*nick_idx
                phi = CAM_ANGLE_PHI_START + STEP_ANGLE_PHI*phi_idx
                ret_dict['viewpts'][str(viewpt_counter)] = {
                    "radius" : radius,
                    "phi" : phi,
                    "theta" : theta,
                    "focal_length" : FOCAL_LENGTH,
                    "top_down" : False,
                }
                viewpt_counter += 1
    for _ in range(STEPS_TD):
        theta = random.uniform(0, 2*CAM_ANGLE_NICK_END)
        radius_min = CAM_RADIUS_START/np.cos(theta)
        radius_max = CAM_RADIUS_END/np.cos(theta)
        radius = random.uniform(radius_min, radius_max)
        phi = random.uniform(CAM_ANGLE_PHI_START, CAM_ANGLE_PHI_END)
        ret_dict['viewpts'][str(viewpt_counter)] = {
            "radius" : radius,
            "phi" : phi,
            "theta" : theta,
            "focal_length" : FOCAL_LENGTH,
            "top_down" : True, 
        }
        viewpt_counter += 1
    return ret_dict

CAMERA_CONFIG = generate_camera_config()

def create_easy_gripper(
        color = [0,255,0,140], sections = 6, show_axis = False, width = None):
    if width:
        w = min((width + PREGRASP_PARALLEL_DISTANCE)/2, 4.1)
    else:
        w = 4.1

    l_center_grasps = GRIPPER_HEIGHT - AREA_CENTER # gripper length till grasp contact

    cfl = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[
            [w, -7.27595772e-10, 6.59999996],
            [w, -7.27595772e-10, l_center_grasps]
        ])
    cfr = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[
            [-w, -7.27595772e-10, 6.59999996],
            [-w, -7.27595772e-10, l_center_grasps],
        ])
    # arm
    cb1 = trimesh.creation.cylinder(
        radius=0.1, sections=sections,
        segment=[
            [0, 0, 0],
            [0, 0, 6.59999996]
        ])
    # queer 
    cb2 = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[
            [-w, 0, 6.59999996],
            [w, 0, 6.59999996]
        ])
    # coordinate system
    if show_axis:
        cos_system = trimesh.creation.axis(
            origin_size=0.04,
            transform=None,
            origin_color=None,
            axis_radius=None,
            axis_length=None)
        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl,cos_system])
    else:
        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color
    return tmp

def create_easy_suctioncup(
        color = [0,255,0,140], sections=6, radius = 2.5, show_axis = False):
    stick_length = SUCTIONCUP_HEIGHT
    cup_height = 1.5
    cup_radius = 1.5
    stick = trimesh.creation.cylinder(
        radius=radius,
        sections=sections,
        segment=[
            [0.0, -7.27595772e-10, 0.0],
            [0.0, -7.27595772e-10, stick_length],
        ])
    # rotate around x-axis and move to the end of stick
    cone_transform = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, stick_length+cup_height],
        [0.0, 0.0, 0.0, 1.0]
        ])

    cup = trimesh.creation.cone(
        radius=cup_radius,
        height=cup_height,
        sections=2*sections,
        transform=cone_transform)
    # coordinate system
    if show_axis:
        cos_system = trimesh.creation.axis(
            origin_size=0.04,
            transform=None,
            origin_color=None,
            axis_radius=None,
            axis_length=None)
        tmp = trimesh.util.concatenate([stick, cup, cos_system])
    else:
        tmp = trimesh.util.concatenate([stick, cup])
    tmp.visual.face_colors = color
    tmp.export("suction_gripper.ply")
    return tmp

def get_parallel_gripper_collision_mesh(root, grasp_width, transform):
    """
    Based on grasp with return different collision manager for franka gripper.
    Maximum grasp width is 8 cm.
    """
    # hand
    #hand_dir = os.path.join(root, "../hand_collision.stl")
    hand_mesh = trimesh.load(root + "/hand_collision.stl")
    hand_mesh = hand_mesh.apply_scale(100) # convert m to cm
    # finger left
    #finger_left_dir = os.path.join(root, "../finger_collsion_left.stl")
    finger_left_mesh = trimesh.load(root + "/finger_collision_left.stl")
    finger_left_mesh = finger_left_mesh.apply_scale(100)
    # finger right
    #finger_right_dir = os.path.join(root, "../finger_collsion_right.stl")
    finger_right_mesh = trimesh.load(root + "/finger_collision_right.stl")
    finger_right_mesh = finger_right_mesh.apply_scale(100)

    # create collsision manager for franka hand
    franka_hand_collision_manager = trimesh.collision.CollisionManager()
    # add hand
    hand_trans = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    hand_trans_world = np.dot(
        transform,
        hand_trans)
    franka_hand_collision_manager.add_object(
        name = "hand",
        mesh = hand_mesh,
        transform = np.array(hand_trans_world))
    
    # add finger left
    finger_left_trans = [
        [1, 0, 0, -grasp_width/2],
        [0, 1, 0, 0],
        [0, 0, 1, 5.81],
        [0, 0, 0, 1]
    ]
    finger_left_trans_world_ = np.dot(
        hand_trans_world,
        finger_left_trans)
    franka_hand_collision_manager.add_object(
        name = "finger_left",
        mesh = finger_left_mesh,
        transform = np.array(finger_left_trans_world_))
    
    # add finger right
    finger_right_trans = [
        [1, 0, 0, grasp_width/2],
        [0, 1, 0, 0],
        [0, 0, 1, 5.81],
        [0, 0, 0, 1]
    ]
    finger_right_trans_world = np.dot(
        hand_trans_world,
        finger_right_trans)
    franka_hand_collision_manager.add_object(
        name = "finger_right",
        mesh = finger_right_mesh,
        transform = np.array(finger_right_trans_world))
    return franka_hand_collision_manager

def interpolate_between_red_and_green(score, alpha = 255):
    """
    Returns RGBA color between RED and GREEN.
    """
    delta = int(score*255)
    COLOR = [255-delta, 0+delta, 0, alpha] # RGBA
    return COLOR

def load_single_mesh(_path, output_on=True, scale=100):
    if output_on:
        print(f"-> load with scale 100 [m]->[cm]{_path}")
    # Load mesh and rescale with factor 100        
    mesh = trimesh.load(_path)
    mesh = mesh.apply_scale(scale) # convert m to cm
    return mesh

def Gf_pose_2_np_4x4(pose):
    """Convert Gf pose to homographic 4x4 transform matrix as np."""
    result = np.identity(4)
    result[:3,3] = Gf.Vec3f(pose.p.x, pose.p.y, pose.p.z)
    result[:3,:3] = Gf.Matrix3f(
        Gf.Quatf(pose.r.w, Gf.Vec3f(pose.r.x, pose.r.y, pose.r.z)
        )).GetTranspose()
    return result
         
class BinScene(torch.utils.data.IterableDataset):
    """
    Description not yet created.
    """

    def __init__(
            self, root, dataset_name, categories, output_folder, collision_check, keypts_check, max_asset_size=None,
            num_assets_min=1, num_assets_max=2, split=1.0, train=True,
            show_grasps=False, debug_on=False, occlusion_on=True, idx_offset=0, randomize=False):
        self.kit = OmniKitHelper(config=RENDER_CONFIG)
        self.sd_helper = SyntheticDataHelper()
        self.dr_helper = DomainRandomization()
        self.dr_helper.toggle_manual_mode()
        self.stage = self.kit.get_stage()

        self.categories = categories
        self.range_num_assets = (num_assets_min, max(num_assets_min, num_assets_max))
        self.references = self.find_usd_assets(root, dataset_name, categories)
        self.root = root
        self.dataset_name = dataset_name
        self.idx_offset=idx_offset
        self.show_grasps = show_grasps
        self.output_folder = pathlib.Path(output_folder)
        self.occlusion_on = occlusion_on
        self.collision_check = collision_check
        self.keypts_check = keypts_check
        self.randomize_on = randomize
        self.setup_physics()
        self.setup_world()
        self.cur_idx = 0
        self.exiting = False
        self.trimesh_scene = None
        self.collision_manager = None
        self.debug = debug_on

        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, *args, **kwargs):
        print("exiting dataset generation for next scene...")
        self.exiting = True
        # self.kit.shutdown()  # Cleanup application

    def setup_physics(self):
        # Initialize dynamic control
        self.dc = dc.acquire_dynamic_control_interface()
        # Add physics scene
        self.scene = PhysicsSchema.PhysicsScene.Define(
            self.stage, Sdf.Path("/World/physicsScene"))
        # Set gravity vector
        self.scene.CreateGravityAttr().Set(Gf.Vec3f(0, 0, -981))
        # Set physics scene to use gpu physics
        PhysxSchema.PhysxSceneAPI.Apply(
            self.stage.GetPrimAtPath("/World/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(
            self.stage, "/World/physicsScene")
        physxSceneAPI.CreatePhysxSceneEnableCCDAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableStabilizationAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableGPUDynamicsAttr(True)
        physxSceneAPI.CreatePhysxSceneBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreatePhysxSceneSolverTypeAttr("TGS")
        self.kit.update()

    def setup_world(self):
        """Setup lights, walls, floor, ceiling and camera"""
        # set Z-axis pointing upwards
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        # set units to cm
        UsdGeom.SetStageMetersPerUnit(self.stage, 0.01)

        # light DR
        self.kit.create_prim(
            "/World/LightDR", "RectLight",
            translation=(0, 0, 100),
            rotation=(0, 0, 0),
            attributes={"height": 150, "width": 150, "intensity": 6000.0, "color": (1.0, 1.0, 1.0)}
        )
        # add scene
        self.kit.create_prim("/World/Asset", "Xform")
        self.load_all_assets()

        #scene_usd = self.root + "/Physics_Scene_PlasticBin.usd"
        scene_usd = self.root + "/Physics_Scene_DR.usd"
        
        print(f"-> load {scene_usd}")
        prim = self.kit.create_prim(
            "/World/Room/Scene",
            "Xform",
            ref=scene_usd)

        # generate 9 cameras, one for each viewpoint
        self.all_cameras = []
        self.num_camera_viewpts = len(CAMERA_CONFIG['viewpts'])
        for i in range(self.num_camera_viewpts):
            # get extrinsic parameters
            _phi = CAMERA_CONFIG['viewpts'][str(i)]['phi']
            _theta = CAMERA_CONFIG['viewpts'][str(i)]['theta']
            _r = CAMERA_CONFIG['viewpts'][str(i)]['radius']
            _focal_length = CAMERA_CONFIG['viewpts'][str(i)]['focal_length']
            _top_down = CAMERA_CONFIG['viewpts'][str(i)]['top_down']
            
            # set transform of each camera_rig prim
            _camera_rig = self.kit.create_prim(f"/World/CameraRig{i}", "Xform")
            xform = UsdGeom.Xformable(_camera_rig)
            transform = xform.AddTransformOp()
            mat = Gf.Matrix4d()
            if _top_down is False:
                mat.SetTranslateOnly(
                    Gf.Vec3d(
                        _r * np.cos(_phi) * np.sin(_theta),
                        _r * np.sin(_phi) * np.sin(_theta),
                        _r * np.cos(_theta)
                    ))
                _rot = Gf.Matrix3d(
                    -np.sin(_phi), np.cos(_phi), 0.0, np.cos(_theta)*np.cos(_phi), 
                    np.cos(_theta)*np.sin(_phi), -1*np.sin(_theta), 
                    -1*np.sin(_theta)*np.cos(_phi), -1*np.sin(_theta)*np.sin(_phi), -1*np.cos(_theta)
                    )
            else:
                mat.SetTranslateOnly(
                    Gf.Vec3d(
                        _r * np.cos(_phi) * np.sin(_theta),
                        _r * np.sin(_phi) * np.sin(_theta),
                        _r * np.cos(_theta)
                    ))
                _rot = Gf.Matrix3d(
                    0,1,0,
                    1,0,0,
                    0,0,-1)

            mat.SetRotateOnly(_rot)
            transform.Set(mat)
            _transform_relative_to_world = np.transpose(np.array(mat))        

            print(f"-> set up cam {i}.")
            _camera = self.kit.create_prim(f"/World/CameraRig{i}/Camera", "Camera", translation=(0.0,0.0,0.0), rotation = (180.0, 0.0, 0.0))
            # set focal length and other camera parameters
            if _focal_length is not None:
                focal = _camera.GetAttribute("focalLength")
                focal.Set(_focal_length)

            self.all_cameras.append({
                "cam" : _camera,
                "rig" : _camera_rig,
                "T" : _transform_relative_to_world,
                })
        
        # set default camera
        _camera_rig = self.kit.create_prim(
            f"/World/CameraRigDefault", "Xform",
            translation=(0.0,0.0,400),
            rotation = (180, 0, 0))
        _camera = self.kit.create_prim(
            f"/World/CameraRigDefault/Camera", "Camera",
            translation=(0.0,0.0,0.0),
            rotation = (180.0, 0.0, 0.0))
        vpi = omni.kit.viewport.get_viewport_interface()
        vpi.get_viewport_window().set_active_camera(
            f"/World/CameraRigDefault/Camera")

        # generate DR
        self.create_dr_comp()

        self.kit.update()

    def create_dr_comp(self):
        """Creates DR components with various attributes.
        The asset prims to randomize is an empty list for most components
        since we get a new list of assets every iteration.
        The asset list will be updated for each component in update_dr_comp()
        """
        material_list = glob.glob(os.path.join(self.root + "/Samples/DR/Materials", "**/*.mdl"), recursive=True)
        material_list_surface = glob.glob(os.path.join(self.root + "/Samples/DR/Surface", "**/*.mdl"), recursive=True)

        texture_list = glob.glob(os.path.join(self.root + "/Samples/DR/Materials", "**/*.png"), recursive=True)
        texture_list_surface = glob.glob(os.path.join(self.root + "/Samples/DR/Surface", "**/*.png"), recursive=True)
        
        light_list = ["/World/LightDR"]
        self.texture_comp = self.dr_helper.create_texture_comp([], True, texture_list)
        self.texture_comp_surface = self.dr_helper.create_texture_comp([], True, texture_list_surface)

        self.color_comp = self.dr_helper.create_color_comp([])
        self.material_comp = self.dr_helper.create_material_comp([], material_list)
        self.material_comp_surface = self.dr_helper.create_material_comp([], material_list_surface)

        self.light_comp = self.dr_helper.create_light_comp(light_paths=light_list, first_color_range=(0.8, 0.8, 0.8), second_color_range=(1.2, 1.2, 1.2), intensity_range=(2000.0, 12000.0), temperature_range=(4500.0, 7500.0), enable_temperature=True, duration=0.0, include_children=True, seed=random.randint(0,1000))


    def update_dr_comp(self, dr_comp, asset_path="/World/Room/Scene/KLT_Festo_low/BOX_VOL4/FOF_Mesh_Magenta_Box"):
        """Updates DR component with the asset prim paths that will be randomized"""
        comp_prim_paths_target = dr_comp.GetPrimPathsRel()
        comp_prim_paths_target.ClearTargets(True)
        comp_prim_paths_target.AddTarget(asset_path)

    def find_usd_assets(self, root, dataset_name, categories):
        """
        Look for USD files under root/category for each category specified.
        For each category, generate a list of all USD files.
        """
        references = {}
        for category in categories:
            # find paths.
            asset_usd = glob.glob(
                os.path.join(root, dataset_name, category, "**/textured.usd"),
                recursive=True)
            asset_hdf5 = glob.glob(
                os.path.join(root, dataset_name, category, "**/*.hdf5"),
                recursive=True)
            asset_obj = glob.glob(
                os.path.join(root, dataset_name, category, "**/textured.obj"),
                recursive=True)
            # assert
            if len(asset_usd) != 1:
                raise ValueError(
                    f"No or multiple USDs found for category {category}.")
            if len(asset_hdf5) != 1:
                raise ValueError(
                    f"No or multiple HDF5s found for category {category}.")
            if len(asset_obj) != 1:
                raise ValueError(
                    f"No or multiple OBJs found for category {category}.")
            references[category] = {
                "usd" : asset_usd[0],
                "hdf5" : asset_hdf5[0],
                "obj" : asset_obj[0],
                }
        return references
    
    def load_single_asset(self, ref, semantic_label, suffix=""):
        """Load a USD asset with ranplane watercraft rocketdom pose.
        args
            ref (str): Path to the USD that this prim will reference.
            semantic_label (str): Semantic label.
            suffix (str): String to add to the end of the prim's path.
        """
        # get random initial pose
        x = random.uniform(*RANDOM_TRANSLATION_X)
        y = random.uniform(*RANDOM_TRANSLATION_Y)
        z = random.uniform(*RANDOM_TRANSLATION_Z)
        rot_x = random.uniform(*RANDOM_ROTATION_X)
        rot_y = random.uniform(*RANDOM_ROTATION_Y)
        rot_z = random.uniform(*RANDOM_ROTATION_Z)
        
        print(f"-> load {ref}")
        prim = self.kit.create_prim(
            f"/World/Asset/mesh{suffix}",
            "Xform",
            scale=((1, 1, 1)),
            rotation=((rot_x, rot_y, rot_z)),
            translation=((x,y,z)),
            ref=ref,
            semantic_label=semantic_label,
        )
        return prim

    def alternate_scene(self):
        for i in range(len(self.all_assets)):
            # Change pose of asset randomly at every iteration
            x = random.uniform(*RANDOM_TRANSLATION_X)
            y = random.uniform(*RANDOM_TRANSLATION_Y)
            z = random.uniform(*RANDOM_TRANSLATION_Z)
            rot_x = random.uniform(*RANDOM_ROTATION_X)
            rot_y = random.uniform(*RANDOM_ROTATION_Y)
            rot_z = random.uniform(*RANDOM_ROTATION_Z)
            
            UsdGeom.XformCommonAPI(self.all_assets[i]).SetTranslate([x, y, z])
            UsdGeom.XformCommonAPI(self.all_assets[i]).SetRotate(
                [rot_x, rot_y, rot_z],
                UsdGeom.XformCommonAPI.RotationOrderZYX)
            

    def load_all_assets(self):
        """
        Load all usd, obj and grasps in memory.
        """
        
        print("start loading all assets into memory.")
        self.all_assets = []
        self.all_grasps = []
        self.all_meshes = []
        self.all_keypts = []
        self.all_categories = []
        idx = 0
        for i, cat in enumerate(self.categories):
            _num_obj_per_cat = random.randint(*self.range_num_assets)
            for j in range(_num_obj_per_cat):
                ref_usd = self.references[cat]["usd"] #random.choice(self.references[cat]["usd"])
                ref_hdf5 = self.references[cat]["hdf5"]
                ref_obj = self.references[cat]["obj"]
                self.all_assets.append(self.load_single_asset(ref_usd, cat, idx))
                self.all_grasps.append(load_single_grasp_config(ref_hdf5))
                self.all_meshes.append(load_single_mesh(ref_obj))
                self.all_keypts.append(load_single_keypts_config(ref_hdf5))
                self.all_categories.append(cat)
                self.kit.update()
                while self.kit.is_loading():
                    self.kit.update()
                idx += 1
        print(f"done. summary {len(self.categories)} categories, {idx} assets.")
        return

    def get_poses(self):
        """
        Saves poses of objects into memory when called.
        """
        self.all_poses = []
        self.all_lin_velocities = []
        self.all_ang_velocities = []
        for i in range(len(self.all_assets)):
            _asset_path = str(self.all_assets[i].GetPath())
            _dc_asset = self.dc.get_object(_asset_path)
            _p = self.dc.get_rigid_body_pose(_dc_asset)
            _lin_vel = self.dc.get_rigid_body_linear_velocity(_dc_asset)
            _ang_vel = self.dc.get_rigid_body_angular_velocity(_dc_asset)
            print(f"Asset {i} Position: [xyz] [{round(_p.p.x, 2), round(_p.p.y, 2), round(_p.p.z, 2)}]")
            print(f"Asset {i} Rotation: [wxyz] [{round(_p.r.w, 2), round(_p.r.x, 2), round(_p.r.y, 2), round(_p.r.z, 2)}]")
            self.all_poses.append(Gf_pose_2_np_4x4(_p))
            self.all_lin_velocities.append(_lin_vel)
            self.all_ang_velocities.append(_ang_vel)
        print("done checking poses.")
        return

    def collect_groundtruth(self, occlusion_values=None):
        """Collect groundtruth."""
        print("collect groundtruth.")
        # Change render mode to default render config.
        self.kit.set_setting("/rtx/rendermode", RENDER_CONFIG["renderer"])
        # Collect gt
        gt = self.sd_helper.get_groundtruth([
            "camera",
            "rgb",
            "depth",
            "boundingBox2DTight",
            "boundingBox2DLoose",
            "instanceSegmentation",
            "semanticSegmentation"
            ])
        
        # get camera matrix
        camera_params = self.sd_helper.get_camera_params()
        
        # RGB
        # Drop alpha channel
        image = gt["rgb"][..., :3]
        # Cast to tensor if numpy array
        if isinstance(gt["rgb"], np.ndarray):
            image = torch.tensor(image, dtype=torch.float, device="cuda")
        # Normalize between 0. and 1. and change order to channel-first.
        image = image.float() / 255.0
        image = image.permute(2, 0, 1)

        # Depth (inverse depth)
        depth = gt['depth']
        # Cast to tensor if numpy array
        if isinstance(gt["depth"], np.ndarray):
            depth = torch.tensor(depth, dtype=torch.float, device="cuda")       

        # Bounding Boxes
        gt_bbox_tight = gt["boundingBox2DTight"]
        gt_bbox_loose = gt['boundingBox2DLoose']

        #print(gt_bbox_tight)
        #print(gt_bbox_loose)

        # Create mapping from categories to index
        mapping = {cat: i + 1 for i, cat in enumerate(self.categories)}
        bboxes_tight = torch.tensor(
            gt_bbox_tight[["x_min", "y_min", "x_max", "y_max"]].tolist())
        bboxes_loose = torch.tensor(
            gt_bbox_loose[["x_min", "y_min", "x_max", "y_max"]].tolist())
        # For each bounding box, map semantic label to label index
        labels = torch.LongTensor(
            [mapping[bb["semanticLabel"]] for bb in gt_bbox_tight])


        # only objects which are at least partly visible are taken into account
        # Calculate tight bounding box area for each area : width * height
        areas = (bboxes_tight[:,2]-bboxes_tight[:,0]) * (bboxes_tight[:,3]-bboxes_tight[:,1])
        # Idenfiy invalid bounding boxes to filter final output
        valid_areas = (areas > 0.0) * (areas < (image.shape[1] * image.shape[2]))

        # Instance Segmentation
        names, masks = gt["instanceSegmentation"]
        #print("valid_areas", valid_areas)

        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks, device="cuda")

        valid_names = [
            names[id] for id, suc in enumerate(valid_areas) if suc == True]

        target = {
            "boxes_tight": bboxes_tight[valid_areas],
            "boxes_loose" : bboxes_loose[valid_areas],
            "labels": labels[valid_areas], # (list) with category labels of objects in scene 
            "masks": masks[valid_areas], # (list) of masks for each object in scene with True and False
            "names" : valid_names, # (list) of mesh names (no semantic!) of objects in scene 
            "occlusion" : occlusion_values, # (list) of occlusion values for each object in scene
            "image_id": torch.LongTensor([self.cur_idx]),
            "area": areas[valid_areas],
            "iscrowd": torch.BoolTensor([False] * len(bboxes_tight[valid_areas])),  # Assume no crowds
        }
        print("done.")

        return image, depth, target, camera_params

    def change_visibility_assets(self, mode, id=None):
        """
        mode = 0 : make invisible
        mode = 1 : make visible
        """
        assert mode == 0 or mode == 1
        if id is None:
            #print(f"change visibility {mode} of all")
            # Make all assets invisible
            for i in range(len(self.all_assets)):
                _asset_path = str(self.all_assets[i].GetPath())
                print(_asset_path)
                _imageable = UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(_asset_path))
                if mode == 0:
                    _imageable.MakeInvisible()
                else:
                    _imageable.MakeVisible()
            # bin
            bin_path =  "/World/Room/Scene/KLT_Festo_low/BOX_VOL4"

            if mode == 0:
                _imageable = UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(bin_path))
                _imageable.MakeInvisible()
            else:
                _imageable = UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(bin_path))
                _imageable.MakeVisible()

        else:
            # Only specified object id
            _asset_path = str(self.all_assets[id].GetPath())
            _imageable = UsdGeom.Imageable(
                self.stage.GetPrimAtPath(_asset_path))
            #print(f"change visibility {mode} of asset {_asset_path}")
            if mode == 0:
                _imageable.MakeInvisible()
            else:
                _imageable.MakeVisible()
        self.kit.update()
        return True
    
    def change_visibility_bin(self, mode):
        """
        Change visibility of bin. Same ideas as change_visibility_assets.
        mode = 0 : make invisible
        mode = 1 : make visible
        """
        if mode == 0:
            _imageable = UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(_asset_path))

    def check_for_occlusion(self):
        """Check for occluded objects in scene"""       
        # iterate trough scene and make objects visible individually
        ret = {}
        rgb, depth, target, _ = self.collect_groundtruth()
        # make all assets invisible
        self.change_visibility_assets(mode=0)
        # iterate through all assets, make them visible 
        for i in range(len(self.all_assets)):
            # make asset with id i visible
            _visible_asset = str(self.all_assets[i].GetPath())
            self.change_visibility_assets(mode=1, id=i)
            # check segmentation masks
            # Change render mode to default render config.
            self.kit.set_setting("/rtx/rendermode", RENDER_CONFIG["renderer"])
            # Collect gt segmentation
            gt = self.sd_helper.get_groundtruth([
                "instanceSegmentation"])
            _names , _masks = gt['instanceSegmentation']
            
            # get segmentation mask of visible object
            _visible_object_mask = None
            for mask_idx, name in enumerate(_names):
                if name == _visible_asset:
                    _visible_object_mask = _masks[mask_idx]
            assert _visible_object_mask is not None

            # look for differences
            _scene_object_mask = None
            #print("_visible_asset", _visible_asset)
            #print("true assets", target['names'])
            for mask_idx, name in enumerate(target['names']):
                if name == _visible_asset:
                    _scene_object_mask = target['masks'][mask_idx]
                    # compare masks and check for occlusion
                    unique_scene, counts_scene = np.unique(
                        _scene_object_mask.cpu(),
                        return_counts=True)
                    scene_dic = dict(zip(unique_scene, counts_scene))
                    unique_isolated, counts_isolated = np.unique(
                        _visible_object_mask,
                        return_counts=True)
                    isolated_dic = dict(zip(unique_isolated, counts_isolated))
                    #print("scene_dic", scene_dic)
                    #print("unique_dic", isolated_dic)

                    occlusion_percentage = (1 - scene_dic[True]/isolated_dic[True]) 
                    print(f"asset {_visible_asset} occluded by {occlusion_percentage*100} %.")
                    ret[_visible_asset] = occlusion_percentage
            
            self.change_visibility_assets(mode=0)

        # make everything visible
        self.change_visibility_assets(mode=1)

        return ret

    def save_dataset_offline(
                self, view_idx, image, depth, target, camera_params):
        """
        Save scene with index {self.cur_idx}. 
        """
        # RGB
        image_np = image.permute(1, 2, 0).cpu().numpy() # [0,1]
        rgb_image = PIL.Image.fromarray((image_np*255).astype(np.uint8)) # [0,255]
        rgb_path = self.scene_folder / f"{view_idx}_rgb.png"
        rgb_image.save(str(rgb_path))
        # DEPTH
        depth_np = depth.cpu().numpy() # inverse depth from isaac sim
        depth_path_np = self.scene_folder / f"{view_idx}_depth.npy"
        depth_np_mm = 1 / depth_np # inv -> in mm
        
        # normalize for better visualization
        depth_vis = depth_np
        d_max = np.amax(depth_vis)
        d_min = np.amin(depth_vis)
        depth_vis -= d_min
        if d_min != d_max:
            depth_vis = depth_vis/(d_max-d_min)
        depth_image = PIL.Image.fromarray((depth_vis*255).astype(np.uint8)) # same as REDWOOD dataset
        depth_path = self.scene_folder / f"{view_idx}_depth.png"
        depth_image.save(str(depth_path))
        # INSTANCES PIXELWISE WITH CATEGORY
        instances_semantic_np = np.zeros_like(depth_vis)
        mapping = {i + 1: cat for i, cat in enumerate(self.categories)}
        categories = [mapping[label.item()] for label in target["labels"]]
        for mask, category in zip(target["masks"].cpu().numpy(), categories):
            instances_semantic_np[mask] = category
        # INSTANCES PIXELWISE WITH UNIQUE ID
        instances_objects_np = np.zeros_like(depth_vis)
        object_ids = {i + 1: i for i in range(len(target['names']))}
        # print("object_ids", object_ids)
        for mask, object_id in zip(target["masks"].cpu().numpy(), object_ids):
            # print("object_id", object_id)
            instances_objects_np[mask] = object_id
        # OCCLUSION
        occlusions_np = np.zeros_like(depth_vis)
        if self.occlusion_on or view_idx == 0:
            occlusions = [round(occlusion,4) for occlusion in target['occlusion'].values()] 
            for mask, occlusion in zip(target["masks"].cpu().numpy(), occlusions):
                occlusions_np[mask] = occlusion

        # use different data type to reduce storage
        
        # save as compressed numpy file
        np.savez_compressed(str(self.scene_folder / f"{view_idx}"),
            depth=depth_np_mm,
            instances_semantic=instances_semantic_np,
            instances_objects=instances_objects_np,
            occlusion=occlusions_np)
        # CAMERA PARAMS
        json_path = self.scene_folder / f"{view_idx}_camera_params.json"
        _focal_length = camera_params['focal_length']
        _width = camera_params['resolution']['width']
        _height = camera_params['resolution']['height']
        _horizontal_fov = camera_params['fov']
        _horizontal_aperture = camera_params['horizontal_aperture']
        _vertical_fov = _height / _width * _horizontal_fov
        _vertical_aperture = _height/_width * _horizontal_aperture
        _fx = _width * 0.5 / np.tan(_horizontal_fov*0.5)
        _fy = _height * 0.5 / np.tan(_vertical_fov*0.5)
        _json_dict = {
            "horizontal_fov" : _horizontal_fov,
            "vertical_fov" : _vertical_fov,
            "focal_length" : camera_params['focal_length'],
            "fx" : _fx,
            "fy" : _fy,
            "horizontal_aperture" : _horizontal_aperture,
            "vertical_aperture" : _vertical_aperture,
            "resolution" : camera_params['resolution'],
            "clipping_range" : camera_params['clipping_range']
        }
        with open(str(json_path), 'w') as f:
            json.dump(_json_dict, f, sort_keys=True)
        
        ## HDF5 - File 
        # create scene hdf5 description
        scene_h5py_file = h5py.File(
            str(self.scene_folder / f"{view_idx}_scene.hdf5"), 'w')
        
        ## add visible objects' poses
        grp_objects = scene_h5py_file.create_group("objects")
        visible_objects_idx = []
        for idx in range(len(self.all_assets)):
            if self.all_assets[idx].GetPath() in target['names']:
                visible_objects_idx.append(idx)
        num_objects_in_scene = len(visible_objects_idx)
        
        obj_dset = grp_objects.create_dataset(
            name="poses_relative_to_camera",
            shape=(num_objects_in_scene, 4, 4))
        categories_dset = grp_objects.create_dataset(
            name="categories",
            shape=(num_objects_in_scene, ))
        # add bbox
        bbox_dset = grp_objects.create_dataset(
            name="bbox_loose",
            shape=(num_objects_in_scene, 4)
        )

        current_cam_pose_world = np.array(self.all_cameras[view_idx]['T'])
        print("visible_obejcts_idx", visible_objects_idx)

        for idx, visible_obj_idx in enumerate(visible_objects_idx):
            pose_relative_to_world = self.all_poses[visible_obj_idx]
            pose_relative_to_camera = np.dot(
                    np.linalg.inv(current_cam_pose_world),
                    pose_relative_to_world)
            obj_dset[idx,:,:] = pose_relative_to_camera
            categories_dset[idx] = float(self.all_categories[visible_obj_idx])
            bbox_dset[idx] = dataset_utilities.bbox2coco(*target['boxes_loose'][idx].tolist())


        ## add viewpt camera
        grp_camera = scene_h5py_file.create_group("camera")
        cam_dset = grp_camera.create_dataset(
            name="pose_relative_to_world",
            shape=(1, 4, 4))
        cam_dset[0] = current_cam_pose_world
        
        ## add non colliding grasps
        grp_grasps = scene_h5py_file.create_group("non_colliding_grasps")

        if self.collision_check:
            # check if parallel grasp is visible for current camera
            visible_parallel_grasps_idx = []
            for idx in range(len(self.SceneCollision.non_colliding_parallel_gripper_poses)):
                if self.SceneCollision.non_colliding_parallel_asset_names[idx] in target['names']:
                    visible_parallel_grasps_idx.append(idx)
                else:
                    if self.debug:
                        print(f"-> parallel grasp {idx} not visible")

            if self.debug:
                print("visible_parallel_grasps_idx", visible_parallel_grasps_idx)

            # only store visible grasps
            # parallel gripper
            paralleljaw_subgrp = grp_grasps.create_group("paralleljaw")
            num_non_colliding_parallel_grasps = len(visible_parallel_grasps_idx)
            parallel_gripper_dset = paralleljaw_subgrp.create_dataset(
                name="franka_poses_relative_to_camera",
                shape=(num_non_colliding_parallel_grasps, 4, 4))
            
            parallel_analytical_score_dset = paralleljaw_subgrp.create_dataset(
                name="score_analytical",
                shape=(num_non_colliding_parallel_grasps, ))
            
            parallel_simulation_score_dset = paralleljaw_subgrp.create_dataset(
                name="score_simulation",
                shape=(num_non_colliding_parallel_grasps, ))

            parallel_object_id_dset = paralleljaw_subgrp.create_dataset(
                name="object_id",
                shape=(num_non_colliding_parallel_grasps, ))    
            
            parallel_contact_dset = paralleljaw_subgrp.create_dataset(
                name="contact_poses_relative_to_camera",
                shape=(num_non_colliding_parallel_grasps, 4, 4))

            parallel_width_dset = paralleljaw_subgrp.create_dataset(
                name="contact_width",
                shape=(num_non_colliding_parallel_grasps, ))
            


            for i, idx in enumerate(visible_parallel_grasps_idx):
                # only add if object is in fov of camera
                assert self.SceneCollision.non_colliding_parallel_asset_names[idx] in target['names']

                if self.debug:
                    print(f"i : {i} idx : {idx}")

                gripper_pose_relative_to_camera = np.dot(
                        np.linalg.inv(current_cam_pose_world),
                        self.SceneCollision.non_colliding_parallel_gripper_poses[idx])
                    
                contact_pose_relative_to_camera = np.dot(
                        np.linalg.inv(current_cam_pose_world),
                        self.SceneCollision.non_colliding_parallel_contact_poses[idx])

                parallel_gripper_dset[i,:,:] = gripper_pose_relative_to_camera
                parallel_analytical_score_dset[i] = self.SceneCollision.non_colliding_parallel_analytical_score[idx]
                parallel_simulation_score_dset[i] = self.SceneCollision.non_colliding_parallel_simulation_score[idx]
                parallel_object_id_dset[i] = self.SceneCollision.non_colliding_parallel_object_id[idx]
                parallel_contact_dset[i,:,:] = contact_pose_relative_to_camera
                parallel_width_dset[i] = self.SceneCollision.non_colliding_parallel_contact_width[idx]


            # check if suction grasp is visible for current camera
            visible_suction_grasps_idx = []
            for idx in range(len(self.SceneCollision.non_colliding_suction_gripper_poses)):
                if self.SceneCollision.non_colliding_suction_asset_names[idx] in target['names']:
                    visible_suction_grasps_idx.append(idx)
                else:
                    if self.debug:
                        print(f"-> suction grasp {idx} not visible.")

            # suction gripper
            suction_subgrp = grp_grasps.create_group("suctioncup")
            num_non_colliding_suction_grasps = len(visible_suction_grasps_idx)
            suction_gripper_dset = suction_subgrp.create_dataset(
                name="suction_poses_relative_to_camera",
                shape=(num_non_colliding_suction_grasps, 4, 4))
            
            suction_analytical_score_dset = suction_subgrp.create_dataset(
                name="score_analytical",
                shape=(num_non_colliding_suction_grasps, ))
            
            suction_simulation_score_dset = suction_subgrp.create_dataset(
                name="score_simulation",
                shape=(num_non_colliding_suction_grasps, ))

            suction_object_id_dset = suction_subgrp.create_dataset(
                name="object_id",
                shape=(num_non_colliding_suction_grasps, ))    
            
            suction_contact_dset = suction_subgrp.create_dataset(
                name="contact_poses_relative_to_camera",
                shape=(num_non_colliding_suction_grasps, 4, 4))
                
            for i, idx in enumerate(visible_suction_grasps_idx):
                # only add if object is in fov of camera
                assert self.SceneCollision.non_colliding_suction_asset_names[idx] in target['names']
                if self.debug:
                    print("visible_suction_grasps_idx", visible_suction_grasps_idx)
                
                gripper_pose_relative_to_camera = np.dot(
                        np.linalg.inv(current_cam_pose_world),
                        self.SceneCollision.non_colliding_suction_gripper_poses[idx])
                    
                contact_pose_relative_to_camera = np.dot(
                        np.linalg.inv(current_cam_pose_world),
                        self.SceneCollision.non_colliding_suction_contact_poses[idx])
                    
                suction_gripper_dset[i,:,:] = gripper_pose_relative_to_camera
                suction_analytical_score_dset[i] = self.SceneCollision.non_colliding_suction_analytical_score[idx]
                suction_simulation_score_dset[i] = self.SceneCollision.non_colliding_suction_simulation_score[idx]
                suction_object_id_dset[i] = self.SceneCollision.non_colliding_suction_object_id[idx]
                suction_contact_dset[i,:,:] = contact_pose_relative_to_camera
        
        grp_keypts = scene_h5py_file.create_group("keypts")
        if self.keypts_check:
            ## check scene for visible keypts
            # create instance
            KeypointInstance = KeypointsInScene(
                meshes=copy.deepcopy(self.all_meshes), # all meshes, not only visible
                poses=self.all_poses, # all poses not only visible
                keypts=self.all_keypts, # all keypts of all meshes
                cam_pose=current_cam_pose_world,
                box_dir=self.root + "/KLT/BOX_VOL4.obj",
                assets=self.all_assets
            )
            KeypointInstance.set_up_scene()
            KeypointInstance.check_for_keypts(simulation=True, visualize=False)

            # BYHAND
            # check if visible keypt is visible for current fov
            visible_fov_keypts_byhand_idx = []
            for idx in range(len(KeypointInstance.visible_keypts_byhand_world)):
                if KeypointInstance.visible_keypts_byhand_asset_name[idx] in target['names']:
                    visible_fov_keypts_byhand_idx.append(idx)
                else:
                    if self.debug:
                        print(f"-> keypt byhand {idx} not visible")

            byhand_subgrp = grp_keypts.create_group("byhand")
            num_visible_fov_keypts_byhand = len(visible_fov_keypts_byhand_idx)
            keypts_byhand_camera_dset = byhand_subgrp.create_dataset(
                name="keypts_relative_to_camera",
                shape=(num_visible_fov_keypts_byhand, 4))
            keypts_byhand_object_id_dset = byhand_subgrp.create_dataset(
                name="object_id",
                shape=(num_visible_fov_keypts_byhand, ))
            
            for i, idx in enumerate(visible_fov_keypts_byhand_idx):
                # only add if object is in fov of camera
                assert KeypointInstance.visible_keypts_byhand_asset_name[idx] in target['names']
                pt_id = KeypointInstance.visible_keypts_byhand_world[idx][0]
                pt = np.array(KeypointInstance.visible_keypts_byhand_world[idx][1:4])
                pt_homogeneous = np.array([pt[0], pt[1], pt[2], 1.0])
                pt_rel_cam = np.dot(
                    np.linalg.inv(current_cam_pose_world),
                    pt_homogeneous)

                keypts_byhand_camera_dset[i,:] = [pt_id, pt_rel_cam[0], pt_rel_cam[1], pt_rel_cam[2]]
                keypts_byhand_object_id_dset[i] = KeypointInstance.visible_keypts_byhand_object_id[idx]

            # Center Of Mass (COM)
            visible_fov_keypts_com_idx = []
            for idx in range(len(KeypointInstance.visible_keypts_com_world)):
                if KeypointInstance.visible_keypts_com_asset_name[idx] in target['names']:
                    visible_fov_keypts_com_idx.append(idx)
                else:
                    if self.debug:
                        print(f"-> keypt com {idx} not visible")

            com_subgrp = grp_keypts.create_group("com")
            num_visible_fov_keypts_com = len(visible_fov_keypts_com_idx)
            keypts_com_camera_dset = com_subgrp.create_dataset(
                name="keypts_relative_to_camera",
                shape=(num_visible_fov_keypts_com, 4))
            keypts_com_object_id_dset = com_subgrp.create_dataset(
                name="object_id",
                shape=(num_visible_fov_keypts_com, ))
            
            for i, idx in enumerate(visible_fov_keypts_com_idx):
                # only add if object is in fov of camera
                assert KeypointInstance.visible_keypts_com_asset_name[idx] in target['names']
                pt_score = KeypointInstance.visible_keypts_com_world[idx][0]
                pt = np.array(KeypointInstance.visible_keypts_com_world[idx][1:4])
                pt_homogeneous = np.array([pt[0], pt[1], pt[2], 1.0])
                pt_rel_cam = np.dot(
                    np.linalg.inv(current_cam_pose_world),
                    pt_homogeneous)

                keypts_com_camera_dset[i,:] = [pt_score, pt_rel_cam[0], pt_rel_cam[1], pt_rel_cam[2]]
                keypts_com_object_id_dset[i] = KeypointInstance.visible_keypts_com_object_id[idx]

        scene_h5py_file.close()

        return

    def alternate_camera(self, view_idx):
        """Randomize the camera position."""
        # By simply rotating a camera "rig" instead repositioning the camera
        # itself, we greatly simplify our job.

        # set active camera according to view_idx
        vpi = omni.kit.viewport.get_viewport_interface()
        _cam = str(self.all_cameras[view_idx]['cam'].GetPath())
        vpi.get_viewport_window().set_active_camera(_cam)
        print(f"-> activate cam {_cam}.")
        return

    def create_scene_folder(self):
        # create data folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        # create scene folder in data folder
        self.scene_folder = self.output_folder / f"scene{self.cur_idx + self.idx_offset}"
        self.scene_folder.mkdir(parents=True, exist_ok=True)
        return

    def __iter__(self):
        return self

    def __next__(self):
        
        # Generate a new scene
        # self.populate_scene()
        self.alternate_scene()
        # set deault cam
        vpi = omni.kit.viewport.get_viewport_interface()
        vpi.get_viewport_window().set_active_camera(
            f"/World/CameraRigDefault/Camera")
        self.kit.update()

        # randomize once
        if self.randomize_on:
            print("randomize szene.")
            self.update_dr_comp(self.material_comp, "/World/Room/Scene/KLT_Festo_low/BOX_VOL4/FOF_Mesh_Magenta_Box")
            self.update_dr_comp(self.texture_comp, "/World/Room/Scene/KLT_Festo_low/BOX_VOL4/FOF_Mesh_Magenta_Box")
            self.update_dr_comp(self.color_comp, "/World/Room/Scene/KLT_Festo_low/BOX_VOL4/FOF_Mesh_Magenta_Box")
            self.update_dr_comp(self.material_comp_surface, "/World/Room/Scene/Ground/Plane")
            self.update_dr_comp(self.texture_comp_surface, "/World/Room/Scene/Ground/Plane")
        
            self.dr_helper.randomize_once()
            self.kit.update()

        # step once and then wait for materials to load
        self.kit.update()
        print("waiting for materials to load...")
        while self.kit.is_loading():
            self.kit.update()
        print("done")
        
        # start physics simulation 
        print("starting physics simulation...")
        # force RayTracedLighting mode for better performance 
        # while simulating physics
        self.kit.set_setting("/rtx/rendermode", "RayTracedLighting")
        self.kit.play()

        frame = 0
        while frame < SIMULATION_CONFIG['simulation_time'] / SIMULATION_CONFIG['dt_rendering']:
            self.kit.update(
                dt=SIMULATION_CONFIG['dt_rendering'],
                physics_dt=SIMULATION_CONFIG['dt_physics'],
                physics_substeps=SIMULATION_CONFIG['physics_substeps'])    
            frame += 1
        print("done")
        
        # Get Poses in World COS
        self.get_poses()
        
        # create file folder to store dataset offline
        self.create_scene_folder()

        if self.collision_check:
            ## Check Grasps for collision with scene
            # create instance
            self.SceneCollision = CollisionWithScene(
                meshes=self.all_meshes,
                poses=self.all_poses,
                grasps=self.all_grasps,
                root_dir=self.root,
                assets=self.all_assets,
                box_dir=self.root + "/KLT/BOX_VOL4.obj")

            # create collision scene and trimesh scene
            self.SceneCollision.load_potential_grasps_and_generate_trimesh_scene(simulation=True)
            # check for collision
            self.SceneCollision.check_for_collision(simulation=True, show_scene=False)
        
        # iterate over different camera poses
        for cur_view_idx in range(self.num_camera_viewpts):
            # Activate Camera
            self.alternate_camera(cur_view_idx)
            if self.randomize_on:
                self.dr_helper.randomize_once()
                self.update_dr_comp(self.material_comp, "/World/Room/Scene/KLT_Festo_low/BOX_VOL4/FOF_Mesh_Magenta_Box")
                self.update_dr_comp(self.texture_comp, "/World/Room/Scene/KLT_Festo_low/BOX_VOL4/FOF_Mesh_Magenta_Box")
                self.update_dr_comp(self.material_comp_surface, "/World/Room/Scene/Ground/Plane")
                self.update_dr_comp(self.texture_comp_surface, "/World/Room/Scene/Ground/Plane")

            self.kit.update()
            # Collect GT
            print("self.occlusion_on ", self.occlusion_on)
            cur_occlusion_values = self.check_for_occlusion() if (self.occlusion_on or cur_view_idx == 0) else None
            cur_image, cur_depth, cur_target, cur_camera_params = self.collect_groundtruth(cur_occlusion_values)
            # Save gt dataset offline
            """
            self.save_dataset_offline(
                view_idx=cur_view_idx,
                image=cur_image,
                depth=cur_depth,
                target=cur_target,
                camera_params=cur_camera_params)
            """
        # Stop simulation
        self.kit.stop()

        # Increment one
        self.cur_idx += 1
        return cur_image, cur_depth, cur_target

def start_simulation(
            categories, root, dataset_name, output_folder, min_objects_per_category,
            max_objects_per_category, occlusion_for_all_scenes, idx_offset, ifl_models,
            stop_counter=None, grasp_collision_check=True, keypts_check=True, randomize=False, show_grasps=False, debug=False):

        if categories is not None:
            # selected categories
            if ifl_models:
                selection = [
                    dataset_utilities.IFL_LABEL_TO_SYNSET.get(c, c) for c in categories
                ]
            else:
                selection = [
                    dataset_utilities.GRASPNET_LABEL_TO_SYNSET.get(c, c) for c in categories
                ]
        else:
            print("Error : specify [--categories] or set [--random_categories]")
            return False

        print("Selected Models", selection)

        if grasp_collision_check is False:
            print("[WARNING] no collision check for grasps selected!")

        # Generate dataset object
        dataset = BinScene(
            root=root,
            dataset_name=dataset_name,
            categories=selection,
            output_folder = output_folder, 
            num_assets_min=min_objects_per_category, 
            num_assets_max=max_objects_per_category,  
            show_grasps=show_grasps,
            debug_on=debug,
            occlusion_on=occlusion_for_all_scenes,
            collision_check=grasp_collision_check,
            keypts_check=keypts_check,
            idx_offset=idx_offset,
            randomize=randomize)

        for scene_idx, (image, depth, target) in enumerate(dataset):
            print(f"Scene {scene_idx} | {scene_idx + idx_offset}")
            if dataset.exiting or scene_idx == stop_counter-1:
                break
        
        # TODO shutdown
        dataset.handle_exit()

        return scene_idx



if __name__ == "__main__":
    "Typical usage"
    import argparse
    import matplotlib.pyplot as plt
    from omni.isaac.synthetic_utils import visualization as vis

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument(
        "--categories", type=str,
        nargs="+",
        help="List of object classes to use")
    parser.add_argument(
        "--min_objects_per_category", type=int, 
        default=2,
        help="Maximum number of categories to load in GPU memory.")
    parser.add_argument(
        "--max_objects_per_category", type=int, 
        default=5,
        help="Maximum number of categories to load in GPU memory.")
    parser.add_argument(
        "--root", type=str,
        default="../models",
        help="Root directory containing USDs.")
    parser.add_argument(
        "--dataset_name", type=str,
        default="models_ifl",
        help="Dataset foldername containing USDs.")
    parser.add_argument(
        "--output_folder", type=str,
        default="./data",
        help="Folder to store data.")
    parser.add_argument(
        "--idx_offset", type=int,
        default=0,
        help="Offset of scene idx to store dataset.")
    parser.add_argument(
        "--stop_counter", type=int,
        default=30000,
        help="Stop simulation after XXX scene generations.")
    parser.add_argument(
        "--collect_gt", 
        action="store_true",
        help="Set flag to collect groundtruth.")
    parser.add_argument(
        "--show_grasps",
        action="store_true",
        help="Set flag to show trimesh scene with grasps.")
    parser.add_argument(
        "--occlusion_for_all_scenes",
        default=False, action="store_true",
        help="Set flag to compute occlusion score for all viewpts.")
    parser.add_argument(
        "--grasp_collision_check",
        default=False, action="store_true",
        help="Set flag to check for collision with scene.")
    parser.add_argument(
        "--keypts_check",
        default=False, action="store_true",
        help="Set flag to check for visible object keypts in scene.")
    parser.add_argument(
        "--randomize",
        default=False, action="store_true",
        help="Set flag to check for visible object keypts in scene.")
    
    args = parser.parse_args()

    start_simulation(
        categories=args.categories,
        root=args.root,
        dataset_name=args.dataset_name,
        output_folder=args.output_folder,
        min_objects_per_category=args.min_objects_per_category,
        max_objects_per_category=args.max_objects_per_category,
        occlusion_for_all_scenes=args.occlusion_for_all_scenes,
        idx_offset=args.idx_offset,
        stop_counter=args.stop_counter,
        grasp_collision_check=args.grasp_collision_check,
        keypts_check=args.keypts_check,
        randomize=args.randomize,
        show_grasps=False,
        debug=False,
        ifl_models=True)
    
