import re
from typing import List, Optional, Tuple

import omni
import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.core import World
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from foodverse.configs import sim_configs as sc
from foodverse.utils import geometry_utils as gu


class Scene:
    def __init__(self, kit) -> None:
        """Initialize scene."""
        self.kit = kit
        self.world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)

        self.stage = self.kit.context.get_stage()

        # Define the scope that all generated prims will be parented to.
        self.scope_name = sc.SCOPE_NAME
        UsdGeom.Scope.Define(self.stage, self.scope_name)

        self.initialized_orchestrator = False

    def cleanup(self) -> None:
        rep.BackendDispatch.wait_until_done()
        rep.orchestrator.stop()
        self.kit.update()
        self.kit.close()

    def initialize_orchestrator(self) -> None:
        """Initialize the orchestrator."""
        rep.orchestrator.step()
        self.initialized_orchestrator = True

    def step_orchestrator(self, max_attempts: int = 100) -> None:
        """Step orchestrator until a new frame is available or
        max_attempts is reached. WARNING: This function is based on
        omni/replicator/core/scripts/orchestrator.py:step() but does not
        perform any of the safety checks they execute. Use with caution.

        Args:
            max_attempts: Maximum number of attempts to wait for a new frame.

        Returns:
            Number of attempts it took to get a new frame.
        """
        if not self.initialized_orchestrator:
            self.initialize_orchestrator()

        # Wait for a frame to be available
        attempts = 0
        # It appears that a new frame becomes available every
        # samples_per_pixel_per_frame attempts. Why is this the case?
        while (
            len(rep.orchestrator.get_frames_to_write()) == 0 or attempts > max_attempts
        ):
            self.kit.update()
            attempts += 1

        attempts = 0
        cur_frame = rep.orchestrator.get_frames_to_write()[-1]
        # continue_stepping = True
        # while(continue_stepping):

        while (
            rep.orchestrator.get_is_started()
            and cur_frame == rep.orchestrator.get_frames_to_write()[-1]
            or attempts > max_attempts
        ):
            attempts += 1
            self.kit.update()

        return attempts

    def compute_prim_position(self, prim: Usd.Prim):
        matrix = omni.usd.get_world_transform_matrix(prim)
        translation = matrix.ExtractTranslation()

        return translation

    def apply_physics_to_prim(
        self,
        prim: Usd.Prim,
        approximation_shape: Optional[str] = None,
        contact_offset: float = 0.1,
        rest_offset: float = 0.1,
    ) -> None:
        """Enables static collision for prim.
        This turns the prim into a static collider that other objects can
        collide with. However, the prim will not move. To enable dynamic
        collision, rigid body physics will need to be enabled for the prim.

        Args:
            prim: Prim to apply physics to.
            approximation_shape: The approximation used in the collider
                (by default, convexHull). Other approximations
                include "convexDecomposition", "boundingSphere",
                "boundingCube", "meshSimplification", and "none".
                "none" will just use default mesh geometry.
            contact_offset: Distance from the surface of the collision
                geometry at which contacts start being generated.
            rest_offset: Distance from the surface of the collision
                geometry at which the effective contact with the shape
                takes place.

        Returns:
            None
        """
        # https://forums.developer.nvidia.com/t/unexpected-collision-behaviour-with-replicator-python-api/233575/9
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

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
        prim.GetAttribute("physxCollision:contactOffset").Set(contact_offset)

        # The Rest Offset defines a small distance from the surface
        # of the collision geometry at which the effective contact with
        # the shape takes place. It can be both positive and negative, and
        # may be useful in cases where the visualization mesh is slightly
        # smaller than the collision geometry: setting an appropriate
        # negative rest offset results in the contact occurring at the
        # visually correct distance.
        prim.GetAttribute("physxCollision:restOffset").Set(rest_offset)

    def apply_material_to_prim(
        self, prim: Usd.Prim, material_properties: sc.MaterialProperties
    ) -> None:
        """Applies a physics material to a prim.
        This simulates the friction and restitution of a surface.

        Args:
            prim: Prim to apply physics material to.
            material_properties: Dictionary containing the physics material.
                TODO(snair): Change this to a NamedTuple.

        Returns:
            None
        """
        prim_path = prim.GetPath()
        material_path = prim_path.AppendChild("PhysicsMaterial")
        UsdShade.Material.Define(self.stage, material_path)

        material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(material_path))
        material.CreateStaticFrictionAttr().Set(material_properties.static_friction)
        material.CreateDynamicFrictionAttr().Set(material_properties.dynamic_friction)
        material.CreateRestitutionAttr().Set(material_properties.restitution)

    def get_prims(
        self,
        path_pattern: str = None,
        path_pattern_exclusion: str = None,
        prim_types: str = None,
        prim_types_exclusion: str = None,
        semantics: str = None,
        semantics_exclusion: str = None,
        cache_prims: bool = True,  # ignored
    ) -> List[Usd.Prim]:
        """
        Returns a list of prims that match the given criteria.
        TODO(snair): Add better documentation.
        """
        # Code modified from:
        # isaac_sim/exts/omni.replicator.core-1.4.3+lx64.r.cp37/omni/replicator/core/ogn/python/_impl/nodes/OgnGetPrims.py
        # rep.get.prims returns ReplicatorItems, which aren't very useful
        # this function does the same as rep.get.prims but
        # returns the prim paths instead of Replicator Items.

        gathered_prims = []
        for prim in self.stage.Traverse():
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
                if not any(
                    [prim_semantic in semantics for prim_semantic in prim_semantics]
                ):
                    continue
            if semantics_exclusion:
                if any(
                    [
                        prim_semantic in semantics_exclusion
                        for prim_semantic in prim_semantics
                    ]
                ):
                    continue

            gathered_prims.append(prim)
        return gathered_prims

    def create_render_products(
        self,
        look_at_prim: Usd.Prim,
        pos: Tuple[float, float, float],
        num_cameras: int,
        radius: float,
        focal_length: float = sc.CAMERA_FOCAL_LENGTH,
        resolution: Tuple[int, int] = (sc.CAMERA_HEIGHT, sc.CAMERA_WIDTH),
    ) -> List[og.Node]:
        """
        Generates camera poses on the surface of a sphere.

        Args:
            look_at_prim: Prim to look at
            pos: Initial position of the camera
            num_cameras: Number of cameras to generate
            radius: Radius of the fibonacci sphere. NOTE: Cameras will still
                be generated slightly beyond the sphere.
            focal_length: Focal length of the camera
            resolution: Resolution of the camera (height, width)

        Returns:
            List of render products to call writer.attach(...) on
        """
        pts = gu.make_views(pos, num_cameras, radius)
        render_products = []
        for idx, pt in enumerate(pts):
            focal_length_scale = 1.0 if idx % 2 == 0 else 1.5
            camera = rep.create.camera(
                position=pt,
                look_at=str(look_at_prim.GetPrimPath()),
                focal_length=focal_length * focal_length_scale,
            )

            render_product = rep.create.render_product(camera, resolution)
            render_products.append(render_product)

        return render_products
