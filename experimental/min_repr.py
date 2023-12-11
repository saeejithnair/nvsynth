from omni.isaac.kit import SimulationApp

kit = SimulationApp({'headless': False, 'renderer': "RayTracedLighting", 'num_frames': 100})


def run_orchestrator():
    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        kit.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        kit.update()

    rep.BackendDispatch.wait_until_done()

    rep.orchestrator.stop()


def apply_physics_to_prim(prim):
    # https://forums.developer.nvidia.com/t/unexpected-collision-behaviour-with-replicator-python-api/233575/9
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.GetAttribute("physxCollision:contactOffset").Set(0.001)
    prim.GetAttribute("physxCollision:restOffset").Set(0.001)


def apply_physics(prim_paths):
    prims = rep.utils.find_prims(prim_paths, mode='prims')
    for prim in prims:
        apply_physics_to_prim(prim)


    # for path, prim in zip(prim_paths, prims):
    #     apply_physics_to_prim(prim)

    #     if "*" in path:
    #         n_prims = int(path.split("*")[1])
    #         for _ in range(n_prims-1):
    #             prim = prim.GetNextSibling()
    #             apply_physics_to_prim(prim)


import omni
import omni.replicator.core as rep
from pxr import UsdPhysics, PhysxSchema
with rep.new_layer():
    floor = rep.create.plane(position=(0,0,0), scale=0.5)

    with floor:
        rep.physics.collider()

    n_cubes = 10
    cubes = [rep.create.cube() for _ in range(n_cubes)]
    with rep.create.group(cubes):
        rep.physics.collider()
        rep.modify.pose(position=rep.distribution.uniform((-1, -1, 50), (1, 1, 50)), scale=0.03)
        rep.physics.rigid_body(
                velocity=0,
                angular_velocity=100
        )
        rep.physics.physics_material(
            static_friction=0.7,
            dynamic_friction=0.5,
            restitution=0.2
        )


    cube_path = "/Replicator/Cube_Xform"
    prim_paths = [cube_path]
    prim_paths.extend([cube_path + f"_{i+1:02d}" for i in range(n_cubes-1)])
    prim_paths.append("/Replicator/Plane_Xform")
    apply_physics(prim_paths)

    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1024, 1024))

    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=6000,
            intensity=30000,
            position=rep.distribution.uniform((-3, -3, 60), (3, 3, 80)),
            scale=1,
            count=num
        )
        return lights.node
    rep.randomizer.register(sphere_lights)

    with rep.trigger.on_time(interval=1000,num=1):
        rep.randomizer.sphere_lights(10)

        with camera:
            rep.modify.pose(position=(160, 0, 20), look_at=floor)


    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="/tmp/omni_output/", rgb=True)
    writer.attach([render_product])

    run_orchestrator()
kit.update()
kit.close()
    

