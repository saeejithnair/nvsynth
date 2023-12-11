from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1024, "height": 1024, "num_frames": 100}
kit = SimulationApp(launch_config=CONFIG)


import omni.replicator.core as rep


# This allows us to run replicator, which will update the random
# parameters and save out the data for as many frames as listed
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

with rep.new_layer():
    # Define paths for the character, the props, the environment and the surface where the assets will be scattered in.
    PROPS = 'omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Props/YCB/Axis_Aligned_Physics'
    # PROPS = '/pub2/nrc/aging/snair_cvis_data'
    SURFACE = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd'
    ENVS = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Interior/ZetCG_ExhibitionHall.usd'

    # Define randomizer function for Base assets. This randomization includes placement and rotation of the assets on the surface.
    def env_props(size=50):
        instances = rep.randomizer.instantiate(rep.utils.get_usd_files(PROPS, recursive=True), size=size, mode='scene_instance')
        with instances:
            rep.modify.pose(
                position=rep.distribution.uniform((-50, 5, 50), (50, 20, 50)),
                rotation=rep.distribution.uniform((0,-180, 0), (0, 180, 0)),
                scale = 10
            )

            rep.physics.rigid_body(
                velocity=rep.distribution.uniform((-0,0,-0),(0,0,0)),
                angular_velocity=rep.distribution.uniform((-0,0,-100),(0,0,0)))
        return instances.node

    # Register randomization
    rep.randomizer.register(env_props)
    # Setup the static elements
    env = rep.create.from_usd(ENVS)
    surface = rep.create.from_usd(SURFACE)
    # with surface:
        # rep.physics.collider()
        # rep.modify.pose(rotation=(0,-180,0))

    # with env:
    #     rep.modify.pose(rotation=(0,-180,0))

    # Setup camera and attach it to render product
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    # sphere lights for extra randomization
    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(35000, 5000),
            position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
            scale=rep.distribution.uniform(50, 100),
            count=num
        )
        return lights.node
    rep.randomizer.register(sphere_lights)
    # trigger on frame for an interval

    with rep.trigger.on_time(interval=2,num=10):
        ycb_objects = rep.randomizer.env_props(30)
        rep.randomizer.sphere_lights(10)
        with camera:
            rep.modify.pose(position=(50, 50, 150), look_at=ycb_objects)
    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize( output_dir="/home/smnair/work/nutrition/_output/ycb_physics", rgb=True,   bounding_box_2d_tight=True)
    writer.attach([render_product])

    run_orchestrator()

    kit.update()
    kit.close()
