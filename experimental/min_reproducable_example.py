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


import omni.replicator.core as rep
with rep.new_layer():
    with rep.create.cube():
        rep.physics.collider()
        rep.modify.pose(position=(0,0,10), scale=0.01)
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
            mass=10,
        )

    floor = rep.create.plane(position=(0,0,0), scale=0.1)

    with floor:
        rep.physics.collider()

    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1024, 1024))

    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=6000,
            intensity=30000,
            position=rep.distribution.uniform((-3, -3, 12), (-3, 3, 15)),
            scale=1,
            count=num
        )
        return lights.node
    rep.randomizer.register(sphere_lights)

    with rep.trigger.on_time(interval=1000,num=1):
        rep.randomizer.sphere_lights(10)

        with camera:
            rep.modify.pose(position=(16, 0, 12), rotation=(0, -15, 0))


    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="/tmp/omni_output/", rgb=True)
    writer.attach([render_product])

    run_orchestrator()
kit.update()
kit.close()
    

