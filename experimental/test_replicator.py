from omni.isaac.kit import SimulationApp
import argparse


CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1024, "height": 1024}
kit = SimulationApp(launch_config=CONFIG)

import omni.replicator.core as rep
import carb
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni

camera1_pos = [(0, 1, 5),(5, 1, 5),(-5, 1, 5),]
camera2_pos = [(0, 5, 0),(5, 5, 0),(-5, 5, 0),]

PLATE_USD_PATH = "/home/smnair/work/nutrition/vip-omni/assets/plate/plate.usd"
# PLATE_USD_PATH = "/pub2/nrc/aging/snair_cvis_data/id-12-carrot-9g/textured_obj.usd"
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


def run_with_sim_context():
    from omni.isaac.core import SimulationContext
    """WARNING: EXPERIMENTAL DOESNT WORK"""
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()

    for i in range(10000):
        simulation_context.step(render=True)

    simulation_context.stop()

MUSTARTD_USD_PATH = '/home/smnair/work/nutrition/vip-omni/assets/mustard/006_mustard_bottle.usd'
MUSTARD_SCALE=3
PLATE_SCALE = 3
def main(usd_path):
    with rep.new_layer():

        scale = 0.005
        # env = rep.create.from_usd(usd_path, semantics=[("class", "world")])
        # plate = rep.create.from_usd(PLATE_USD_PATH, semantics=[("class", "plate")])
        env = rep.create.from_usd(usd_path)
        plate = rep.create.from_usd(PLATE_USD_PATH)

        # plate = rep.create.from_usd(PLATE_USD_PATH)
        # cube = rep.create.cube(position=rep.distribution.uniform((-0.5, -0.5, 1), (0.5, 0.5, 1)),
        #                         semantics=[('class', 'cube')],
        #                         scale=rep.distribution.uniform(0.001, 0.002), count=1)
        # sphere = rep.create.sphere(position=(0, 0, 0), semantics=[('class', 'sphere')], scale=scale)
        # cone = rep.create.cone(position=(-2, 0, 0), semantics=[('class', 'cone')], scale=scale)
        
        # plate_mesh = rep.get.prims(path_pattern="/Replicator/Ref_Xform_01/Ref/model/mesh")
        # plate_mesh = rep.get.prims(
        #                 path_pattern=".*model/mesh",
        #                 semantics=[("class", "plate")])
        with plate:
            rep.physics.collider()
            rep.physics.rigid_body(
                    velocity=0,
                    angular_velocity=0)
            rep.modify.pose(position=(0,0,1), scale=PLATE_SCALE)
            rep.physics.physics_material(
                static_friction=1.0,
                dynamic_friction=0.5,
                restitution=0.2
            )
            rep.physics.mass(
                mass=10,
            )


        camera = rep.create.camera(position=(5,2,1), look_at=plate)
        camera2 = rep.create.camera(position=(-5,2,1), look_at=plate)

        table = rep.get.prims(path_pattern=".*table_low_327/table_low")
        camera3 = rep.create.camera(position=(5,-2,1), look_at=plate)

        # with cube:
        #     rep.physics.collider()
        #     rep.physics.mass(mass=1)
        #     rep.physics.rigid_body(
        #             velocity=rep.distribution.uniform((-0,0,-0),(0,0,0)),
        #             angular_velocity=rep.distribution.uniform((-0,0,0),(0,0,0)))

        with table:
            rep.physics.collider()

        # with camera:
        #     # rep.modify.pose(look_at=(0,0,0), position=rep.distribution.sequence(camera1_pos))
        #     rep.modify.pose(position=rep.distribution.uniform((-5, 2, 1), (5, 5, 1)), look_at=table)
        # with camera2:
        #     rep.modify.pose(position=rep.distribution.uniform((-5, 2, -5), (5, 5, 5)), look_at=table)
        #     # rep.modify.pose(look_at=(0,0,0), position=rep.distribution.sequence(camera2_pos))
        # with camera3:
        #     rep.modify.pose(position=rep.distribution.uniform((-5, 2, 2), (5, 5, 2)), look_at=table)


        
        # with sphere:
        #     rep.physics.collider()
        
        # with cone:
        #     rep.physics.collider()
        
        rep.trigger.on_frame(num_frames=1000)
        # rep.trigger.on_time(interval=0.1, num=10)


        # Will render 512x512 images and 320x240 images
        render_product = rep.create.render_product(camera, (1024, 1024))
        render_product2 = rep.create.render_product(camera2, (1024, 1024))
        render_product3 = rep.create.render_product(camera2, (1024, 1024))

        basic_writer = rep.WriterRegistry.get("BasicWriter")
        basic_writer.initialize(
            output_dir=f"/home/smnair/work/nutrition/_output/rep_room",
            rgb=True,
            bounding_box_2d_loose=True,
            bounding_box_2d_tight=True,
            bounding_box_3d=True,
            distance_to_camera=True,
            distance_to_image_plane=True,
            instance_segmentation=True,
            normals=True,
            semantic_segmentation=True,
        )
        # Attach render_product to the writer
        basic_writer.attach([render_product, render_product2, render_product3])
        # basic_writer.attach([render_product, render_product2])
        # Run the simulation graph
        run_orchestrator()

    kit.update()
    kit.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A script to render a USD file with omniverse')
    parser.add_argument(
        '--usd-path', '-u', required=True, 
        default="/home/smnair/work/nutrition/data/simple_room/simple_room.usd",
        help='Path of the USD file you want to load')
    args = parser.parse_args()

    # with rep.new_layer():
    #     env = rep.create.from_usd(args.usd_path)
    #     # floor = rep.get.prims(path_pattern="/World/Floor")
    #     # table = rep.get.prims(path_pattern="/World/table_low_327/table_low")

    main(args.usd_path)
