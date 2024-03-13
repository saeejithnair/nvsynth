import os
import yaml
import numpy as np
os.system('rm _outputs/*.png')


from omni.isaac.kit import SimulationApp

kit = SimulationApp({'headless': False, 'renderer': "RayTracedLighting", 'num_frames': 100})

food_path = "/assets/food/{}/poly_obj.usd"


def get_objects(yaml_path):
	with open(yaml_path, 'r') as f:
		yaml_dict = yaml.safe_load(f)

	items = yaml_dict['scene']['items']


	objects = []
	for item in items:
		(name, props), = item.items()
		fnames = props['names']
		positions = props['positions']

		x, y, z, _ = np.broadcast_arrays(*positions.values(), np.empty((1,2))) # N x 2
		pos = np.stack([x,y,z], axis=1).reshape((-1,3,2))                     # N x 3 x 2
		low, high = pos.transpose(2,0,1)				                      # 2 x N x 3
		pos_rnd = np.random.uniform(low, high) 			                      # N x 3

		N = pos_rnd.shape[0]
		fname_inds = np.random.choice(len(fnames), N)
		fnames_rnd = [food_path.format(fnames[i]) for i in fname_inds]

		objects.extend(zip(fnames_rnd, pos_rnd))
	
	return objects


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


usd_path = "/nvsynth/assets/scene/simple_room/simple_room.usd"
plate_path = "/nvsynth/assets/tableware/plate/plate.usd"


import omni.replicator.core as rep
with rep.new_layer():
    for fpath, pos in get_objects('configs/sushi.yaml'):
        with rep.create.from_usd(fpath):
            rep.modify.pose(position=list(pos), scale=0.15)

    room = rep.create.from_usd(usd_path)
    plate = rep.create.from_usd(plate_path)

    camera = rep.create.camera(focal_length=96)
    render_product = rep.create.render_product(camera, (1024, 1024))

    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(10000, 5000),
            position=rep.distribution.uniform((-3, -3, 12), (3, 3, 15)),
            scale=rep.distribution.uniform(0.5, 1),
            count=num
        )
        return lights.node
    rep.randomizer.register(sphere_lights)


    with rep.trigger.on_time(interval=1000,num=1):
        # rep.randomizer.get_plate()
        rep.randomizer.sphere_lights(10)

        with camera:
            rep.modify.pose(position=(2, 0, 2), look_at=plate)
            # rep.modify.pose(position=(16, 0, 12), rotation=(0, -15, 0))

    rep.trigger.on_frame(num_frames=100)


    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="/nvsynth/_output/", rgb=True)
    writer.attach([render_product])

    run_orchestrator()
kit.update()
kit.close()
    


