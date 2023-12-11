import argparse

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.kit import SimulationApp

kit = SimulationApp({"headless": False})


def get_stage(kit):
    return kit.context.get_stage()


def get_prim_from_stage(stage, prim_id):
    return stage.GetPrimAtPath(prim_id)


def render(usd_path):
    simulation_context = SimulationContext()
    add_reference_to_stage(usd_path, "/World")
    # need to initialize physics getting any articulation..etc
    simulation_context.initialize_physics()
    simulation_context.play()

    for i in range(10000):
        simulation_context.step(render=True)
        print(i)

    simulation_context.stop()
    kit.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script to render a USD file with omniverse")
    parser.add_argument(
        "--usd-path", "-u", required=True, help="Path of the USD file you want to load"
    )
    args = parser.parse_args()
    render(args.usd_path)
