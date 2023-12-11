import argparse
import asyncio

from omni.isaac.kit import SimulationApp


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file, out_file, progress_callback, converter_context
    )

    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def asset_convert(*args, **kwargs):
    asyncio.get_event_loop().run_until_complete(convert(*args, **kwargs))


if __name__ == "__main__":
    kit = SimulationApp()

    parser = argparse.ArgumentParser("Convert a single OBJ/STL asset to USD")
    parser.add_argument(
        "--in_file",
        help="Input OBJ/STL file. There must also be a texture.png"
        "file in the same directory!",
    )
    parser.add_argument("--out_file", help="Output USD file")
    parser.add_argument(
        "--load-materials",
        action="store_true",
        help="If specified, materials will be loaded from meshes",
    )
    args = parser.parse_args()

    asset_convert(args.in_file, args.out_file, args.load_materials)

    # cleanup
    kit.close()
