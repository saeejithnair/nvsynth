"""
This script converts the mesh files from OBJ to USD format, and stores the
generated meshes in the same folder as the original model, along with a new
`textures/` sub-folder containing the generated meshes.

Based on standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py

Usage (for CVIS data):
    ./isaac_sim/python.sh convert_obj2usd.py \
        --root_folder /pub2/nrc/aging/snair_cvis_data
"""
import argparse
import asyncio
import os

import omni
from omni.isaac.kit import SimulationApp


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    # converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
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


def asset_convert(args):
    supported_file_formats = ["stl", "obj", "fbx"]
    subfolders = [f.path for f in os.scandir(args.root_folder) if f.is_dir()]

    for folder in subfolders:
        print(f"\nConverting folder {folder}...")

        (result, models) = omni.client.list(folder)
        for i, entry in enumerate(models):
            model = str(entry.relative_path)
            model_name = os.path.splitext(model)[0]
            model_format = (os.path.splitext(model)[1])[1:]
            # Supported input file formats
            if model_format in supported_file_formats:
                input_model_path = folder + "/" + model
                converted_model_path = (
                    folder + "/" + model_name + "_" + model_format + ".usd"
                )
                if not os.path.exists(converted_model_path):
                    status = asyncio.get_event_loop().run_until_complete(
                        convert(input_model_path, converted_model_path, True)
                    )
                    if not status:
                        print(f"ERROR Status is {status}")
                    print(f"---Added {converted_model_path}")


if __name__ == "__main__":
    kit = SimulationApp()

    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument(
        "--root_folder", type=str, default=None, help="Root CVIS data folder."
    )
    parser.add_argument(
        "--load-materials",
        action="store_true",
        help="If specified, materials will be loaded from meshes",
    )
    args, unknown_args = parser.parse_known_args()

    if args.root_folder is None:
        raise ValueError("No folders specified via --folders argument")

    # Ensure Omniverse Kit is launched via SimulationApp before
    # asset_convert() is called.
    asset_convert(args)

    # cleanup
    kit.close()
