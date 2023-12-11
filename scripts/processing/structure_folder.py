import argparse
import pathlib
import re
import shutil

INCREMENT_NUMBER = 0 # CHANGE TO AVOID REWRITING OF PRIOR SCENES IF PROCESS ON DIFFERENT FOLDERS

IMAGE_2D_SRC_SUF = "rgb"
IMAGE_2D_DES_SUF = "2D_Image"
MASK_SRC_SUF = "semantic_segmentation"
MASK_DES_SUF = "masks"
META_SRC_SUF = "semantic_segmentation"
META_DES_SUF = "metadata"

MASK_FILE_SUFFIX = '.png'
METADATA_FILE_SUFFIX = '.json'

DATA_SRC_DEST_PAIR = [(IMAGE_2D_SRC_SUF, IMAGE_2D_DES_SUF), (MASK_SRC_SUF, MASK_DES_SUF), (META_SRC_SUF, META_DES_SUF)]

def copy_file(destination_parent_folder, file, scene_number, viewport_number):
    scene_destination = destination_parent_folder.joinpath(f"scene_{scene_number+INCREMENT_NUMBER}")
    if not scene_destination.exists():
        scene_destination.mkdir()
    shutil.copy(file, scene_destination.joinpath(f"{scene_number}_viewport_{viewport_number}{file.suffix}"))

# Initialize parser
parser = argparse.ArgumentParser()

# Adding arguments
parser.add_argument('parent_folder', type=pathlib.Path, help="Give the abs path to the datafolder")
parser.add_argument('-d', '--dest', type=pathlib.Path, help="The output folder")

args = parser.parse_args()

data_folder = args.parent_folder.resolve(strict=True)
output_folder = args.dest

if not data_folder.exists():
    raise ValueError(f"Non-existing data folder {data_folder}")

if not output_folder:
    output_folder = data_folder.parent

viewpoint_folders = [x for x in data_folder.iterdir() if x.is_dir()]

image_2d_des = output_folder.joinpath(IMAGE_2D_DES_SUF)
if image_2d_des.exists():
    shutil.rmtree(image_2d_des)
image_2d_des.mkdir()
mask_des = output_folder.joinpath(MASK_DES_SUF)
if mask_des.exists():
    shutil.rmtree(mask_des)
mask_des.mkdir()
metadata_des = output_folder.joinpath(META_DES_SUF)
if metadata_des.exists():
    shutil.rmtree(metadata_des)
metadata_des.mkdir()

for sub_folder in viewpoint_folders:
    # sub_folder = viewpoint_folders[0]
    folder_name = sub_folder.stem
    viewport_number = re.search(r"(\d+)$", folder_name)
    if viewport_number:
        viewport_number = viewport_number.group()
    else:
        viewport_number = 1

    # Match rgb
    image_2d_src = sub_folder.joinpath(IMAGE_2D_SRC_SUF).resolve(strict=True)
    for file in image_2d_src.iterdir():
        scene_number = re.search(r"(\d+)", file.stem).group()
        copy_file(destination_parent_folder=image_2d_des, file=file, scene_number=scene_number, viewport_number=viewport_number)

    # Match mask & metadata
    mask_src = sub_folder.joinpath(MASK_SRC_SUF).resolve(strict=True)
    for file in mask_src.iterdir():
        scene_number = re.search(r"(\d+)", file.stem).group()
        if file.suffix == MASK_FILE_SUFFIX:
            copy_file(destination_parent_folder=mask_des, file=file, scene_number=scene_number, viewport_number=viewport_number)
        elif file.suffix == METADATA_FILE_SUFFIX:
            copy_file(destination_parent_folder=metadata_des, file=file, scene_number=scene_number, viewport_number=viewport_number)
