"""
This file contains utility functions for file operations.
"""

import os
import shutil
import json
import io
from typing import Optional, Dict

def create_new_folder(folder_path, folder_name) -> str:
    """Creates a new folder if it doesn't already exist.
    """

    folder_path = os.path.join(folder_path, folder_name)
    
    # If folder exists, remove existing folder.
    remove_folder_if_exists(folder_path)
    
    # Create new folder.
    os.makedirs(folder_path)
    return folder_path

def remove_folder_if_exists(folder_path) -> None:
    """Removes the folder at the given path if it exists.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def compose_file_basename(output_type: str, scene_name: str, filename: str,
                          basename_prefix: Optional[str] = None) -> str:
    """Composes the file basename from the given output type, scene name, and
    filename. If a basename prefix is given, it is prepended to the basename.
    """
    file_basename = f"{output_type}/{scene_name}/{filename}"
    if basename_prefix:
        file_basename = f"{basename_prefix}_{file_basename}"

    return file_basename

def compose_file_path(file_basename: str, ext: str) -> str:
    """Composes the file path from the given basename and extension.
    """
    return f"{file_basename}.{ext}"

def write_dict_to_json_backend(output_dict: Dict,
                               backend: "omni.replicator.core.BackendDispatch",
                               file_basename: str) -> None:
    """Writes the given dict to a JSON file using the given backend.

    Args:
        output_dict: The dict to write to JSON.
        backend: The backend to use for writing the JSON file.
        file_basename: Relative file path minus extension.
            Note that absolute path is determined by the backend.

    Returns:
        None
    """
    ext = "json"
    file_path = f"{file_basename}.{ext}"

    buf = io.BytesIO()
    buf.write(json.dumps({
        str(k): v for k, v in output_dict.items()}).encode())
    backend.write_blob(file_path, buf.getvalue())


def write_json_to_file(path, data, scene_idx, frame_padding=4) -> None:
    """Serializes the data to JSON and writes it to a file.
    
    Args:
        path: The path to the file.
        data: The data to serialize.

    Returns:
        None
    """
    rp_folder_prefix = "RenderProduct_Viewport"
    annotator = "amodal"

    for rp_idx in data:
        if rp_idx > 0:
            rp_folder_name = f"{rp_folder_prefix}_{rp_idx}"
        else:
            rp_folder_name = rp_folder_prefix
        rp_output_path = os.path.join(path, rp_folder_name, annotator)
        
        # Create folder if it doesn't exist.
        if not os.path.exists(rp_output_path):
            os.makedirs(rp_output_path)
        
        filename = f"{annotator}_{scene_idx:0{frame_padding}}.json"
        output_path = os.path.join(rp_output_path, filename)
        
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile)
