"""
This file contains utility functions for dataset preparation.
"""

import colorsys
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def compose_prim_name(model_label: str, id: Optional[int] = None) -> str:
    """Composes a unique name for a USD prim to allow differentiation between
    multiple prims of the same class.

    Args:
        model_label: The label of the model
            (e.g. "plate" or "id_94_chicken_wing_27g").
        id: The id of the primitive.

    Returns:
        The name of the primitive (label concatenated with id if provided)
            (e.g. "plate" or "id_94_chicken_wing_27g_1").
    """
    if id is None:
        # A unique id must be provided for all food models.
        # We assume that all food models begin with "id_".
        if model_label.startswith("id_"):
            raise ValueError("ID cannot be None for food models.")
        return model_label
    else:
        return f"{model_label}_{id}"


def compose_prim_path(scope_name: str, prim_name: str) -> str:
    """Composes a path to place a USD prim at.

    Args:
        scope_name: The name of the scope to place the prim in.
            (e.g. "/MyScope").
        prim_name: The name of the prim.
            (e.g. "plate" or "id_94_chicken_wing_27g_1")

    Returns:
        The path to place the prim at.
            (e.g. "/MyScope/id_94_chicken_wing_27g_1")
    """
    return f"{scope_name}/{prim_name}"


def decompose_prim_path(prim_path: str) -> Tuple[str, str]:
    """Decomposes a prim path into the scope name and prime name.

    Args:
        prim_path: The path to the prim.
            (e.g. "/MyScope/id_94_chicken_wing_27g_1")

    Returns:
        Tuple of (scope_name, prim_name).
            E.g. ("/MyScope", "id_94_chicken_wing_27g_1")
    """
    parts = prim_path.split("/")
    return "/".join(parts[:-1]), parts[-1]


def decompose_prim_name(prim_name: str) -> Tuple[str, Optional[int]]:
    """Decomposes a prim name into its constituent parts.

    Args:
        prim_name: The name of the prim.
            (e.g. "plate" or "id_94_chicken_wing_27g_1")

    Returns:
        The label of the model
            (e.g. "plate" or "id_94_chicken_wing_27g").
        The id of the primitive (if exists).
    """
    parts = prim_name.split("_")
    if len(parts) == 1:
        return parts[0], None
    else:
        return "_".join(parts[:-1]), int(parts[-1])


def get_idx_from_annotator_name(annotator_name: str) -> int:
    """Gets the index of the render product from the annotator name.

    Args:
        annotator_name: The name of the annotator.
            (e.g. "rgb-RenderProduct_Viewport"
            or "instance_segmentation-RenderProduct_Viewport_2")

    Returns:
        The index of the render product.
    """
    annotator_split = annotator_name.split("-")

    if len(annotator_split) > 1:
        # Indicates that there are multiple render products.
        # Otherwise, the annotator name is just the annotator name.
        # Get render product name (e.g. "RenderProduct_Viewport_2").
        render_product_name = annotator_split[-1]

        # Split render product name by underscore.
        # E.g. ["RenderProduct", "Viewport", "2"]
        render_product_split = render_product_name.split("_")

        if len(render_product_split) > 2:
            # There is a viewport index specified.
            render_product_idx = int(render_product_split[-1])
        else:
            # Render product index is ommitted for viewport 1.
            # Note that render product index is 1-indexed.
            render_product_idx = 1
    else:
        # There is only one render product.
        render_product_idx = 1

    return render_product_idx


def compose_viewport_name(render_product_idx: int) -> str:
    """Composes a viewport name from a render product index.

    Args:
        render_product_idx: The index of the render product.

    Returns:
        The name of the viewport.
            E.g. "viewport_2"
    """
    return f"viewport_{render_product_idx}"


def get_prim_path_from_instance_label(instance_label: str) -> str:
    """Gets the prim prim from its instance label.

    Args:
        instance_label: The instance label of the prim. (e.g.
            '/MyScope/id_42_costco_california_sushi_roll_2_27g_4/poly/mesh',
            '/Replicator/Ref_Xform/Ref/table_low_327/table_low',
            '/MyScope/plate/model/mesh',
            '/MyScope/id_70_chicken_leg_31g_3/poly/mesh')

    Returns:
        Path to the corresponding prim (e.g.
            '/MyScope/id_42_costco_california_sushi_roll_2_27g_4',
            '/Replicator/Ref_Xform/Ref',
            '/MyScope/plate',
            '/MyScope/id_70_chicken_leg_31g_3')
    """
    return "/".join(instance_label.split("/")[:-2])


def map_food_model_label_to_class_name(model_label: str) -> str:
    """Maps a food model label to its corresponding class name.

    Args:
        model_label: The label of the model
            (e.g. "id_94_chicken_wing_27g").

    Returns:
        The class name of the model (e.g. "chicken_wing").
    """
    if not model_label.startswith("id_"):
        raise ValueError("Food model label must start with 'id_'")

    pattern = r"id_\d+_(.+)_\d+g"
    match = re.search(pattern, model_label)

    if not match:
        raise ValueError(
            f"Food model label ({model_label}) did not match expected pattern."
        )

    class_name = match.group(1)

    # Handle edge cases where there's an additional underscore in the classname
    # (e.g. "id_37_costco_cucumber_sushi_roll_1_16g" -> "costco_cucumber_sushi_roll_1"
    # but we want "costco_cucumber_sushi_roll")
    class_name_split = class_name.split("_")
    if class_name_split[-1].isdigit():
        class_name = "_".join(class_name_split[:-1])

    return class_name


def has_duplicates(input: list) -> bool:
    """Checks if a list contains duplicates.

    Args:
        input: The list to check.

    Returns:
        True if the list contains duplicates, False otherwise.
    """
    return len(input) != len(set(input))


def generate_unique_colors(N: int) -> List[List[int]]:
    """
    Generates N unique RGB colors.

    Args:
        N: The number of unique colors to generate.

    Returns:
        A list of N unique RGBA colors.
    """
    # Distribute colors evenly on HSV color wheel with max saturation
    # and V alternating between 3 different levels so that not all
    # colors have the same brightness.
    hsv_colors = [(i / N, 1, (i % 3 + 1) / 3) for i in range(N)]

    # Shuffle colors to avoid having similar colors next to each other.
    np.random.seed(27)
    random.seed(27)
    np.random.shuffle(hsv_colors)
    rgba_colors = [
        list(map(int, np.array(colorsys.hsv_to_rgb(*hsv)) * 255)) for hsv in hsv_colors
    ]

    if has_duplicates([tuple(color) for color in rgba_colors]):
        raise ValueError("Error, duplicate colors generated.")

    return rgba_colors


def id_to_rgba(
    semantic_array: np.ndarray,
    semantic_id_to_rgba: Dict[int, List[int]],
    num_channels=3,
) -> np.ndarray:
    """
    Converts an array of semantic IDs to an array of RGBA values.

    Args:
        semantic_array: The array of semantic IDs.
        semantic_id_to_rgba: A dictionary mapping semantic IDs to list
            of RGBA values.

    Returns:
        The array of RGBA values remapped based on the semantic IDs.
    """
    # Create an array to store the RGBA values
    rgba_shape = (*semantic_array.shape, num_channels)
    rgba_array = np.zeros(rgba_shape, dtype=np.uint8)

    unique_ids = np.unique(semantic_array)
    for id in unique_ids:
        rgba = semantic_id_to_rgba[id]
        rgba_array[semantic_array == id] = rgba

    return rgba_array


def remap_ids(
    semantic_array: np.ndarray, semantic_id_to_true_id: Dict[int, int]
) -> np.ndarray:
    """Remaps semantic IDs in an array.

    Args:
        semantic_array: The array of semantic IDs.
        semantic_id_to_true_id: A dictionary mapping semantic IDs to
            their corresponding true IDs.

    Returns:
        The array of semantic IDs remapped based on the true IDs.
    """
    unique_ids = np.unique(semantic_array)
    for id in unique_ids:
        true_id = semantic_id_to_true_id[id]
        semantic_array[semantic_array == id] = true_id

    return semantic_array


def colorize_normals(data: np.ndarray) -> np.ndarray:
    """Convert normals data into colored image.

    Args:
        data: data returned by the annotator.

    Return:
        Data converted to uint8 RGB image.
    """

    colored_data = ((data * 0.5 + 0.5) * 255).astype(np.uint8)
    return colored_data
