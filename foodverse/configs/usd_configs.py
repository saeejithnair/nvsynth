import copy
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, Optional

import git
import numpy as np

from foodverse.utils import dataset_utils as du


@dataclass
class ModelConfig:
    """Stores important properties for USD models."""

    label: str
    """Model name/ID (e.g. "id_9_nature_valley_granola_bar_13g", "plate")."""
    usd_path: str
    """Path to model USD file."""
    type: str
    """Model type (e.g. scene/food)."""
    scale: float = 1.0
    """Scale factor to apply to model."""
    uid: Optional[int] = None
    """Unique ID for food models (e.g. 9 for "id_9_nature_valley_granola_bar_13g")."""


@dataclass
class SemanticConfig:
    """Stores properties related to a semantic class."""

    class_name: str
    """Name of the semantic class (e.g. "nature-valley-granola-bar",
    "salad_chicken_strip", "plate, "scene", etc)."""
    semantic_id: int
    """Semantic ID of the class (e.g. 0, 1, 2, 3, etc.)."""
    rgba: Optional[np.ndarray] = None
    """RGBA color of the class (e.g. [183, 45, 111, 255])."""


def get_repo_working_dir(path_to_repo="."):
    """Returns path to the git repo root dir."""
    repo = git.Repo(path_to_repo, search_parent_directories=True)
    return repo.working_dir


# Dictionary storing the path of models, grouped based on model type.
MODELS_DIR = {
    "food": "/assets/food",
    "scene": "/nvsynth/assets/scene",
    "plate": "/nvsynth/assets/tableware",
    # "misc": "/nvsynth/assets/misc/ycb/Axis_Aligned_Physics"
}


def parse_csv(csv_path):
    """Parses a CSV file and returns a dictionary of {model_label: scale}.

    Reads in a CSV file with the following format:
        model_label,scale
        id_1_salad_chicken_strip_7g,11.56466937
        id_2_salad_chicken_strip_9g,10.65809374
        id_3_salad_chicken_strip_10g,11.93543211

    Returns a dictionary of {model_label: scale}.
    {
        "id_1_salad_chicken_strip_7g": 11.56466937,
        "id_2_salad_chicken_strip_9g": 10.65809374,
        "id_3_salad_chicken_strip_10g": 11.93543211
    }
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Remove header.
    lines = lines[1:]

    # Remove newlines.
    lines = [line.strip() for line in lines]

    # Split each line by comma.
    lines = [line.split(",") for line in lines]

    # Convert to dictionary.
    scale_factors = {line[0]: float(line[1]) for line in lines}

    return scale_factors


def register_models(models_dir_dict) -> Dict[str, ModelConfig]:
    """Scans assets directory to register all USD model files.

    Returns a dictionary of {model_label: ModelConfig}. Model label is generated
    based on the name of the directory the USD model was found in.
    """
    models = {}
    repo_working_dir = get_repo_working_dir()

    food_scale_factors = parse_csv(f"{repo_working_dir}/configs/scale_factors.csv")

    for model_type in models_dir_dict:
        # Model dir relative to repo.
        model_dir_abs = models_dir_dict[model_type]
        # Model dir absolute path.
        # model_dir_abs = f"{repo_working_dir}/{model_dir_rel}"

        # Get all subfolder paths in model dir.
        subfolders = [f.path for f in os.scandir(model_dir_abs) if f.is_dir()]

        for subfolder in subfolders:
            model_label = os.path.basename(subfolder)

            if "-" in model_label:
                # Replace all hyphens in string with underscores.
                model_label = model_label.replace("-", "_")

            # Find USD file within subfolder
            usds = glob(f"{subfolder}/*.usd")
            if len(usds) == 0:
                print(f"WARNING: No USD file found in {subfolder}")
                continue

            model_path = usds[0]
            if model_type == "food":
                model_uid = int(re.search(r"^id_(\d+)_*", model_label).group(1))
                scale_factor = food_scale_factors[model_label]
            else:
                model_uid = None
                scale_factor = 1.0

            models[model_label] = ModelConfig(
                label=model_label,
                usd_path=model_path,
                type=model_type,
                scale=scale_factor,
                uid=model_uid,
            )

    return models


def generate_unique_semantics(models: Dict[str, ModelConfig]) -> Dict[str, SemanticConfig]:
    """Generates IDs, class names, and colours for unique semantic types.
    Although the dataset has different labels for food items of the same type
    but different masses, we want to treat them as the same class. For example,
    we want to treat "id_1_salad_chicken_strip_7g" and
    "id_2_salad_chicken_strip_9g" as the same class ("salad_chicken_strip").

    Args:
        models: Dictionary of {model_label: ModelConfig}.

    Returns:
        Dictionary of {model_label: SemanticConfig}.
    """
    food_models = [model for model in models.values() if model.type == "food"]
    food_models.sort(key=lambda x: x.uid)
    food_models = [model.label for model in food_models]

    semantics = {}
    semantic_ids = {}

    # Add class for unlabelled pixels, but set it to ID 0.
    semantic_id_counter = 0
    label = "UNLABELLED"
    semantics[label] = SemanticConfig(class_name=label, semantic_id=semantic_id_counter)
    semantic_id_counter += 1

    # Add classes for plate model.
    plate_models = [model.label for model in models.values() if model.type == "plate"]
    assert len(plate_models) == 1, "Expected only one plate model."
    label = "plate"
    semantics[label] = SemanticConfig(class_name=label, semantic_id=semantic_id_counter)
    semantic_id_counter += 1

    for model_label in food_models:
        class_name = du.map_food_model_label_to_class_name(model_label)
        if class_name not in semantic_ids:
            semantic_ids[class_name] = semantic_id_counter
            semantic_id_counter += 1

        semantic_id = semantic_ids[class_name]
        semantics[model_label] = SemanticConfig(
            class_name=class_name, semantic_id=semantic_id
        )

    semantic_ids_list = list(
        set([sem_cfg.semantic_id for sem_cfg in semantics.values()])
    )
    if du.has_duplicates(semantic_ids_list):
        raise ValueError(f"Semantic IDs are not unique. {semantics}")

    # Sanity check that all semantic IDs are within expected range.
    if max(semantic_ids_list) != len(semantic_ids_list) - 1:
        raise ValueError(f"Semantic IDs are not consecutive. {semantics}")

    if min(semantic_ids_list) != 0:
        raise ValueError(f"Semantic IDs are not consecutive. {semantics}")

    # Generate unique colours for each semantic ID.
    unique_colors = du.generate_unique_colors(N=len(semantics))
    for model_label, sem_cfg in semantics.items():
        sem_cfg.rgba = np.array(unique_colors[sem_cfg.semantic_id], dtype=np.uint8)

    # Force the scene model to be part of the "UNLABELLED" class.
    semantics["scene"] = copy.deepcopy(semantics["UNLABELLED"])

    return semantics


MODELS: Dict[str, ModelConfig] = register_models(models_dir_dict=MODELS_DIR)
SEMANTIC_MAPPINGS: Dict[str, SemanticConfig] = generate_unique_semantics(models=MODELS)

# print(MODELS)
# print_mappings = {}
# for model_label, semantics in SEMANTIC_MAPPINGS.items():
#     output = f"{model_label}: ({semantics.semantic_id}, {semantics.class_name}, {semantics.rgba})"
#     if semantics.semantic_id in print_mappings:
#         print_mappings[semantics.semantic_id].append(output)
#     else:
#         print_mappings[semantics.semantic_id] = [output]

# for semantic_id, outputs in print_mappings.items():
#     print(outputs)
#     print("-----------------")
