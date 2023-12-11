from dataclasses import dataclass

@dataclass
class MaterialProperties:
    static_friction: float
    dynamic_friction: float
    restitution: float


# Material properties for the plate.
PLATE_MATERIAL = MaterialProperties(
    static_friction=0.7,
    dynamic_friction=0.5,
    restitution=0.2,
)

# Material properties for the food.
FOOD_MATERIAL = MaterialProperties(
    static_friction=0.9,
    dynamic_friction=0.9,
    restitution=0.001,
)

SIM_APP_CFG = {
    "renderer": "PathTracing", # Can also be "RayTracedLightning"
    # The number of samples to render per frame, increase for improved
    # quality, used for `PathTracing` only.
    "samples_per_pixel_per_frame": 12,
    "headless": True,
    "multi_gpu": False,
}

CAMERA_WIDTH = 1200
CAMERA_HEIGHT = 1200
CAMERA_FOCAL_LENGTH = 150

PLATE_SCALE = 1.0

# The number of seconds to wait for an object to settle before
# we decide to remove it, and move on to the next item.
SIM_ITEM_TIMEOUT_PERIOD_SECS = 30
SIM_ITEM_SETTLE_VELOCITY = 0.001

# Maximum number of items to place in a scene. This is a rough
# heuristic to avoid scenes with too many items, and not a technical limit.
MAX_ITEMS_PER_SCENE = 7

# Scope name to encompass all Foodverse related prims.
SCOPE_NAME = "/MyScope"

DEFAULT_NUM_SCENES = 6000
DEFAULT_OUTPUT_DIR = "/pub3/smnair/foodverse/output"
DEFAULT_NUM_CAMERAS = 12

# Number of threads to utilize for writing data
WRITE_THREADS = 32
