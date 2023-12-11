import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import tyro


@dataclass
class Position:
    def as_cartesian(self) -> "CartesianPosition":
        """Convert to Cartesian coordinates."""
        if isinstance(self, CartesianPosition):
            return self
        else:
            raise NotImplementedError(
                f"Subclass {type(self)} must implement this method.")

@dataclass
class CartesianPosition(Position):
    x: float
    """X coordinate of object."""
    y: float
    """Y coordinate of object."""
    z: Optional[float] = None
    """Height of object above reference plane."""

    def as_list(self) -> List[float]:
        """Convert to list of Cartesian coordinates [x, y, z]."""
        return [self.x, self.y, self.z] if self.z is not None else [self.x, self.y]

@dataclass
class CylindricalPosition(Position):
    r: float
    """Radius of object from origin."""
    theta: float
    """Angle of object from origin in degrees."""
    z: Optional[float] = None
    """Height of object above reference plane."""

    def as_cartesian(self) -> CartesianPosition:
        """Convert to Cartesian coordinates."""
        x = self.r * math.cos(math.radians(self.theta))
        y = self.r * math.sin(math.radians(self.theta))
        return CartesianPosition(x, y, self.z)

@dataclass
class EulerOrientation:
    roll: float = 0.0
    """Roll of object in degrees."""
    pitch: float = 0.0
    """Pitch of object in degrees."""
    yaw: float = 0.0
    """Yaw of object in degrees."""

    def as_list(self, radians=False) -> List[float]:
        """Convert to list of Euler angles [roll, pitch, yaw]."""
        orientation = [self.roll, self.pitch, self.yaw]
        if radians:
            orientation = [math.radians(angle) for angle in orientation]

        return orientation

@dataclass
class PoseConfig:
    position: Union[CartesianPosition, CylindricalPosition]
    """Position of object (x, y, z)."""
    orientation: Optional[EulerOrientation] = None
    """Orientation of object in euler angles."""

@dataclass
class FoodItemConfig:
    model_label: str
    """Label of model to use for food item (e.g. 'id_94_chicken_wing_27g')."""
    pose: PoseConfig
    """Pose of food item."""
    scale: Optional[float] = None
    """Scale of food item."""
    static: bool = False
    """Whether food item is static (i.e. not movable)."""

@dataclass
class PlacementRule:
    """Base class for placement rules."""
    def generate_food_items(self) -> List[FoodItemConfig]:
        raise NotImplementedError

@dataclass
class CircularPlacement(PlacementRule):
    model_label: str
    num_items: int
    radius: float
    start_angle: float
    increment_angle: float
    height: float
    orientation: Optional[EulerOrientation] = None
    description: Optional[str] = ""

    def generate_food_items(self) -> List[FoodItemConfig]:
        food_items = []
        angle = self.start_angle
        for _ in range(self.num_items):
            food_item = FoodItemConfig(
                model_label=self.model_label,
                pose=PoseConfig(
                    position=CylindricalPosition(
                        r=self.radius,
                        theta=angle,
                        z=self.height),
                    orientation=self.orientation
                ),
                static=True,
            )
            food_items.append(food_item)
            angle += self.increment_angle
        return food_items

@dataclass
class SceneItems:
    food_items: List[FoodItemConfig] = field(default_factory=list)
    """List of food items."""
    procedural_items: List[PlacementRule] = field(default_factory=list)
    """List of items to procedurally generate."""

    @staticmethod
    def from_yaml(yaml_file: Union[Path, str]) -> "SceneItems":
        """Load scene items from yaml file."""
        if isinstance(yaml_file, str):
            yaml_file = Path(yaml_file)
        return tyro.extras.from_yaml(SceneItems, yaml_file.read_text())

    def to_yaml(self, yaml_file: Union[str,Path]) -> None:
        """Save scene items to yaml file."""
        if isinstance(yaml_file, str):
            yaml_file = Path(yaml_file)
        # Serialize self to yaml string.
        yaml_str = tyro.extras.to_yaml(self)

        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text(yaml_str)
@dataclass
class FoodverseSceneConfig:
    """Config for instantiating a FoodverseScene."""

    seed: Optional[int] = None
    """Seed for random number generator. If None, sim start time is used."""
    root_output_dir: str = "/pub3/smnair/foodverse/output"
    """Root directory to save generated scenes."""
    plate_scale: float = 1.0
    """Scale of plate."""
    scene_start_idx: int = 0
    """Scene index to start generating from."""
    num_cameras: int = 12
    """Number of cameras to use for capturing the scene."""
    num_scenes: int = 6000
    """Number of scenes to generate."""
    scene_items: Optional[Union[Path, SceneItems]] = None
    """Scene items or path to yaml config file."""
