import importlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Type

import tyro
from foodverse.configs.base_configs import InstantiateConfig
from foodverse.configs.scene_configs import FoodverseSceneConfig
from omni.isaac.kit import SimulationApp
from typing_extensions import Literal


@dataclass
class SimConfig(InstantiateConfig):
    """Config for instantiating an OV SimulationApp."""

    _target: Type = field(default_factory=lambda: SimulationApp)
    """The target class to instantiate."""

    renderer: tyro.conf.Fixed[
        Literal["PathTracing", "RayTracedLighting"]
    ] = "PathTracing"
    """Strongly recommend sticking with PathTracing, RayTracedLighting
    gives weird artifacts."""
    samples_per_pixel_per_frame: int = 12
    """The number of samples to render per frame, increase for improved quality.
    Used for `PathTracing` only."""
    headless: bool = True
    """Whether to run the simulation in headless mode."""
    multi_gpu: tyro.conf.Fixed[bool] = False
    """Whether to use multiple GPUs for rendering."""

    def setup(self, **kwargs: Any) -> SimulationApp:
        return self._target(launch_config=asdict(self))


@dataclass
class FoodverseSceneBuilderConfig(InstantiateConfig):
    """Config for building an OV SimulationApp."""

    _target: Type = field(default_factory=lambda: FoodverseSceneBuilder)
    """The target class to instantiate."""

    app_module: tyro.conf.Fixed[str] = "foodverse.foodverse_scene"
    """The module containing the user app to instantiate."""
    app_class: tyro.conf.Fixed[str] = "FoodverseScene"
    """The class name of the user app to instantiate."""
    scene: FoodverseSceneConfig = field(default_factory=FoodverseSceneConfig)
    """The config for instantiating the user app."""
    sim: SimConfig = field(default_factory=SimConfig)
    """The config for instantiating the OV SimulationApp."""
    display_gpu_id: int = 0
    """The GPU ID to use for rendering."""


class FoodverseSceneBuilder:
    """Builds an OV SimulationApp."""

    def __init__(self, config: FoodverseSceneBuilderConfig):
        self.config = config
        if config.sim.multi_gpu:
            raise ValueError("Multi-GPU rendering not supported.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.display_gpu_id)

        self.kit: SimulationApp = config.sim.setup()

    def build(self) -> Any:
        """Builds and returns the user app."""
        module = importlib.import_module(self.config.app_module)
        app_class = getattr(module, self.config.app_class)

        return app_class(self.kit, self.config.scene)
