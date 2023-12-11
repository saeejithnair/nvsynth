from typing import TYPE_CHECKING

import tyro

from foodverse.app_builder import FoodverseSceneBuilder, FoodverseSceneBuilderConfig

if TYPE_CHECKING:
    # Cannot import directly since SimulationApp must be loaded first, but we
    # expose it here for type checking.
    from foodverse.foodverse_scene import FoodverseScene


def main(
    config: FoodverseSceneBuilderConfig,
):
    scene_builder: FoodverseSceneBuilder = config.setup()
    fv_scene: FoodverseScene = scene_builder.build()

    fv_scene.generate_persistent_food_items(capture_falling_every_n_steps=1)

    fv_scene.cleanup()


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(
        tyro.conf.SuppressFixed[  # Don't show fixed arguments in helptext.
            tyro.conf.FlagConversionOff[  # Require explicit True/False for booleans.
                FoodverseSceneBuilderConfig
            ]
        ]
    )
    config.scene.root_output_dir = (
        "/home/smnair/work/nutrition/vip-omni/_fv_test_output"
    )
    main(config)
