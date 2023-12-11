# Creating Custom Food Scenes

## YAML Format

### Layers

To generate randomized scenes, we use a high-level scene description that's based on layers of 2D gaussian-distributed food items.
An example of a layer config is shown below:

```
layer1:
  name: "wing"
  usd_path: "/pub2/nrc/aging/CVIS_Data/id-21-chicken-wing-27g/textured_obj.usd"
  scale: [50,70]
  num: 12
  position: [[0,0,30], [5,2,0]]
  rotation: [[0,0,0], [360,360,360]]
  angular_velocity: [[0,0,0],[100,100,100]]
  static_friction: 0.9
  dynamic_friction: 0.9
  restitution: 0.05
  mass: 100
```

A layer has a name, a USD file path, and a list of attributes that are either scalar values, normal distirbutions or uniform distributions.
Normal distributions are defined as [mean, std] or [[mean_x, mean_y, mean_z], [std_x, std_y, std_z]].
Uniform distirbutions are defined as [min, max] or [[min_x, min_y, min_z], [max_x, max_y, max_z]].

### Scene Metadata

Besides the layers, each scene requires some metadata like the camera positions, and simulation time.
These are defined at the top level of the YAML. Here's an example:

```
scene:
  num_steps: 200
  camera:
    position: [0,200,100]
    horizontal_range:
      max: 360
      steps: 4
    vertical_range:
      max: 45
      steps: 3
  layer1:
  	...
```

First, the number of steps to run the simultation for is given by num_steps.
Second, the camera field requires a position, horizontal range and vertical range specification.
Position is the starting position of the camera in 3D coordinates.
Each of the ranges is defined by taking `steps` discrete steps along a circle about the origin for `max` degrees.
In this example, 4 steps along a horizontal circle along a 360 degree arc (the whole circle) means cameras will be place at:

`[0, 200, 100], [200, 0, 100], [0, -200, 100], [-200, 0, 100]`

Similarly, the vertical range spans the same sphere about the origin, covering the angles of 0, 22.5 and 45 degrees above the original camera position.
Combining these two ranges, we get a total of 12 camera positions on a spherical grid around the origin.

### Full Config

The full YAML config schema is shown below:

```
---
scene:
  num_steps: SCALAR
  camera:
    position: [SCALAR, SCALAR, SCALAR]
    horizontal_range:
      max: SCALAR[0,360]
      steps: SCALAR
    vertical_range:
      max: SCALAR[0,180]
      steps: SCALAR
  layer1:
    name: STRING
    usd_path: STRING
    scale: SCALAR or UNIFORM(1)
    num: SCALAR
    position: NORMAL(1) or NORMAL(3)
    rotation: UNIFORM(1) or UNIFORM(3)
    velocity: NORMAL(1) or NORMAL(3)
    angular_velocity: NORMAL(1) or NORMAL(3)
    static_friction: SCALAR
    dynamic_friction: SCALAR
    restitution: SCALAR
    mass: SCALAR
  layer2:
   	...
  ...

```

**Definitions:**

- SCALAR[a,b]: Float or integer in the range [a, b]
- NORMAL(d): Normal distribution of the form [mean, std] where each of mean, std is a length d list (or scalar if d=1)
- UNIFORM(d): Uniform distribution of the form [min, max] where each of min, max is a length d list (or scalar if d=1)

## Running a Scene

First, you're going to need to configure a conda environment (see vip-omni/README.md).
Once you have an environment, say `omni`, run the following to initialize it:

```
conda activate omni
source isaac_sim/setup_conda.sh

```

To run a scene, just make a config, say `configs/hotdog.yaml`, cd to into `vip-omni` and run:

`experimental/make_food_scene.py configs/hotdog.yaml`
