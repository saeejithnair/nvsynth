# nvsynth
NutritionVerse-Synth: Synthetic generation of food scenes for dietary intake estimation

## Set Up for Docker (Streaming)

### On Remote Machine (Streaming Server)

#### Pre-Requisites

- Docker Installation (check with: `docker run hello-world`)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker). (check: `nvidia-container-cli -V`)
- A [NGC Account](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account) and an [NGC API Key](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key)
    - See this [this page](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html)
    - Make an account and configure/verify your docker credentials locally with: (`docker login nvcr.io`)

#### Set Up

- Clone this repository:
```bash
git clone git@github.com:saeejithnair/nvsynth.git
cd nvsynth
```

- Setup your environment variables:
```bash
./initialize.sh
```

- Build the Isaac Sim docker container:
```bash
docker compose build
```

### On Your Local Machine (Streaming Client)

  - A chromium-based browser (Google Chrome, Chromium, Microsoft Edge)

## Usage for Docker (Streaming)

### On Remote Machine (Streaming Server)

- There are currently two usages of this docker container:
  - (1) the first is simply running the base `isaac-sim` app
  - (2) the second is running our custom python scripts in this repo
  - (3) the third (default) is starting the docker container and running scripts/the app manually


#### (1) Running the Base Isaac Sim App

- For the basic usage of the app, supply the following command to the service in the `docker-compose.yml`
```yaml
services:
  isaac-sim:
    command: ["/isaac-sim/runheadless.webrtc.sh"]
```

- You can also add arguments by appending them to the list:
```yaml
services:
  isaac-sim:
    command: [
      "/isaac-sim/runheadless.webrtc.sh",
      "--/app/livestream/logLevel=debug", # set livestream loglevels
      "--/app/window/dpiScaleOverride=1.5", # rescale livestream window UI
      "--/exts/omni.services.transport.server.http/port=8045", # change WebRTC server port
      "--/app/livestream/port=48010", # change livestream data port
    ]
```

- Then we can start our docker container with
```bash
docker compose up
```

#### (2) Running Isaac Sim via a Python Script

- To run our scripts, we also modify the `docker-compose.yml` file by changing the target of our `Dockerfile` to `nvsynth` and add the command `"python path/to/script.py"`:
```yaml
services:
  isaac-sim:
    command: ["python", "scripts/gen_random_scenes.py"] # path of script is relative to repo root
```

- Then we can start our docker container with
```bash
docker compose up
```

#### (3) Running Isaac Sim Manually

- Start the docker container:
```bash
docker compose up -d
```

- Enter the container
```bash
docker compose exec isaac-sim bash
```

- Run the a command to start the Isaac-Sim app
```bash
/isaac-sim/runheadless.webrtc.sh
# OR
python scripts/gen_random_scenes.py
```

### On Your Local Machine (Streaming Client)

- Now that your the Omniverse Streamer is running on the remote machine, we can view the stream on our local machine

- Open a local terminal and forward the stream URLs on the remote machine to your machine:
```bash
ssh -NL 8211:localhost:8211 -L 49100:localhost:49100 <ip of remote machine>
```

- Go to a browser and open: http://127.0.0.1:8211/streaming/webrtc-demo/?server=127.0.0.1
  - it may have to be chromium-based, firefox did not work for me

### Debugging

- On the remote machine you can check that the correct ports are being used with: 
```bash
docker compose exec isaac-sim bash -c "lsof -nP -iTCP -sTCP:LISTEN"
# You should see an output similar to this:
COMMAND PID USER   FD   TYPE   DEVICE SIZE/OFF NODE NAME
kit      21 root  258u  IPv4 43127461      0t0  TCP *:48010 (LISTEN)
kit      21 root  277u  IPv4 43151425      0t0  TCP *:8211 (LISTEN)
```

- On your client machine try and ping the ports, that you have forwarded on your machine
  - you should get `404 Not Found` or `501 Not Implemented` errors, but you should not be getting `failed: Connection refused.`
```bash
wget localhost:8211 # 404 Not Found
wget localhost:49100 # 501 Not Implemented
```

- You can also monitor the logs in the container with:
```bash
# filename is in format `kit_YYYYMMDD_HHMMSS.log` use tab to complete
tail -f ~/docker/isaac-sim/logs/Kit/Isaac-Sim/2023.1/kit_
```

- If you get any permission errors when running isaac-sim, check if there are files/folders not owned by you with:
```bash
find ~/docker/isaac-sim/ ! -user $(whoami) -print
# If files are printed out, try running: chown -R $(id -u):$(id -g) ~/docker
```

## Set Up for Lavazza

### Install Omniverse
Download Omniverse Installer from https://www.nvidia.com/en-us/omniverse/download/

The easiest way is to download it to your local machine and then `scp` the AppImage onto the desired server.
You can also find an existing copy of the AppImage on `guacamole:/pub2/nrc/omniverse/omniverse-launcher-linux.AppImage`

To install omniverse, simply run the AppImage and follow the installer prompts. 
`./omniverse-launcher-linux.AppImage`

Make sure to install IsaacSim, Omniverse Code, and optionally Omniverse Create. You can install all of these packages through the Nucleus Launcher.

Omniverse should be installed to `/home/$USER/.local/share/ov/`.

The packages should be installed to `/home/$USER/.local/share/ov/pkg`.


### Installation Errors
#### FUSE error upon running the AppImage
`fuse: failed to exec fusermount: No such file or directory`

The solution is to follow the instructions listed on https://github.com/AppImage/AppImageKit/wiki/FUSE#install-fuse

Since `guacamole` runs Ubuntu 22.04, the following commands were executed

* `sudo add-apt-repository universe`
* `sudo apt install libfuse2`
* [**CAUTION**: Only try this if the error hasn't been resolved] `sudo apt-get install fuse3`


### Clone repo
`cd $WORKSPACE`

`git clone git@github.com:saeejithnair/vip-omni.git`

### Create symbolic link to Isaac Sim
`cd $WORKSPACE/vip-omni`

`ln -s /home/$USER/.local/share/ov/pkg/isaac_sim-2022.1.1 isaac_sim`

### Set up conda environment
`cd $WORKSPACE/vip-omni/isaac_sim`

`conda env create --name "${USER}_isaac" --file=environment.yml`

`conda activate "${USER}_isaac"`

`source setup_conda_env.sh`

`export DISPLAY_GPU_ID=2`

`export CUDA_VISIBLE_DEVICES=$DISPLAY_GPU_ID`

### Install Foodverse packages
`cd $WORKSPACE/vip-omni/`

`pip install -e .`

`pip install GitPython`

### Create symbolic link to food assets directory
`cd $WORKSPACE/vip-omni/assets`

`ln -s /pub2/nrc/aging/snair_iccv_data food`

# Data Preprocessing
Once the 2D images are generated, the following scripts need to be run to structure the data and create the desired metadata files:

1. `structure_folder.py`: This script takes the 2D images and structures them into an organized file format for later processing.

```
project
└───2D_Image
│   └───scene_0000
│       │   0000_viewport_1.png
│       │   0000_viewport_2.png
│       │   ...
│       │   0000_viewport_12.png
│   └───scene_0001
│       │   0001_viewport_1.png
│       │   0001_viewport_2.png
│       │   ...
│       │   0001_viewport_12.png
│   └───...
│       │   ...
│   └───scene_50000
│       │   0001_viewport_1.png
│       │   0001_viewport_2.png
│       │   ...
│       │   0001_viewport_12.png
│   
└───masks
│   └───scene_0000
│       │   0000_viewport_1.png
│       │   0000_viewport_2.png
│       │   ...
│       │   0000_viewport_12.png
│   └───scene_0001
│       │   0001_viewport_1.png
│       │   0001_viewport_2.png
│       │   ...
│       │   0001_viewport_12.png
│   └───...
│       │   ...
│   └───scene_50000
│       │   0001_viewport_1.png
│       │   0001_viewport_2.png
│       │   ...
│       │   0001_viewport_12.png
│   
└───metadata
│   └───scene_0000
│       │   0000_viewport_1.json
│       │   0000_viewport_2.json
│       │   ...
│       │   0000_viewport_12.json
│   └───scene_0001
│       │   0001_viewport_1.json
│       │   0001_viewport_2.json
│       │   ...
│       │   0001_viewport_12.json
│   └───...
│       │   ...
│   └───scene_50000
│       │   0001_viewport_1.json
│       │   0001_viewport_2.json
│       │   ...
│       │   0001_viewport_12.json
```

2. `generate_metadata_file.py`: This script uses the metadata folder generated by the previous script to create a dish metadata csv file containing the following fields:

`dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, num_ingrs, (ingr_1_id, ingr_1_name, ingr_1_grams, ingr_1_calories, ingr_1_fat, ingr_1_carb, ingr_1_protein, ...)`

This dish metadata csv file is the same format as that of the Nutrition5k paper with the last 8 fields repeated for every ingredient present in the dish. 
