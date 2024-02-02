#!/bin/bash

# This script adds all the python packages in the directories listed below
#   directly to the PYTHONPATH environment variable.
# Omniverse applications will add the packages they *need* automatically,
#   but this script will enable you to load everything -- without having
#   to be inside this omniverse application environment.
# Note: most omni.* python modules will require you to be in the app env
#   for you to run them anyways. Thus, this script shouldn't really be used
#   outside of testing; it may break things. Beware.
# This script was mostly a bi-product of getting pyright to work in a vscode
#   devcontainer. See [this file](.devcontainer/gen_pyrightconfig.sh).
#
# Usage: 
#   - within our docker container run:
#       `source /nvsynth/docker/omni_pythonpath.sh`
#   - you should now be able to import *some* more packages in python:
#       `python -c "import pxr"`

dirs=("/isaac-sim/exts" "/isaac-sim/kit/exts" "/isaac-sim/kit/extscore" "/isaac-sim/extscache" "/isaac-sim/extsPhysics")
result=""
for dir in "${dirs[@]}"; do
    result+=$(find "$dir" -maxdepth 1 -type d -name 'omni*' -printf "%p:")
done
export PYTHONPATH=$PYTHONPATH:${result%:}
