#!/bin/bash

# This script adds all the python packages in the directories listed
#   below to a `pyrightconfig.json` file.
# Omniverse applications import each package defined in their *.kit`
#   configuration file. However, in order to get proper language server
#   functionality without having to add each package's path to the 
#   PYTHONPATH environment variable.
# This script generates the config automatically when this vscode 
#   devcontainer started. It should not be run outside of the
#   devcontainer since the packages only exist within the container.
#   Another potential solution is the ~/.pylintrc config.

config_file=/nvsynth/pyrightconfig.json
printf "{\n  \"extraPaths\": [\n" > $config_file.tmp
dirs=("/isaac-sim/exts" "/isaac-sim/kit/exts" "/isaac-sim/kit/extscore" "/isaac-sim/extscache" "/isaac-sim/extsPhysics")
for dir in "${dirs[@]}"; do
    find "$dir" -maxdepth 1 -type d -name 'omni*' -printf "    \"%p\",\n" >> $config_file.tmp
done
cat $config_file.tmp | head -c -2 > $config_file
printf "\n  ]\n}" >> $config_file
rm $config_file.tmp
