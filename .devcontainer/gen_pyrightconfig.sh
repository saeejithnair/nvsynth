#!/bin/bash
# other potential options are ~/.pylintrc or PYTHONPATH
config_file=/nvsynth/pyrightconfig.json
printf "{\n  \"extraPaths\": [\n" > $config_file.tmp
dirs=("/isaac-sim/exts" "/isaac-sim/kit/exts" "/isaac-sim/extscache" "/isaac-sim/extsPhysics")
for dir in "${dirs[@]}"; do
    find "$dir" -maxdepth 1 -type d -name 'omni*' -printf "    \"%p\",\n" >> $config_file.tmp
done
cat $config_file.tmp | head -c -2 > $config_file
printf "\n  ]\n}" >> $config_file
rm $config_file.tmp
