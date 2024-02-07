#!/bin/bash
#
# The `source`ing of this script is added .bashrc when only when a
#   devcontainer is created. Place any exports within the last if
#   statement. It checks that vscode's `code` binary is available
#   so that the exports are only done 

# checks the difference between our `.kit` configuration files and
#   the default configs provided by nvidia, using vscode's diff.
cmp-kit() {
    if [ "$#" -ne 1 ]; then
        echo "Usage: cmp-kit <filename>"
        return 1
    fi

    nvsynth_kit_cfg="/nvsynth/.configs/isaac-sim/${1}"
    base_kit_cfg="/isaac-sim/apps/${1}"

    if [ ! -f $nvsynth_kit_cfg ]; then
        echo "Error: Source file '$nvsynth_kit_cfg' does not exist."
        return 1
    fi

    if [ ! -f $base_kit_cfg ]; then
        echo "Error: Source file '$base_kit_cfg' does not exist."
        return 1
    fi

    code --diff $base_kit_cfg $nvsynth_kit_cfg
}

if type code >/dev/null 2>&1; then
    export -f cmp-kit
fi
