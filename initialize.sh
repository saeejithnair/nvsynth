#!/bin/bash
# derived from https://github.com/saeejithnair/elastic-nerf/blob/main/initialize.sh

print_error() {
    echo -e "\033[31m[ERROR]: $1\033[0m"
}

print_warning() {
    echo -e "\033[33m[WARNING]: $1\033[0m"
}

print_info() {
    echo -e "\033[32m[INFO]: $1\033[0m"
}

SHARED_GROUP_NAME="vip_user"
SHARED_GROUP_ID=$(getent group $SHARED_GROUP_NAME | cut -d: -f3)

# Get UID and GID.
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Validate that the user is a member of the group.
if ! groups $USER | grep &>/dev/null "\b$SHARED_GROUP_NAME\b"; then
    echo "User ${USER} with UID ${USER_ID} is not a member of group ${SHARED_GROUP_NAME} with GID ${SHARED_GROUP_ID}"
    exit 1
fi

PROJECT_NAME="nvsynth"

# Set username for non-root docker user.
if [ "$USER_ID" != "1000" ] && [ "$GROUP_ID" != "1000" ]; then
    DOCKER_USERNAME="user"
else 
    # When the uid and gid are 1000, it means the current user
    # was the first non-root user created. The base image already
    # has user with uid=gid=1000, 'ubuntu', so we must use it.
    DOCKER_USERNAME="ubuntu"
fi

# Configure data directory
printf "Please enter an absolute path to your food dataset directory (empty for none): "
read DATASET_PATH

if [ -n "$DATASET_PATH" ]; then
    if [[ "$DATASET_PATH" != /* ]]; then
        echo "The provided path, '$DATASET_PATH', is not an absolute path. Exiting."
        exit 1
    fi
    if [ ! -d "$DATASET_PATH" ]; then
        echo "The provided path, '$DATASET_PATH', is not a directory. Exiting."
        exit 1
    fi
fi

> ".env"
echo "COMPOSE_PROJECT_NAME=$PROJECT_NAME-${USER}" >> ".env"
echo "USER_ID=$USER_ID" >> ".env"
echo "GROUP_ID=$GROUP_ID" >> ".env"
echo "SHARED_GROUP_NAME=$SHARED_GROUP_NAME" >> ".env"
echo "SHARED_GROUP_ID=$SHARED_GROUP_ID" >> ".env"
echo "USERNAME=${USER}" >> ".env"
echo "DOCKER_USERNAME=${DOCKER_USERNAME}" >> ".env"
if [ -n "$DATASET_PATH" ]; then
    echo DATASET_PATH=$DATASET_PATH >> .env
fi

# Create required folders
isaac_sim_folders=(
    ~/.docker/isaac-sim/cache/kit/ogn_generated
    ~/.docker/isaac-sim/cache/kit/shadercache/common
    ~/.docker/isaac-sim/cache/ov
    ~/.docker/isaac-sim/cache/pip
    ~/.docker/isaac-sim/cache/warp
    ~/.docker/isaac-sim/cache/glcache
    ~/.docker/isaac-sim/cache/computecache
    ~/.docker/isaac-sim/cache/omni-pycache
    ~/.docker/isaac-sim/logs
    ~/.docker/isaac-sim/local/share/ov/data
    ~/.docker/isaac-sim/documents
)
mkdir -p ${isaac_sim_folders[@]}

# Check that relevant folders are owned by the current user
not_owned_by_user=$(find ~/.docker/isaac-sim/ ! -user $(whoami) -print)
if [ -n "$not_owned_by_user" ]; then
    printf "\nWARNING: the following files/folders are not owned by you:\n"
    echo $not_owned_by_user
    printf "\nThis may lead to permission errors for isaac-sim. Try running: 'chown -R $(id -u):$(id -g) ~/.docker/isaac-sim'\n"
fi

echo "Done setting up .env file for user, '$USER', with uid, '$UID'."
