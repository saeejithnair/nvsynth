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
DOCKER_USERNAME="user"

> ".env"
echo "COMPOSE_PROJECT_NAME=$PROJECT_NAME-${USER}" >> ".env"
echo "USER_ID=$USER_ID" >> ".env"
echo "GROUP_ID=$GROUP_ID" >> ".env"
echo "SHARED_GROUP_NAME=$SHARED_GROUP_NAME" >> ".env"
echo "SHARED_GROUP_ID=$SHARED_GROUP_ID" >> ".env"
echo "USERNAME=${USER}" >> ".env"
echo "DOCKER_USERNAME=${DOCKER_USERNAME}" >> ".env"
