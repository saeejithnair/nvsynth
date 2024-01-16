## -----------------------------------------
## Stage 1: Base dependencies installation.
## -----------------------------------------
FROM nvcr.io/nvidia/isaac-sim:2023.1.1 AS base

# Project name.
ENV PROJECT="nvsynth"

# Update system packages and install the basic ones
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends \
    apt-utils \
    software-properties-common \
    build-essential \
    sudo

# Copy apt dependencies file
COPY ./docker/deps/apt_requirements.txt /docker/deps/apt_requirements.txt

# Install additional dependencies from requirements file
RUN xargs -d '\n' -a /docker/deps/apt_requirements.txt \
    apt-get -y install \
    --no-install-recommends

# Clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/*

## -----------------------------------------
## Stage 2: User setup.
## -----------------------------------------
FROM base AS setup

# User and group ids to map the docker user to the host user.
ARG USER_ID=1000
ARG GROUP_ID=1000
# Group ID to add the docker user to a shared group the host user is part of.
ARG SHARED_GROUP_ID

ARG DOCKER_USERNAME="user"
ARG HOST_USERNAME="null"
ARG HOSTNAME
ENV USER_NAME $DOCKER_USERNAME
ENV HOST_USERNAME $HOST_USERNAME
ENV HOSTNAME $HOSTNAME
ENV USER_HOME="/home/${USER_NAME}"

# Add a non-root docker user with passwordless sudo permissions and map this
# docker user to the host user so that created files in the docker container
# are owned by the host user.
RUN addgroup --gid $GROUP_ID $USER_NAME && \
    adduser \
    --uid $USER_ID \
    --ingroup $USER_NAME \
    --home $USER_HOME \
    --shell /bin/bash \
    --disabled-password \
    --gecos "" \
    $USER_NAME && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Create shared group with same GID as a shared host group (e.g. vip_user) and
# add docker user to this secondary group so that they can access any shared
# dirs that might need to be mounted.
ENV SHARED_GROUP_NAME="host_shared"
RUN if [ ! -z $SHARED_GROUP_ID ]; then \
    groupadd -g $SHARED_GROUP_ID $SHARED_GROUP_NAME && \
    usermod -a -G $SHARED_GROUP_NAME $USER_NAME; \
    fi

# kinda hacky but MUCH FASTER than recursive chown. see https://github.com/docker/for-linux/issues/388
COPY --from=base --chown=${USER_ID}:${GROUP_ID} /isaac-sim /isaac-sim

USER $USER_NAME
WORKDIR $USER_HOME

# Add local user binary folder to PATH variable
ENV PATH="${PATH}:${USER_HOME}/.local/bin"

# Set default shell to bash.
SHELL ["/bin/bash", "-c"]

## -----------------------------------------
## Stage 3: Application building.
## -----------------------------------------
FROM setup AS app

# Copy dependency files
COPY ./docker/deps/pip_requirements.txt /docker/deps/pip_requirements.txt
COPY ./docker/deps/entrypoint.sh /home/user/entrypoint.sh
COPY ./docker/deps/setup_docker_env.sh /home/user/setup_docker_env.sh

USER root
RUN chmod +x /home/user/entrypoint.sh && \
    chown $USER_NAME:$USER_NAME /home/user/entrypoint.sh && \
    chmod +x /home/user/setup_docker_env.sh && \
    chown $USER_NAME:$USER_NAME /home/user/setup_docker_env.sh && \
    chown $USER_NAME:$USER_NAME /home/user/setup_docker_env.sh

# configure omni-provided python
RUN sed -i '1s/^/source \/home\/user\/setup_docker_env.sh\n/' /home/user/.bashrc && \
    ln -s /isaac-sim/kit/python/bin/python3 /isaac-sim/kit/python/bin/python && \
    source /home/user/setup_docker_env.sh && \
    python -m pip install --upgrade pip && \
    pip install -r /docker/deps/pip_requirements.txt

COPY . /nvsynth
WORKDIR /nvsynth

USER $USER_NAME
ENTRYPOINT ["/bin/bash", "-i", "/home/user/entrypoint.sh"]
