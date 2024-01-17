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
## Stage 2: Application building.
## -----------------------------------------
FROM base AS app

# Copy dependency files
COPY ./docker/deps/pip_requirements.txt /docker/deps/pip_requirements.txt
COPY ./docker/deps/entrypoint.sh /root/entrypoint.sh
COPY ./docker/deps/setup_docker_env.sh /root/setup_docker_env.sh

# Set default shell to bash.
SHELL ["/bin/bash", "-c"]

# configure omni-provided python
RUN sed -i '1s/^/source \/root\/setup_docker_env.sh\n/' /root/.bashrc && \
    ln -s /isaac-sim/kit/python/bin/python3 /isaac-sim/kit/python/bin/python && \
    source /root/setup_docker_env.sh && \
    python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /docker/deps/pip_requirements.txt

COPY . /nvsynth
WORKDIR /nvsynth

USER $USER_NAME
ENTRYPOINT ["/bin/bash", "-i", "/root/entrypoint.sh"]
