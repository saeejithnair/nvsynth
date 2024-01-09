FROM nvcr.io/nvidia/isaac-sim:2023.1.1 AS base

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev build-essential && \
    apt-get clean

# personal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget lsof git neovim

COPY . /nvsynth

SHELL [ "/bin/bash", "-c" ]

# configure omni-provided python
RUN echo "export PATH=$PATH:/isaac-sim/kit/python/bin" >> /root/.bashrc && \
    echo "source /nvsynth/docker/setup_docker_env.sh" >> /root/.bashrc && \
    ln -s /isaac-sim/kit/python/bin/python3 /isaac-sim/kit/python/bin/python && \
    source /isaac-sim/setup_python_env.sh && \
    /isaac-sim/kit/python/bin/python -m pip install --upgrade pip && \
    /isaac-sim/kit/python/bin/pip install -r /nvsynth/docker/requirements.docker.txt

# overrides entrypoint from base image
WORKDIR /nvsynth
ENTRYPOINT bash
