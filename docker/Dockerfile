FROM nvcr.io/nvidia/isaac-sim:2023.1.1 AS isaac-sim

# personal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget lsof git neovim

# override entrypoint as list to enable command args in docker-compose.yml
ENTRYPOINT ["/isaac-sim/runheadless.native.sh"]

FROM isaac-sim AS nvsynth

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev build-essential && \
    apt-get clean

COPY . /nvsynth

WORKDIR /nvsynth

SHELL [ "/bin/bash", "-c" ]

# configure omni-provided python
RUN echo "source /nvsynth/docker/setup_docker_env.sh" >> /root/.bashrc && \
    ln -s /isaac-sim/kit/python/bin/python3 /isaac-sim/kit/python/bin/python && \
    source /nvsynth/docker/setup_docker_env.sh && \
    python -m pip install --upgrade pip && \
    pip install -r /nvsynth/docker/requirements.docker.txt

# set entrypoint to run a command, with the default being bash
ENTRYPOINT ["/bin/bash", "-i", "-c"]
CMD ["/bin/bash"]
