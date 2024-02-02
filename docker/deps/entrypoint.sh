#!/bin/bash
source ~/setup_docker_env.sh

# Run the CMD passed as command-line arguments or keep container open.
if [ $# -eq 0 ]; then
  echo "Starting container and waiting forever..."
  exec sleep inf
else
  echo "Running command: $@"
  exec "$@"
fi
