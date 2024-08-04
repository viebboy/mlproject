#!/bin/bash
# start container for interactive dev
PACKAGE_NAME=mlproject

# Initialize an empty variable to store ADD_DOCKER_OPTION
ADD_DOCKER_OPTION=""

# Loop through all the arguments
while [[ "$#" -gt 0 ]]; do
    ADD_DOCKER_OPTION="$1"
    shift
done

if [[ -z "$ADD_DOCKER_OPTION" ]]; then
  docker run --gpus all --ipc=host -it ${ADD_DOCKER_OPTION} ${PACKAGE_NAME}:latest
else
  docker run --gpus all --ipc=host -it ${PACKAGE_NAME}:latest
fi
