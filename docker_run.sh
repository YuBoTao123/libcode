#!/bin/bash

HERE=$(pwd)
DOCKER_USER="${USER}"
DEV_CONTAINER="nvcr.io/nvidia/tensorrt:21.05-py3"
# DEV_CONTAINER="nvcr.io/nvidia/tensorrt:24.06-py3-igpu"
# DATA_DIR="pingshan_data"
# echo "data dir is $DATA_DIR"
xhost +

docker run \
    -v $HERE:/workspace/onnx:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -w /workspace/onnx \
    -u root \
    --gpus all \
    -it --privileged=true --rm \
    --name=tensorrt \
    -e DISPLAY=unix$DISPLAY \
    $DEV_CONTAINER\
    /bin/bash
