#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${CONTAINER_ENGINE:=docker}" "${IMAGE:=gblastn:thor}" "${JOBS:=4}" "${CUDA_ARCH:=110}"
exec "$CONTAINER_ENGINE" build --platform linux/arm64 --target "${TARGET:-runtime}" \
    --build-arg "JOBS=$JOBS" --build-arg "CUDA_ARCH=$CUDA_ARCH" \
    -t "$IMAGE" "$@" "$repo"
