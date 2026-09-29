#!/bin/sh
# Mount a work directory; output files retain the caller's uid/gid.
set -eu
: "${CONTAINER_ENGINE:=docker}" "${IMAGE:=gblastn:thor}" "${WORK_DIR:=$PWD}" "${GPU:=1}"
WORK_DIR=$(CDPATH= cd -- "$WORK_DIR" && pwd)
[ "$#" -gt 0 ] || set -- blastn -help
if [ "$GPU" = 1 ]; then
    case $(basename "$CONTAINER_ENGINE") in
        podman) set -- --group-add keep-groups --device=nvidia.com/gpu=all "$IMAGE" "$@";;
        *) set -- --gpus all --runtime=nvidia "$IMAGE" "$@";;
    esac
elif [ "$GPU" = 0 ]; then
    # Override the CUDA image's NVIDIA_VISIBLE_DEVICES=all for genuine CPU checks.
    set -- --env NVIDIA_VISIBLE_DEVICES=void "$IMAGE" "$@"
else
    echo 'GPU must be 1 (passthrough) or 0 (no GPU access)' >&2; exit 2
fi
if [ "$(basename "$CONTAINER_ENGINE")" = podman ]; then
    set -- --userns=keep-id "$@"
fi
# Forward only documented tuning/diagnostic settings that the caller exported.
for setting in GBLASTN_GPU_TRACE GBLASTN_GPU_MEMORY_LIMIT_MB GBLASTN_PIPELINE_TRACE BATCH_SIZE; do
    if printenv "$setting" >/dev/null 2>&1; then
        set -- --env "$setting" "$@"
    fi
done
set -- run --rm --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --workdir /work --mount "type=bind,src=$WORK_DIR,dst=/work" "$@"
if [ "$(basename "$CONTAINER_ENGINE")" = podman ] && [ -n "${OCI_RUNTIME:-}" ]; then
    set -- "--runtime=$OCI_RUNTIME" "$@"
fi
exec "$CONTAINER_ENGINE" "$@"
