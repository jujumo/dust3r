#!/bin/bash
#
# Run a standalone TensorBoard server in its own container. Reuses the
# dust3r image so no extra dependencies are needed. The host repo is bind-
# mounted read-only at /dust3r so TensorBoard sees the same checkpoints/
# directory the training container writes to.
#
# Usage:
#   bash tensorboard.sh [--cpu] [--engine=docker|podman] [--logdir=PATH] [--port=N]
#     --cpu               use the CPU image (default: CUDA — TB doesn't use the GPU
#                         but the image name follows whichever compose file you build)
#     --engine=<name>     force docker or podman (default: auto-detect, prefer podman)
#     --logdir=<path>     log directory inside the container
#                         (default: /dust3r/checkpoints — covers all nested output_dirs)
#     --port=<n>          host port to publish (default: 6006)
#
# Open http://localhost:<port> on the host. To reach it from another machine,
# tunnel through SSH from your PC:
#     ssh -fNL 6006:localhost:6006 <server>
#
# Run this alongside shell.sh / train.sh — TensorBoard picks up new event
# files automatically (--reload_interval 5).

set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

compose_file="docker-compose-cuda.yml"
forced_engine=""
tb_logdir="/dust3r/checkpoints"
tb_port="6006"

for arg in "$@"; do
    case $arg in
        --cpu)
            compose_file="docker-compose-cpu.yml"
            ;;
        --engine=*)
            forced_engine="${arg#*=}"
            case $forced_engine in
                docker|podman) ;;
                *)
                    echo "Unknown engine: $forced_engine (expected docker or podman)"
                    exit 1
                    ;;
            esac
            ;;
        --logdir=*)
            tb_logdir="${arg#*=}"
            ;;
        --port=*)
            tb_port="${arg#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done

detect_compose_cmd() {
    local engines
    if [ -n "$forced_engine" ]; then
        engines="$forced_engine"
    else
        engines="podman docker"
    fi
    for engine in $engines; do
        command -v "$engine" &>/dev/null || continue
        if command -v "${engine}-compose" &>/dev/null; then
            compose_cmd="${engine}-compose"
            return
        elif "$engine" compose version &>/dev/null; then
            compose_cmd="$engine compose"
            return
        fi
    done
    if [ -n "$forced_engine" ]; then
        echo "Engine '$forced_engine' was requested but no working compose command found for it. Install ${forced_engine}-compose or '${forced_engine} compose'."
    else
        echo "No compose-capable container engine found. Install podman+podman-compose or docker+docker-compose and try again."
    fi
    exit 1
}
detect_compose_cmd

echo "Starting TensorBoard container — will be available at http://localhost:${tb_port}"

cd "$SCRIPT_DIR"

exec $compose_cmd -f "$compose_file" run --rm \
    -p "${tb_port}:6006" \
    -e TB_LOGDIR="$tb_logdir" \
    -v "$REPO_ROOT:/dust3r" \
    --entrypoint /dust3r/docker/files/tensorboard_entrypoint.sh \
    dust3r-demo
