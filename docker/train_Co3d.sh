#!/bin/bash
#
# Run a DUSt3R training session in its own container. The actual training
# command lives in docker/files/train_Co3d_entrypoint.sh — edit that file on the
# host to change the training config (bind-mounted, so changes are live
# without rebuilding the image).
#
# Monitor curves by launching docker/tensorboard.sh in another terminal
# (http://localhost:6006 on the host, or tunnel via ssh).
#
# Usage:
#   bash train_Co3d.sh [--cpu] [--engine=docker|podman]
#     --cpu               use the CPU image (default: CUDA, requires NVIDIA toolkit)
#     --engine=<name>     force docker or podman (default: auto-detect, prefer podman)
#
# The host repo is bind-mounted over /dust3r; croco/models/curope is masked
# by an anonymous volume so the image's compiled RoPE CUDA .so stays visible.

set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

compose_file="docker-compose-cuda.yml"
forced_engine=""

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

cd "$SCRIPT_DIR"

exec $compose_cmd -f "$compose_file" run --rm \
    -v "$REPO_ROOT:/dust3r" \
    -v "/dust3r/croco/models/curope" \
    --entrypoint /dust3r/docker/files/train_Co3d_entrypoint.sh \
    dust3r-demo
