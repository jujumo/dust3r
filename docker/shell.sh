#!/bin/bash
#
# Open an interactive bash shell inside the DUSt3R container, intended for
# training and other manual work. Reuses the demo image (built by
# docker-compose-{cuda,cpu}.yml).
#
# Usage:
#   bash shell.sh [--cpu] [--engine=docker|podman]
#     --cpu               use the CPU image (default: CUDA, requires NVIDIA toolkit)
#     --engine=<name>     force docker or podman (default: auto-detect, prefer podman)
#
# The host repo is bind-mounted over /dust3r, so:
#   - code edits are live (no rebuild needed)
#   - data/ and any outputs you create persist on the host
#   - the in-image build of croco/models/curope is shadowed; rebuild inside
#     the container if you need the RoPE CUDA kernels:
#       cd croco/models/curope && python setup.py build_ext --inplace
#
# Inside the shell, follow the README "Training → Demo" section to train on
# the CO3D single-sequence subset. Quickstart (run on host BEFORE launching
# this shell, so the downloaded data ends up under ./data on the host):
#
#   mkdir -p data/co3d_subset
#   cd data/co3d_subset
#   git clone https://github.com/facebookresearch/co3d
#   python3 co3d/co3d/download_dataset.py --download_folder . --single_sequence_subset
#   rm *.zip
#   cd ../..
#   python3 datasets_preprocess/preprocess_co3d.py \
#       --co3d_dir data/co3d \
#       --output_dir data/co3d_processed \
#       --single_sequence_subset
#
# Then inside the shell launch step 1 of the 3-stage curriculum, e.g.:
#   torchrun --nproc_per_node=1 train.py \
#       --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_processed', resolution=224, ...)" \
#       ...  # see README for the full command

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

# Pick a compose command. By default prefer podman over docker; --engine=<name>
# forces a specific engine. For each engine, try "<engine>-compose" then
# "<engine> compose". (Same logic as demo.sh — duplicated rather than factored
# to keep each script self-contained.)
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

# "compose run --rm" gives us a one-shot container with the service's image,
# build settings, GPU reservation, and volumes — but bash instead of the
# default entrypoint. The extra -v overlays the live host repo over the
# image's baked-in /dust3r.
exec $compose_cmd -f "$compose_file" run --rm \
    -v "$REPO_ROOT:/dust3r" \
    --entrypoint bash \
    dust3r-demo
