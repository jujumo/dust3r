#!/bin/bash
#
# Build and launch the DUSt3R Gradio demo via docker-compose / podman-compose.
#
# Usage:
#   bash demo.sh [--with-cuda] [--model_name=<NAME>]
#     --with-cuda           use the CUDA compose file (requires NVIDIA toolkit)
#     --model_name=<NAME>   checkpoint basename (without .pth), default below
#
# The checkpoint is downloaded into ./files/checkpoints/, which the compose
# files bind-mount into the container at /dust3r/checkpoints.

# -e: stop on first error, -u: error on unset vars, -x: trace commands.
set -eux

# Default checkpoint (overridden by --model_name).
model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Fetch the checkpoint into the host-side dir that the container bind-mounts.
# Skipped if the file already exists, so repeated runs are cheap.
download_model_checkpoint() {
    if [ -f "./files/checkpoints/${model_name}" ]; then
        echo "Model checkpoint ${model_name} already exists. Skipping download."
        return
    fi
    echo "Downloading model checkpoint ${model_name}..."
    wget "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/${model_name}" -P ./files/checkpoints
}

# Pick a compose command and store it in $compose_cmd.
# Prefer podman if available, otherwise fall back to docker. For each engine,
# try the standalone "<engine>-compose" binary first, then the
# "<engine> compose" subcommand (compose-v2 / podman compose plugin).
detect_compose_cmd() {
    for engine in podman docker; do
        command -v "$engine" &>/dev/null || continue
        if command -v "${engine}-compose" &>/dev/null; then
            compose_cmd="${engine}-compose"
            return
        elif "$engine" compose version &>/dev/null; then
            compose_cmd="$engine compose"
            return
        fi
    done
    echo "No compose-capable container engine found. Install podman+podman-compose or docker+docker-compose and try again."
    exit 1
}

# Build and run the container. $MODEL is read by the compose files
# (see "${MODEL:-...}" interpolation in docker-compose-*.yml).
run_docker() {
    export MODEL=${model_name}
    if [ "$with_cuda" -eq 1 ]; then
        $compose_cmd -f docker-compose-cuda.yml up --build
    else
        $compose_cmd -f docker-compose-cpu.yml up --build
    fi
}

# Parse CLI flags.
with_cuda=0
for arg in "$@"; do
    case $arg in
        --with-cuda)
            with_cuda=1
            ;;
        --model_name=*)
            # Strip the "--model_name=" prefix and re-append .pth.
            model_name="${arg#*=}.pth"
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done


main() {
    download_model_checkpoint
    detect_compose_cmd
    run_docker
}

main
