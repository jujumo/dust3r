#!/bin/bash

set -eux

# Default model name
model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

download_model_checkpoint() {
    if [ -f "./files/checkpoints/${model_name}" ]; then
        echo "Model checkpoint ${model_name} already exists. Skipping download."
        return
    fi
    echo "Downloading model checkpoint ${model_name}..."
    wget "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/${model_name}" -P ./files/checkpoints
}

detect_compose_cmd() {
    # Prefer podman if available, otherwise fall back to docker.
    # For each engine, try the standalone "<engine>-compose" binary first,
    # then the "<engine> compose" subcommand.
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

run_docker() {
    export MODEL=${model_name}
    if [ "$with_cuda" -eq 1 ]; then
        $compose_cmd -f docker-compose-cuda.yml up --build
    else
        $compose_cmd -f docker-compose-cpu.yml up --build
    fi
}

with_cuda=0
for arg in "$@"; do
    case $arg in
        --with-cuda)
            with_cuda=1
            ;;
        --model_name=*)
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
