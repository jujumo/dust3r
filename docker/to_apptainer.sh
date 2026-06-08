#!/bin/bash
#
# Convert the already-built DUSt3R container image into an Apptainer/Singularity
# .sif, for running the training on an HPC Slurm cluster where there is no
# container daemon and GPUs are handed out by the scheduler (so `compose`'s
# "count: all" GPU reservation is the wrong model). Apptainer is daemonless,
# runs as your user, and `apptainer exec --nv` honours the CUDA_VISIBLE_DEVICES
# that Slurm assigns to the job.
#
# This does NOT rebuild anything: it reads the image you already built with
# `demo.sh` / `shell.sh` and repacks it (a few minutes + a lot of temp space
# for a ~25 GB image).
#
# How it reads the source image depends on the engine, because the EL9
# apptainer build only ships *some* of the containers/image transports:
#   - docker: apptainer reads the running daemon directly (docker-daemon:),
#             no intermediate file.
#   - podman: this apptainer lacks the containers-storage: transport, so we
#             `podman save` the image to an OCI tarball and build from that
#             (oci-archive:). The tarball is ~25 GB and is deleted afterwards.
#
# Usage:
#   bash to_apptainer.sh [--engine=docker|podman] [--image=<ref>] [--output=<path>]
#     --engine=<name>   force docker or podman (default: auto-detect, prefer podman)
#     --image=<ref>     source image reference (default: auto-detect *dust3r-demo*)
#     --output=<path>   destination .sif (default: ./files/dust3r.sif)
#
# Temp space: needs ~50 GB of scratch (podman stages OCI blobs; apptainer
# unpacks the image; the podman path also writes a ~25 GB intermediate tarball).
# Both default to the small root fs (/var/tmp, /tmp) and WILL run out of space
# on a 25 GB image, so this script funnels them at one scratch dir: your
# APPTAINER_TMPDIR if set, otherwise the --output directory. Point it at a roomy
# filesystem (NOT /tmp), e.g.
#   APPTAINER_TMPDIR=/home/$USER/dust3r/apptainer-tmp bash to_apptainer.sh
#
# Once built, launch the training under Slurm with docker/slurm_train_Co3d.sh.

set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

forced_engine=""
image=""
output="$SCRIPT_DIR/files/dust3r.sif"
for arg in "$@"; do
    case $arg in
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
        --image=*)
            image="${arg#*=}"
            ;;
        --output=*)
            output="${arg#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done

command -v apptainer &>/dev/null || {
    echo "apptainer not found on PATH. On the NLE cluster it lives on the GPU/submit nodes."
    exit 1
}

# Pick a plain engine command (not compose) to read the local image store from.
# Prefer podman over docker; --engine=<name> forces one. (Same preference order
# as the sibling compose scripts.)
detect_engine() {
    local engines
    if [ -n "$forced_engine" ]; then
        engines="$forced_engine"
    else
        engines="podman docker"
    fi
    for e in $engines; do
        if command -v "$e" &>/dev/null; then
            engine="$e"
            return
        fi
    done
    echo "No container engine found to read the source image from. Install podman or docker."
    exit 1
}
detect_engine

# Auto-detect the source image if not given. demo.sh / shell.sh build it via
# compose, which auto-names it "<project>_<service>" = docker_dust3r-demo (the
# compose file declares no explicit image:). Pick the first match.
if [ -z "$image" ]; then
    image=$("$engine" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -m1 'dust3r-demo' || true)
    if [ -z "$image" ]; then
        echo "Could not find a *dust3r-demo* image in $engine's store."
        echo "Build it first (cd docker && bash demo.sh), or pass --image=<ref>."
        exit 1
    fi
fi

mkdir -p "$(dirname "$output")"

# Everything here needs ~25 GB of scratch: `podman save` stages OCI blobs in a
# temp dir (defaults to /var/tmp), and `apptainer build` unpacks the image in
# its own temp dir. Both default to the small root filesystem and will run out
# of space on a 25 GB image. Funnel BOTH at one roomy scratch dir: the user's
# APPTAINER_TMPDIR if set, else next to the output (same filesystem as the .sif,
# which is where there's room). podman honours TMPDIR; apptainer APPTAINER_TMPDIR.
scratch="${APPTAINER_TMPDIR:-$(dirname "$output")}"
mkdir -p "$scratch"
export TMPDIR="$scratch"
export APPTAINER_TMPDIR="$scratch"

# docker exposes a daemon apptainer can read directly; podman doesn't, and this
# apptainer can't read podman's containers-storage, so we stage an OCI tarball.
if [ "$engine" = docker ]; then
    source_uri="docker-daemon:$image"
    archive=""
else
    archive="$scratch/dust3r.oci.tar"
    source_uri="oci-archive:$archive"
fi

echo "Engine : $engine"
echo "Image  : $image"
echo "Output : $output"
echo "Scratch: $scratch (needs ~50 GB free)"
[ -n "$archive" ] && echo "Tarball: $archive (intermediate, removed on exit)"
echo

# Stage + clean up the intermediate tarball for the podman path.
if [ -n "$archive" ]; then
    trap 'rm -f "$archive"' EXIT
    echo "[1/2] podman save -> $archive"
    "$engine" save --format oci-archive -o "$archive" "$image"
    echo "[2/2] apptainer build"
fi

# --force overwrites a stale .sif from a previous run.
apptainer build --force "$output" "$source_uri"

echo
echo "Built $output"
echo "Run the training under Slurm with: bash $SCRIPT_DIR/slurm_train_Co3d.sh"
