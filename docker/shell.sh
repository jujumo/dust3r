#!/bin/bash
#
# Open an interactive bash shell inside the DUSt3R container, intended for
# training and other manual work. Reuses the demo image (built by
# docker-compose-{cuda,cpu}.yml).
#
# Usage:
#   bash shell.sh [--cpu] [--engine=docker|podman] [-- CMD [ARG...]]
#     --cpu               use the CPU image (default: CUDA, requires NVIDIA toolkit)
#     --engine=<name>     force docker or podman (default: auto-detect, prefer podman)
#     -- CMD [ARG...]     run CMD once inside the container (non-interactively) and
#                         exit, instead of opening an interactive shell. Everything
#                         after -- is passed through verbatim, e.g.:
#                           bash shell.sh --cpu -- python3 demo.py --help
#                           bash shell.sh -- bash -c 'cd /dust3r && torchrun ...'
#
# The host repo is bind-mounted over /dust3r, so:
#   - code edits are live (no rebuild needed)
#   - data/ and any outputs you create persist on the host
#   - croco/models/curope is masked by an anonymous volume so the image's
#     compiled RoPE CUDA .so survives the bind-mount. Side effect: edits to
#     files under croco/models/curope/ are NOT live in the container. If you
#     change kernels.cu / curope.cpp, rebuild the image (or rebuild in-place
#     inside the container: cd croco/models/curope && python setup.py \
#     build_ext --inplace).
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
run_cmd=()        # if non-empty, run this once inside the container instead of an interactive shell
while [ $# -gt 0 ]; do
    case $1 in
        --cpu)
            compose_file="docker-compose-cpu.yml"
            ;;
        --engine=*)
            forced_engine="${1#*=}"
            case $forced_engine in
                docker|podman) ;;
                *)
                    echo "Unknown engine: $forced_engine (expected docker or podman)"
                    exit 1
                    ;;
            esac
            ;;
        --)
            shift
            run_cmd=("$@")    # everything after -- is the command to run
            break
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "(to run a command instead of an interactive shell, put it after --)"
            exit 1
            ;;
    esac
    shift
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
# default entrypoint. The first -v overlays the live host repo over the
# image's baked-in /dust3r; the second is an anonymous volume that re-masks
# croco/models/curope/ with the image's contents, so the .so compiled at
# build time (which the host repo doesn't have) is visible at runtime.
run_opts=(--rm
    -v "$REPO_ROOT:/dust3r"
    -v "/dust3r/croco/models/curope"
    --entrypoint bash)

if [ ${#run_cmd[@]} -gt 0 ]; then
    # Non-interactive: run the user's command once and exit. With "--entrypoint
    # bash", the args after the service name are bash's argv, so `-c '"$@"' shell
    # CMD ARG...` execs CMD with its arguments intact (no re-quoting / splitting).
    #
    # Two benign messages are filtered from stderr in this mode:
    #   - "The input device is not a TTY" (appears only when stdin is piped);
    #   - "rootless netns: kill network process: permission denied" — a podman
    #     5 / netavark rootless quirk on this NFS workstation: the container is
    #     still removed, no netns process leaks, and the exit code is 0; podman
    #     just can't signal the shared netns helper on teardown.
    # stdout, the command's own stderr, the exit code, and any *other* podman
    # error all pass through untouched.
    exec 3>&1
    set +e
    $compose_cmd -f "$compose_file" run \
        "${run_opts[@]}" \
        dust3r-demo -c '"$@"' shell "${run_cmd[@]}" 2>&1 1>&3 \
        | grep -vE 'The input device is not a TTY|rootless netns: kill network process: permission denied' >&2
    rc=${PIPESTATUS[0]}
    set -e
    exec 3>&-
    exit "$rc"
fi

# Interactive shell: publish the compose file's "ports:" (gradio's 37860) so a
# demo started by hand inside the shell is reachable from the host. (Skipped in
# command mode, where publishing a port is usually unwanted and can clash.)
run_opts+=(--service-ports)
exec $compose_cmd -f "$compose_file" run \
    "${run_opts[@]}" \
    dust3r-demo
