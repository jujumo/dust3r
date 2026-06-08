#!/bin/bash
#
# Run a DUSt3R training session on the Slurm cluster, inside the Apptainer
# image built from the container by docker/to_apptainer.sh.
#
# This is the cluster counterpart of docker/train_overmaps.sh (which runs locally via
# docker/podman compose). BOTH run the exact same training command — it lives
# in docker/files/train_overmaps_entrypoint.sh, bind-mounted into the container, so the
# config (datasets, model, hyperparameters) stays in one place and the local
# and cluster trainers can't drift apart. Edit that file to change the run.
#
# Local (debug, no scheduler):   bash docker/train_overmaps.sh
# Cluster (this script):         bash docker/slurm_train_overmaps.sh
#
# Usage:
#   bash slurm_train_overmaps.sh
#   # override any resource via env var, e.g.:
#   PARTITION=gpu GPUS=2 TIME=4:00:00 bash slurm_train_overmaps.sh
#
#   SIF        path to the .sif            (default: ./files/dust3r.sif)
#   PARTITION  Slurm partition            (default: debug — chaos V100 nodes, 1-day cap)
#   ACCOUNT    Slurm account              (default: ffs-3d; empty to let Slurm pick)
#   GPUS       GPUs to request            (default: 1; see note on multi-GPU below)
#   CPUS       cpus-per-task              (default: 8)
#   MEM        memory                     (default: 48G)
#   TIME       walltime                   (default: 12:00:00)
#   JOBNAME    Slurm job name             (default: dust3r-train)
#
# Logs land in <repo>/slurm-logs/<jobname>-<jobid>.out. Watch a running job:
#   tail -f slurm-logs/dust3r-train-<jobid>.out
# Monitor training curves with TensorBoard (docker/tensorboard.sh) pointed at
# the same checkpoints/ output dir (it is on the bind-mounted host repo).
#
# NOTE on multi-GPU: train_overmaps_entrypoint.sh runs a single `python train.py`, so it
# uses ONE GPU regardless of GPUS. For real multi-GPU training switch that line
# to `torchrun --nproc_per_node=$GPUS train.py ...` (see README "Hyperparameters").

set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

SIF=${SIF:-$SCRIPT_DIR/files/dust3r.sif}
PARTITION=${PARTITION:-debug}
ACCOUNT=${ACCOUNT-ffs-3d}
GPUS=${GPUS:-1}
CPUS=${CPUS:-8}
MEM=${MEM:-48G}
TIME=${TIME:-12:00:00}
JOBNAME=${JOBNAME:-dust3r-train}

command -v sbatch &>/dev/null || { echo "sbatch not found — run this from a Slurm submit node."; exit 1; }
command -v apptainer &>/dev/null || { echo "apptainer not found on PATH."; exit 1; }

if [ ! -f "$SIF" ]; then
    echo "Apptainer image not found: $SIF"
    echo "Build it first: bash $SCRIPT_DIR/to_apptainer.sh"
    exit 1
fi

# Re-expose the image's compiled curope (RoPE CUDA .so) over the live repo
# bind-mount, mirroring what compose does with its anonymous volume in
# train_overmaps.sh/shell.sh. The host repo's croco/models/curope (a submodule) ships
# only sources, so without this the bind-mount would shadow the .so and we'd
# silently drop to the slow PyTorch RoPE. One-time, idempotent extraction.
CUROPE_CACHE="$SCRIPT_DIR/files/curope"
if ! ls "$CUROPE_CACHE"/*.so &>/dev/null; then
    echo "Extracting compiled curope from $SIF -> $CUROPE_CACHE (one-time)..."
    mkdir -p "$CUROPE_CACHE"
    apptainer exec "$SIF" cp -a /dust3r/croco/models/curope/. "$CUROPE_CACHE"/
fi

LOGDIR="$REPO_ROOT/slurm-logs"
mkdir -p "$LOGDIR"

# Optional --account (some clusters require it, some reject unknown ones).
account_opt=()
[ -n "$ACCOUNT" ] && account_opt=(--account="$ACCOUNT")

# Submit a real script file (files/train_overmaps_job.sbatch) rather than `sbatch
# --wrap=...`. A --wrap job has no on-disk script, so `scontrol show job`
# reports `Command=(null)` and TUIs like `slurmer` that read that path to show
# the batch script fail with "Failed to read script from path: (null)". A real
# file gives a concrete, persistent Command= path. The apptainer binds/image
# are resolved here and handed to the job via --export; see train_overmaps_job.sbatch.
JOB_SCRIPT="$SCRIPT_DIR/files/train_overmaps_job.sbatch"
echo "Submitting: partition=$PARTITION account=${ACCOUNT:-<default>} gpus=$GPUS time=$TIME"
echo "Image     : $SIF"
echo "Logs      : $LOGDIR/$JOBNAME-<jobid>.out"

exec sbatch \
    --job-name="$JOBNAME" \
    --partition="$PARTITION" \
    "${account_opt[@]}" \
    --gres="gpu:$GPUS" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOGDIR/%x-%j.out" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,SIF=$SIF,CUROPE_CACHE=$CUROPE_CACHE" \
    "$JOB_SCRIPT"
