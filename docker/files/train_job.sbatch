#!/bin/bash
#
# Batch script executed on the compute node by docker/slurm_train.sh.
#
# We submit this as a real script FILE (not `sbatch --wrap=...`) on purpose:
# a --wrap job has no script on disk, so `scontrol show job` reports
# `Command=(null)`. Tools that display a job's batch script by reading that
# path — e.g. the `slurmer` TUI — then fail with
# "Failed to read script from path: (null)". Submitting a real file makes
# `Command=` a concrete, persistent path, so those tools show this script.
#
# slurm_train.sh passes REPO_ROOT, SIF and CUROPE_CACHE through the job
# environment via `sbatch --export`. All resource directives (partition, gres,
# time, output, ...) are supplied by slurm_train.sh on the sbatch command line,
# so there are deliberately no #SBATCH lines here.
#
# apptainer exec --nv injects the host NVIDIA driver/libs and honours the
# CUDA_VISIBLE_DEVICES that Slurm assigns this job, so the container only sees
# the allocated GPU(s). The two binds overlay the live host repo over the
# image's baked-in /dust3r, then re-mask curope with the compiled-.so cache.

set -eu

apptainer exec --nv \
    --bind "${REPO_ROOT:?REPO_ROOT not set — submit via docker/slurm_train.sh}":/dust3r \
    --bind "${CUROPE_CACHE:?CUROPE_CACHE not set — submit via docker/slurm_train.sh}":/dust3r/croco/models/curope \
    "${SIF:?SIF not set — submit via docker/slurm_train.sh}" \
    bash /dust3r/docker/files/train_entrypoint.sh
