#!/bin/bash
# Container entrypoint for training sessions.
# Starts TensorBoard in the background then opens an interactive shell.
# TB_LOGDIR env var controls the log directory (set by train.sh).

TB_LOGDIR="${TB_LOGDIR:-/dust3r/checkpoints}"

tensorboard --logdir "$TB_LOGDIR" --host 0.0.0.0 --port 6006 --reload_interval 5 \
    >/tmp/tensorboard.log 2>&1 &

echo "TensorBoard started on :6006 (logdir: $TB_LOGDIR, logs: /tmp/tensorboard.log)"

exec bash
