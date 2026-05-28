#!/bin/bash
# Container entrypoint for a standalone TensorBoard server.
# TB_LOGDIR env var controls the log directory (set by tensorboard.sh).

TB_LOGDIR="${TB_LOGDIR:-/dust3r/checkpoints}"

echo "Serving TensorBoard on :6006 (logdir: $TB_LOGDIR)"
exec tensorboard --logdir "$TB_LOGDIR" --bind_all --port 6006 --reload_interval 5
