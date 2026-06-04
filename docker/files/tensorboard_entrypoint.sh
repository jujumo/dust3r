#!/bin/bash
# Container entrypoint for a standalone TensorBoard server.
# TB_LOGDIR env var controls the log directory (set by tensorboard.sh).

TB_LOGDIR="${TB_LOGDIR:-/dust3r/checkpoints}"

# TensorBoard talks to its own data-ingester over a loopback gRPC socket. If the
# container inherits an http(s)_proxy (it does on the NLE hosts via /etc/proxyrc),
# gRPC routes even localhost through the proxy, which 503s and the UI shows no
# data. Exempt loopback from the proxy so the internal connection is direct.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

echo "Serving TensorBoard on :6006 (logdir: $TB_LOGDIR)"
exec tensorboard --logdir "$TB_LOGDIR" --bind_all --port 6006 --reload_interval 5
