#!/bin/bash
#
# Open an interactive shell inside the DUSt3R container with TensorBoard
# running in the background. Intended for training runs where you want live
# curve monitoring without a separate container.
#
# Usage:
#   bash train.sh [--cpu] [--engine=docker|podman] [--logdir=PATH]
#     --cpu               use the CPU image (default: CUDA, requires NVIDIA toolkit)
#     --engine=<name>     force docker or podman (default: auto-detect, prefer podman)
#     --logdir=<path>     TensorBoard log dir inside the container
#                         (default: /dust3r/checkpoints — covers all nested output_dirs)
#
# Once inside the shell, TensorBoard is already running. Open
# http://localhost:6006 on your host to see the curves, then launch your
# training command, e.g.:
#
#   python train.py \
#       --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', \
#           aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter)" \
#       --test_dataset "100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', \
#           resolution=224, seed=777)" \
#       --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), \
#           head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), \
#           conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, \
#           enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
#       --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
#       --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
#       --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
#       --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 \
#       --batch_size 4 --accum_iter 1 --num_workers 0 \
#       --save_freq 1 --keep_freq 5 --eval_freq 1 \
#       --output_dir "checkpoints/dust3r_demo_224"
#
# TensorBoard logs go to /tmp/tensorboard.log inside the container.
# The host repo is bind-mounted over /dust3r; croco/models/curope is masked
# by an anonymous volume so the image's compiled RoPE CUDA .so stays visible.

set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

compose_file="docker-compose-cuda.yml"
forced_engine=""
tb_logdir="/dust3r/checkpoints"

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
        --logdir=*)
            tb_logdir="${arg#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done

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

echo "Starting container — TensorBoard will be available at http://localhost:6006"

cd "$SCRIPT_DIR"

exec $compose_cmd -f "$compose_file" run --rm \
    -p 6006:6006 \
    -e TB_LOGDIR="$tb_logdir" \
    -v "$REPO_ROOT:/dust3r" \
    -v "/dust3r/croco/models/curope" \
    --entrypoint /dust3r/docker/files/train_entrypoint.sh \
    dust3r-demo
