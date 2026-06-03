#!/bin/bash
# One-time bootstrap of the prerequisites for the CO3D-subset training
# smoke-test, run from inside the training container by train_entrypoint.sh.
#
# Both artifacts live under the bind-mounted host repo (/dust3r) so they
# persist across container runs and are only fetched once:
#   - data/co3d_subset_processed/   the preprocessed CO3D single-sequence subset
#   - checkpoints/CroCo_V2_...pth   the CroCo v2 checkpoint train.py warm-starts from
#
# Each step is guarded: if the artifact already exists it is skipped, so
# re-running the container goes straight to training. Mirrors the manual
# steps in the repo-root README.md ("Demo" section).

set -eu

cd /dust3r

CO3D_PROCESSED="data/co3d_subset_processed"
CROCO_CKPT="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth"
CROCO_URL="https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth"

# --- CO3D subset: download + preprocess ----------------------------------
if [ -f "$CO3D_PROCESSED/selected_seqs_train.json" ]; then
    echo "[prepare] CO3D subset already present at $CO3D_PROCESSED, skipping."
else
    echo "[prepare] Preparing CO3D single-sequence subset (one-time, this can take a while)..."
    mkdir -p data/co3d_subset
    if [ ! -d data/co3d_subset/co3d ]; then
        git clone https://github.com/facebookresearch/co3d data/co3d_subset/co3d
    fi
    python3 data/co3d_subset/co3d/co3d/download_dataset.py \
        --download_folder data/co3d_subset --single_sequence_subset
    rm -f data/co3d_subset/*.zip
    python3 datasets_preprocess/preprocess_co3d.py \
        --co3d_dir data/co3d_subset \
        --output_dir "$CO3D_PROCESSED" --single_sequence_subset
    echo "[prepare] CO3D subset ready at $CO3D_PROCESSED."
fi

# --- CroCo v2 checkpoint (--pretrained) ----------------------------------
if [ -f "$CROCO_CKPT" ]; then
    echo "[prepare] CroCo checkpoint already present at $CROCO_CKPT, skipping."
else
    echo "[prepare] Downloading CroCo v2 checkpoint (one-time)..."
    mkdir -p checkpoints
    wget -nc "$CROCO_URL" -P checkpoints/
    echo "[prepare] CroCo checkpoint ready at $CROCO_CKPT."
fi
