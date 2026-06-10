#!/bin/bash
# One-time bootstrap of the prerequisites for the OverMaps training smoke-test,
# run from inside the training container by train_overmaps_entrypoint.sh.
#
# Both artifacts live under the bind-mounted host repo (/dust3r) so they
# persist across container runs:
#   - checkpoints/CroCo_V2_...pth   the CroCo v2 backbone train.py warm-starts from
#   - data/overmaps_processed/      the preprocessed OverMaps dataset
#
# The CroCo checkpoint is fetched if missing. OverMaps preprocessing is a
# deliberate manual step (it needs the raw OverMaps data + pycolmap), so this
# script only *checks* the processed data is present and errors out with the
# command to run otherwise -- it does not regenerate it.

set -eu

cd /dust3r

CROCO_CKPT="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth"
CROCO_URL="https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth"
OVERMAPS_PROCESSED="data/overmaps_processed"

# --- CroCo v2 checkpoint (--pretrained) ----------------------------------
if [ -f "$CROCO_CKPT" ]; then
    echo "[prepare] CroCo checkpoint already present at $CROCO_CKPT, skipping."
else
    echo "[prepare] Downloading CroCo v2 checkpoint (one-time)..."
    mkdir -p checkpoints
    wget -nc "$CROCO_URL" -P checkpoints/
    echo "[prepare] CroCo checkpoint ready at $CROCO_CKPT."
fi

# --- preprocessed OverMaps data (checked, not generated) -----------------
if [ -d "$OVERMAPS_PROCESSED" ] && ls "$OVERMAPS_PROCESSED"/*/pairs.json >/dev/null 2>&1; then
    echo "[prepare] OverMaps processed data present at $OVERMAPS_PROCESSED."
else
    echo "[prepare] ERROR: no preprocessed OverMaps data at $OVERMAPS_PROCESSED." >&2
    echo "[prepare] Run the preprocessor first (raw data + pycolmap required), e.g.:" >&2
    echo "    python3 datasets_preprocess/preprocess_overmaps.py \\" >&2
    echo "        --overmaps_dir data/OverMaps-1K --output_dir $OVERMAPS_PROCESSED" >&2
    exit 1
fi
