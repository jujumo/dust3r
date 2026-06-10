#!/bin/bash
# Container entrypoint for OverMaps training. Edit this file on the host to
# change the training config — the host repo is bind-mounted into /dust3r, so
# changes are live without rebuilding the image.
#
# This is a 224 linear-head smoke-test on the OverMaps dataset, warm-started
# from the CroCo v2 backbone. For the full 3-stage curriculum (224 linear ->
# 512 linear -> 512 dpt), see README "Our Hyperparameters" and overmaps_training.md.
#
# Prerequisites: the CroCo v2 checkpoint (fetched on first run) and the
# preprocessed OverMaps data at data/overmaps_processed/ (generated manually by
# datasets_preprocess/preprocess_overmaps.py). prepare_overmaps.sh checks both.
#
# GPU memory: this config fully fine-tunes the ViT-Large model and needs a
# large GPU (it OOMs an 8 GB card even at batch_size 1 — AdamW states alone are
# ~5 GB). To run on a small GPU, add freeze="encoder" to the --model string
# (trains only the decoders + head) and lower --batch_size; e.g. freeze the
# encoder with --batch_size 2 fits ~5.5 GB. Raise --accum_iter to keep the
# effective batch size up.

set -eu

cd /dust3r

/dust3r/docker/files/prepare_overmaps.sh

exec python train.py \
    --train_dataset "1000 @ OverMaps(split='train', ROOT='data/overmaps_processed', aug_crop=16, resolution=224, transform=ColorJitter)" \
    --test_dataset  "100 @ OverMaps(split='test', ROOT='data/overmaps_processed', resolution=224, seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion  "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 \
    --batch_size 4 --accum_iter 1 --num_workers 0 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_overmaps_224"
