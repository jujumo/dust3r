# Understanding the training example (`train_Co3d_entrypoint.sh`)

This document explains the rationale and behavior of the default training
example shipped in `docker/files/train_Co3d_entrypoint.sh`. It is the baseline we
will adapt to train on a custom dataset — so each section flags what is
**example-specific** (will change for a new dataset) vs. **structural** (stays).

## 1. What the script is and where it runs

`docker/files/train_Co3d_entrypoint.sh` is the command executed **inside the training
container** (launched by `docker/train_Co3d.sh` / the compose files). The host repo
is bind-mounted at `/dust3r`, so editing this file on the host changes the next
run with no image rebuild.

The default config is the README "Demo" smoke-test: a tiny 10-epoch run on a
single-sequence CO3D subset, just to prove the training loop works end-to-end.

```bash
set -eu                              # abort on error / unset var
cd /dust3r                           # repo root (bind-mounted host repo)
/dust3r/docker/files/prepare_co3d.sh # one-time bootstrap of prerequisites
exec python train.py ...             # replace shell with the trainer (PID 1)
```

- `prepare_co3d.sh` is idempotent: downloads + preprocesses the CO3D
  single-sequence subset into `data/co3d_subset_processed/` and fetches the
  CroCo v2 checkpoint into `checkpoints/`. Both live under the bind-mounted
  repo, so they persist on the host and are skipped on later runs.
- `exec` replaces the shell with Python so signals (Ctrl-C, docker/SLURM stop)
  reach the trainer directly.
- `train.py` is a 3-line shim calling `get_args_parser()` + `train(args)` from
  `dust3r/training.py`.

> **Adapt for custom dataset:** `prepare_co3d.sh` is entirely CO3D-specific.
> A custom dataset needs its own bootstrap (download + preprocess into the
> layout the dataset class expects), or the prerequisites prepared manually.

## 2. The key mechanism: eval()'d expression strings

`--model`, `--train_dataset`, `--test_dataset`, `--train_criterion`,
`--test_criterion` are **Python expression strings that get `eval()`'d** inside
`dust3r/training.py`. Every symbol used must be importable there — that's why
`training.py` does `from dust3r.losses import *`, imports the model, the dataset
classes from `dust3r/datasets/__init__.py`, etc.

> **Adapt for custom dataset:** a custom dataset class must be importable from
> `dust3r/datasets/__init__.py` for its name to be usable in `--train_dataset`.

## 3. Flag-by-flag

### Datasets — `"N @ DatasetCls(args)"`

```
--train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed',
                             aug_crop=16, mask_bg='rand',
                             resolution=224, transform=ColorJitter)"
--test_dataset  "100  @ Co3d(split='test',  ROOT='data/co3d_subset_processed',
                             resolution=224, seed=777)"
```

- `N @` is a per-epoch sample-count weight (1000 train pairs, 100 eval pairs).
  Multiple datasets can be summed with `+`.
- `aug_crop=16`, `mask_bg='rand'`, `transform=ColorJitter` are **train-only
  augmentations**. Eval uses none and a fixed `seed=777` for reproducibility.
- `resolution=224` matches the model `img_size`.

> **Adapt for custom dataset:** this is the line that changes the most — swap
> `Co3d(...)` for the custom dataset class, point `ROOT` at the preprocessed
> data, set the sample count, and decide which augmentations apply.

### Model — the asymmetric Siamese ViT

```
--model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224),
         head_type='linear', output_mode='pts3d',
         depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf),
         enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
         dec_embed_dim=768,  dec_depth=12, dec_num_heads=12)"
```

The string is `eval()`'d into a call to `AsymmetricCroCo3DStereo.__init__`
(`dust3r/model.py`). That class subclasses **`CroCoNet`** (the CroCo v2 backbone
in the `croco/` submodule), so the arguments split into two groups: a handful
consumed by DUSt3R itself, and the rest (`**croco_kwargs`) forwarded straight to
the `CroCoNet` constructor. Any kwarg you don't pass falls back to the
constructor default (`fill_default_args` records the effective values).

#### What "asymmetric" means

`__init__` runs the normal CroCo setup, then does
`self.dec_blocks2 = deepcopy(self.dec_blocks)` — a **second, independent decoder
stack**. View1 is decoded by `dec_blocks`, view2 by `dec_blocks2`, but **both
outputs are expressed in view1's coordinate frame**. That asymmetry is the whole
point of the model (and the source of the name). `load_state_dict` duplicates
the pretrained single-decoder weights into `dec_blocks2` when a checkpoint
doesn't already contain them, so warm-starting from CroCo "just works".

#### DUSt3R-specific arguments

- **`output_mode='pts3d'`** — what the head predicts. Combined with `head_type`
  in `head_factory` (`dust3r/heads/__init__.py`); only `'pts3d'` is implemented,
  meaning a per-pixel 3D pointmap.
- **`head_type='linear'`** — selects the downstream head:
  - `'linear'` → `LinearPts3d`, a single linear projection from decoder tokens
    to an `H×W×(3+1)` map (xyz + confidence). Lightweight; used by the 224 model.
  - `'dpt'` → `PixelwiseTaskWithDPT`, a heavier dense DPT decoder used by the
    512 checkpoints.
  Two heads are built (`downstream_head1`/`2`), one per view, then wrapped by
  `transpose_to_landscape` so non-square / portrait inputs are handled.
- **`depth_mode=('exp', -inf, inf)`** — activation + `(vmin, vmax)` clamp applied
  to the xyz channels in `reg_dense_depth` (`dust3r/heads/postprocess.py`).
  `'exp'` regresses an unbounded positive distance via `expm1`; alternatives are
  `'linear'` and `'square'`. The `(-inf, inf)` here means "no clamp".
- **`conf_mode=('exp', 1, inf)`** — activation + clamp for the confidence channel
  in `reg_dense_conf`. `'exp'` gives `1 + exp(x)`, i.e. confidence floored at 1
  and unbounded above (`'sigmoid'` is the other option). Passing `conf_mode=None`
  builds heads with no confidence channel — but then `ConfLoss` can't be used.
- **`freeze='none'`** (default, not shown) — `'mask'` freezes the mask token;
  `'encoder'` freezes patch-embed + encoder blocks (train only the decoders +
  heads). Useful when fine-tuning on a small custom dataset.
- **`patch_embed_cls`** (default `'PatchEmbedDust3R'`, not shown) — at training
  time the README curriculum uses `'ManyAR_PatchEmbed'` to allow mixed aspect
  ratios within a batch; at inference `load_model` rewrites it back to
  `'PatchEmbedDust3R'` and forces `landscape_only=False`.

#### Backbone (`CroCoNet`) arguments — must match the checkpoint

These are forwarded to `CroCoNet` and define the network geometry. Because
`--pretrained` loads weights by tensor name **and shape**, they must match the
checkpoint exactly or `load_state_dict` will mismatch:

- **`pos_embed='RoPE100'`** — rotary 2D position embedding with frequency base
  100 (vs. the `'cosine'` sinusoidal default). The optional `croco/models/curope`
  CUDA kernel accelerates RoPE; without it there's a slower pure-Python path.
- **`img_size=(224, 224)`** — input resolution; must be a multiple of
  `patch_size` (16, the default) and must agree with the dataset `resolution=224`.
- **`enc_embed_dim=1024, enc_depth=24, enc_num_heads=16`** — the encoder is
  **ViT-Large** (24 layers, width 1024).
- **`dec_embed_dim=768, dec_depth=12, dec_num_heads=12`** — a **Base-sized
  decoder** (12 layers, width 768). Both decoder stacks share this geometry.
- Other `CroCoNet` defaults not overridden here: `patch_size=16`, `mlp_ratio=4`,
  `mask_ratio=0.9` (irrelevant for DUSt3R, no masking is used downstream),
  `norm_layer=LayerNorm(eps=1e-6)`.

This exact ViT-Large-encoder + Base-decoder + RoPE100 combination is what the
published `CroCo_V2_ViTLarge_BaseDecoder.pth` and the 224 DUSt3R checkpoint were
trained with — hence it pairs with the `--pretrained` line below.

> **Adapt for custom dataset:** keep the architecture identical so you can
> warm-start from a published checkpoint — changing any `enc_*`/`dec_*` dim,
> `pos_embed`, or `patch_size` breaks weight loading and forces training from
> scratch. The levers you *may* legitimately touch for a custom run:
> `head_type`/`img_size` (must move together with `resolution` and the matching
> checkpoint — e.g. `dpt` + `512` for the high-res model) and `freeze='encoder'`
> to fine-tune cheaply on limited data.

### Losses

```
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"
--test_criterion  "Regr3D_ScaleShiftInv(L21, gt_scale=True)"
```

- Train: confidence-weighted 3D regression. `Regr3D(L21, norm_mode='avg_dis')`
  is scale-robust pointmap regression; `ConfLoss(..., alpha=0.2)` makes the
  model learn a per-pixel confidence.
- Eval: `Regr3D_ScaleShiftInv` — scale-and-shift-invariant 3D regression, the
  standard DUSt3R eval metric.

> **Adapt for custom dataset:** typically unchanged — these are the standard
> DUSt3R train/eval criteria and are dataset-agnostic.

### Warm start

```
--pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth"
```

Initializes from the CroCo v2 backbone (fetched by `prepare_co3d.sh`). In the
full 3-stage curriculum, later stages point this at the previous stage's
`checkpoint-best.pth` instead.

> **Adapt for custom dataset:** likely point this at a DUSt3R checkpoint
> (e.g. the published 224 model) to fine-tune, rather than the raw CroCo
> backbone — depends on whether you fine-tune or train from the backbone.

### Optimization / schedule

```
--lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10
--batch_size 4 --accum_iter 1 --num_workers 0
```

- Absolute LR `1e-4`, cosine-decayed to `1e-6`, 1 warmup epoch, 10 epochs total
  (demo-sized; real runs are far longer).
- `batch_size 4` per GPU; `accum_iter 1` = no gradient accumulation (raise to
  grow effective batch under tight GPU memory).
- `num_workers 0` loads data in the main process — simple/safe in a container
  (avoids `/dev/shm` issues), at the cost of speed.

> **Adapt for custom dataset:** tune epochs / LR / batch size to the size of the
> new dataset and the GPU budget. Mostly knobs, not structural.

### Checkpointing / eval cadence + output

```
--save_freq 1 --keep_freq 5 --eval_freq 1
--output_dir "checkpoints/dust3r_demo_224"
```

- `eval_freq 1` eval every epoch; `save_freq 1` write `checkpoint-last.pth`
  every epoch; `keep_freq 5` keep permanent `checkpoint-5.pth`, `-10.pth`, …;
  `checkpoint-best.pth` written on eval improvement.
- Outputs land in `checkpoints/dust3r_demo_224/` under the bind-mounted repo, so
  they survive after the container exits.

> **Adapt for custom dataset:** change `--output_dir` so a new run doesn't
> collide with the demo's checkpoints.

## 4. How this differs from a real run

This is the smallest possible config (1000 train pairs, one CO3D sequence,
linear head, 224 res, 10 epochs). The full DUSt3R recipe is a 3-stage curriculum
(224 linear → 512 linear → 512 dpt), each stage warm-started from the previous
stage's `checkpoint-best.pth`, using `--model "...(patch_embed_cls='ManyAR_PatchEmbed')"`
for mixed aspect ratios — see repo-root README "Our Hyperparameters".

## 5. Summary: what changes for a custom dataset

| Part | Changes? |
|------|----------|
| `prepare_co3d.sh` bootstrap | **Yes** — needs a dataset-specific equivalent |
| Custom dataset class (importable from `dust3r/datasets/__init__.py`) | **Yes — new code** |
| `--train_dataset` / `--test_dataset` strings | **Yes** |
| `--output_dir` | **Yes** |
| `--pretrained` | **Likely** (fine-tune vs. from backbone) |
| `--lr` / `--epochs` / `--batch_size` | **Tune** |
| `--model` architecture | Usually no |
| `--train_criterion` / `--test_criterion` | Usually no |
