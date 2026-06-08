# Training on OverMaps (`train_overmaps_entrypoint.sh`)

This document is the OverMaps-specific training guide. The launcher scripts are:

| Local (compose) | Cluster (Slurm / Apptainer) |
|---|---|
| `docker/train_overmaps.sh` | `docker/slurm_train_overmaps.sh` |

Both run `docker/files/train_overmaps_entrypoint.sh` inside the container — edit
that file on the host to change the training config (bind-mounted, no image
rebuild needed).

> **Current state:** `train_overmaps_entrypoint.sh` is a copy of the Co3d
> smoke-test. The sections below describe what each part does and flag what
> still needs to be replaced for OverMaps (marked **TODO**).

## 1. What the entrypoint does

`docker/files/train_overmaps_entrypoint.sh` is executed **inside the training
container** (launched by `docker/train_overmaps.sh` or `slurm_train_overmaps.sh`).
The host repo is bind-mounted at `/dust3r`, so edits take effect on the next run
without rebuilding the image.

Structure of the script:

```bash
set -eu                               # abort on error / unset var
cd /dust3r                            # repo root (bind-mounted host repo)
/dust3r/docker/files/prepare_co3d.sh  # one-time bootstrap of prerequisites  ← TODO
exec python train.py ...              # replace shell with the trainer (PID 1)
```

- The bootstrap script is idempotent: it downloads/preprocesses data and fetches
  the CroCo v2 checkpoint into `checkpoints/`. Both persist on the host and are
  skipped on later runs.
- `exec` replaces the shell with Python so signals (Ctrl-C, docker/SLURM stop)
  reach the trainer directly.
- `train.py` is a 3-line shim calling `get_args_parser()` + `train(args)` from
  `dust3r/training.py`.

> **TODO:** replace `prepare_co3d.sh` with an OverMaps-specific bootstrap
> (or prepare the data manually and remove the call). The bootstrap must place
> data in the layout the `OverMaps` dataset class expects and ensure the
> `--pretrained` checkpoint is present.

## 2. The key mechanism: eval()'d expression strings

`--model`, `--train_dataset`, `--test_dataset`, `--train_criterion`,
`--test_criterion` are **Python expression strings that get `eval()`'d** inside
`dust3r/training.py`. Every symbol used must be importable there — that's why
`training.py` does `from dust3r.losses import *`, imports the model, the dataset
classes from `dust3r/datasets/__init__.py`, etc.

> **TODO:** the `OverMaps` dataset class must be importable from
> `dust3r/datasets/__init__.py` for its name to be usable in `--train_dataset`.

## 3. Flag-by-flag

### Datasets — `"N @ DatasetCls(args)"`

Current placeholder (Co3d, to be replaced):

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

> **TODO:** replace `Co3d(...)` with `OverMaps(...)`, point `ROOT` at the
> preprocessed data directory, set appropriate sample counts, and decide which
> augmentations apply.

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

> Keep the architecture identical to warm-start from a published checkpoint —
> changing any `enc_*`/`dec_*` dim, `pos_embed`, or `patch_size` breaks weight
> loading and forces training from scratch. Legitimate levers for OverMaps:
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

> These are dataset-agnostic — no changes needed for OverMaps.

### Warm start

```
--pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth"
```

Initializes from the CroCo v2 backbone (currently fetched by the placeholder
`prepare_co3d.sh`). In the full 3-stage curriculum, later stages point this at
the previous stage's `checkpoint-best.pth` instead.

> **TODO:** decide whether to fine-tune from the published DUSt3R 224 checkpoint
> or warm-start from the raw CroCo v2 backbone, and update this path accordingly.
> The bootstrap script must ensure the chosen checkpoint is present.

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

> **TODO:** tune epochs / LR / batch size once the size of OverMaps and the
> GPU budget are known. Mostly knobs, not structural.

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

> **TODO:** change `--output_dir` to something OverMaps-specific (e.g.
> `checkpoints/dust3r_overmaps_224`) so it doesn't collide with the Co3d run.

## 4. Scale: current placeholder vs. a real OverMaps run

The current entrypoint is a minimal smoke-test config (1000 train pairs, 10
epochs, linear head, 224 res). A real OverMaps run will need more pairs and
more epochs; the full DUSt3R recipe is a 3-stage curriculum
(224 linear → 512 linear → 512 dpt), each stage warm-started from the previous
stage's `checkpoint-best.pth`, using `--model "...(patch_embed_cls='ManyAR_PatchEmbed')"`
for mixed aspect ratios — see repo-root README "Our Hyperparameters".

## 5. OverMaps TODO checklist

| Part | Status |
|------|--------|
| `OverMaps` dataset class in `dust3r/datasets/` | **TODO — new code** |
| Export from `dust3r/datasets/__init__.py` | **TODO** |
| Bootstrap script (`prepare_overmaps.sh` or manual data prep) | **TODO** |
| Replace `prepare_co3d.sh` call in entrypoint | **TODO** |
| `--train_dataset` / `--test_dataset` strings → `OverMaps(...)` | **TODO** |
| `--output_dir` → `checkpoints/dust3r_overmaps_224` | **TODO** |
| `--pretrained` → DUSt3R 224 checkpoint (or CroCo backbone) | **TODO** |
| `--lr` / `--epochs` / `--batch_size` tuning | tune once data size is known |
| `--model` architecture | keep as-is |
| `--train_criterion` / `--test_criterion` | keep as-is |
