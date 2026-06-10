# Training on OverMaps (`train_overmaps_entrypoint.sh`)

This document is the OverMaps-specific training guide. The launcher scripts are:

| Local (compose) | Cluster (Slurm / Apptainer) |
|---|---|
| `docker/train_overmaps.sh` | `docker/slurm_train_overmaps.sh` |

Both run `docker/files/train_overmaps_entrypoint.sh` inside the container — edit
that file on the host to change the training config (bind-mounted, no image
rebuild needed).

> **Current state:** `train_overmaps_entrypoint.sh` is a real 224 linear-head
> run on OverMaps, sized after the reference DUSt3R stage-1 recipe (100 epochs,
> 10 warmup, effective batch 16). It warm-starts from the CroCo v2 backbone and
> writes to `checkpoints/dust3r_overmaps_224_full`. The sections below describe
> what each flag does; the original Co3d-smoke-test TODOs are all resolved
> (see the checklist in §5).

## 1. What the entrypoint does

`docker/files/train_overmaps_entrypoint.sh` is executed **inside the training
container** (launched by `docker/train_overmaps.sh` or `slurm_train_overmaps.sh`).
The host repo is bind-mounted at `/dust3r`, so edits take effect on the next run
without rebuilding the image.

Structure of the script:

```bash
set -eu                                   # abort on error / unset var
cd /dust3r                                # repo root (bind-mounted host repo)
/dust3r/docker/files/prepare_overmaps.sh  # one-time prerequisite check/fetch
exec python -u train.py ...               # replace shell with the trainer (PID 1)
```

- `prepare_overmaps.sh` is idempotent: it fetches the CroCo v2 checkpoint into
  `checkpoints/` if missing, and **checks** (does not generate) the preprocessed
  data at `data/overmaps_processed/`, erroring out with the preprocess command if
  it's absent. Preprocessing is a deliberate manual step (needs the raw OverMaps
  data + pycolmap) — run `datasets_preprocess/preprocess_overmaps.py` yourself.
- `python -u` keeps stdout unbuffered so progress streams live into the Slurm
  `.out` log instead of flushing only at epoch boundaries.
- `exec` replaces the shell with Python so signals (Ctrl-C, docker/SLURM stop)
  reach the trainer directly.
- `train.py` is a 3-line shim calling `get_args_parser()` + `train(args)` from
  `dust3r/training.py`.

## 2. The key mechanism: eval()'d expression strings

`--model`, `--train_dataset`, `--test_dataset`, `--train_criterion`,
`--test_criterion` are **Python expression strings that get `eval()`'d** inside
`dust3r/training.py`. Every symbol used must be importable there — that's why
`training.py` does `from dust3r.losses import *`, imports the model, the dataset
classes from `dust3r/datasets/__init__.py`, etc.

The `OverMaps` dataset class (`dust3r/datasets/overmaps.py`) is re-exported from
`dust3r/datasets/__init__.py`, so the name `OverMaps` is valid in the
`--train_dataset` / `--test_dataset` strings.

## 3. Flag-by-flag

### Datasets — `"N @ DatasetCls(args)"`

```
--train_dataset "1680 @ OverMaps(split='train', ROOT='data/overmaps_processed',
                                 aug_crop=16, resolution=224, transform=ColorJitter)"
--test_dataset  "80   @ OverMaps(split='test',  ROOT='data/overmaps_processed',
                                 resolution=224, seed=777)"
```

- `N @` is a per-epoch sample-count weight; multiple datasets can be summed with
  `+`. The counts above (1680 train / 80 test) cover the full preprocessed set —
  there are 1687 train / 81 test pairs — rounded down to multiples of the
  effective batch. Adjust if you reprocess more OverMaps scenes.
- The train/test split is **per-pair**, stored as the 4th field of each entry in
  each scene's `pairs.json`; the `OverMaps` loader keeps only pairs whose tag
  matches `split`. So one preprocessed scene already yields both splits.
- `aug_crop=16` and `transform=ColorJitter` are **train-only** augmentations
  (`OverMaps` has no `mask_bg`, unlike Co3d). Eval uses none and a fixed
  `seed=777` for reproducibility.
- `resolution=224` matches the model `img_size`.

> **Caveat:** the current `data/overmaps_processed/` holds a *single* scene, so
> 100 epochs memorise rather than generalise — this is really an overfit/
> fine-tune experiment until more scenes are preprocessed.

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

Initializes from the CroCo v2 backbone, fetched by `prepare_overmaps.sh` if
missing. In the full 3-stage curriculum, later stages point this at the previous
stage's `checkpoint-best.pth` instead.

> **Auto-resume gotcha:** `train.py` looks for `checkpoint-last.pth` in
> `--output_dir` and, if found, resumes from it and **ignores `--pretrained`**.
> So this CroCo warm-start only happens when the output dir is empty — see the
> note in the checkpointing section below.

### Optimization / schedule

```
--lr 0.0001 --min_lr 1e-06 --warmup_epochs 10 --epochs 100
--batch_size 4 --accum_iter 4 --num_workers 8
```

- Absolute LR `1e-4`, cosine-decayed to `1e-6`, 10 warmup epochs, 100 epochs
  total — matching the reference DUSt3R stage-1 (224 linear) schedule.
- **Effective batch 16 = `batch_size 4` × `accum_iter 4`.** The reference uses
  `batch_size 16 / accum_iter 1`, but a full ViT-Large fine-tune at bs16 OOMs a
  32 GB V100; splitting it into 4 micro-batches keeps the optimizer identical
  while peaking ~14 GB, so it fits whatever GPU the scheduler hands the job. On a
  bigger card (A100/H100/H200) raise `batch_size` and drop `accum_iter` to go
  faster.
- `num_workers 8` matches the 8 CPUs the Slurm job requests; with `num_workers 0`
  the data pipeline runs in the main process and starves a fast GPU (it was the
  bottleneck — the GPU sat idle waiting on serial image decode/crop).

> Tune epochs / LR / batch size as the OverMaps dataset grows; these are knobs,
> not structural. The architecture and losses below stay fixed to keep the
> CroCo warm-start valid.

### Checkpointing / eval cadence + output

```
--save_freq 1 --keep_freq 20 --eval_freq 1
--output_dir "checkpoints/dust3r_overmaps_224_full"
```

- `eval_freq 1` eval every epoch; `save_freq 1` write `checkpoint-last.pth`
  every epoch; `keep_freq 20` keep permanent `checkpoint-20.pth`, `-40.pth`, …
  (each ~6 GB, so `keep_freq 20` over 100 epochs keeps ~5 of them rather than the
  20 that `keep_freq 5` would hoard); `checkpoint-best.pth` written on eval
  improvement.
- Outputs land in `checkpoints/dust3r_overmaps_224_full/` under the bind-mounted
  repo, so they survive after the container exits.

> **Use a fresh `--output_dir` per run.** `train.py` auto-resumes from a
> `checkpoint-last.pth` found in the dir (and then ignores `--pretrained` — see
> Warm start). Pointing a new run at a previous run's dir silently *continues*
> that run with its old optimizer state instead of warm-starting clean from
> CroCo. This is also handy on purpose: bump `TIME`/`--epochs` and resubmit to
> resume an interrupted run from where it stopped.

## 4. Scale: this run vs. the full curriculum

The entrypoint now runs **stage 1 only** (224 linear, 100 epochs). The full
DUSt3R recipe is a 3-stage curriculum (224 linear → 512 linear → 512 dpt), each
stage warm-started from the previous stage's `checkpoint-best.pth`, using
`--model "...(patch_embed_cls='ManyAR_PatchEmbed')"` for mixed aspect ratios and
`head_type='dpt'` + `resolution=512` for the high-res stages — see repo-root
README "Our Hyperparameters". To run the later stages, copy this entrypoint,
swap `head_type`/`resolution`/`img_size` and point `--pretrained` at the prior
stage's checkpoint. The main bottleneck to a *general* model is data: preprocess
more OverMaps scenes (a single scene overfits — see §3 caveat).

## 5. OverMaps setup checklist

| Part | Status |
|------|--------|
| `OverMaps` dataset class in `dust3r/datasets/overmaps.py` | ✅ done |
| Export from `dust3r/datasets/__init__.py` | ✅ done |
| Bootstrap script `prepare_overmaps.sh` (checks data, fetches CroCo) | ✅ done |
| `prepare_overmaps.sh` wired into the entrypoint | ✅ done |
| `--train_dataset` / `--test_dataset` strings → `OverMaps(...)` | ✅ done |
| `--output_dir` → `checkpoints/dust3r_overmaps_224_full` | ✅ done |
| `--pretrained` → CroCo v2 backbone | ✅ done |
| `--lr` / `--epochs` / `--batch_size` → reference stage-1 (bs16 via accum) | ✅ done |
| `--model` architecture | unchanged (keeps warm-start valid) |
| `--train_criterion` / `--test_criterion` | unchanged |
| Preprocess more scenes / run 512 stages | ⬜ next, when needed |
