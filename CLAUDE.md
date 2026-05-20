# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is the official implementation of **DUSt3R: Geometric 3D Vision Made Easy** (CVPR 2024). It takes a set of images of a scene and outputs aligned 3D pointmaps + camera poses, without needing camera intrinsics. The model is an asymmetric Siamese ViT (`AsymmetricCroCo3DStereo`) built on top of the CroCo v2 backbone, which lives in the `croco/` git submodule.

The CroCo submodule is a hard dependency: `dust3r/utils/path_to_croco.py` injects `croco/` onto `sys.path` so that `from models.croco import CroCoNet` works. If you cloned without `--recursive`, run `git submodule update --init --recursive` first — model code will fail to import otherwise.

There is **no test suite, linter config, or build system** (no `pyproject.toml`, no `setup.py` at the repo root, no `pytest`). The only build step is the optional CUDA extension under `croco/models/curope/`.

## Common commands

### Setup
```bash
# from a fresh clone:
git submodule update --init --recursive
pip install -r requirements.txt
pip install -r requirements_optional.txt  # heif images, pyrender, kapture/poselib for visloc

# optional: compile RoPE CUDA kernels for faster inference
cd croco/models/curope/ && python setup.py build_ext --inplace && cd ../../../
```

### Run the Gradio demo
```bash
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt
# --weights <path>      load a local .pth instead of HF hub
# --image_size 224|512  must match the checkpoint
# --local_network       bind 0.0.0.0 instead of localhost
# --device cpu          override default cuda
```

Docker variant: `cd docker && bash demo.sh --with-cuda --model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"` (drops `--with-cuda` for CPU). The container launches `demo.py --local_network`; UI is on port 7860.

### Training
Multi-GPU training is launched with `torchrun`. The full DUSt3R training is a 3-stage curriculum: 224 linear → 512 linear → 512 dpt, each warm-started from the previous stage's `checkpoint-best.pth` (the first stage starts from the CroCo v2 checkpoint). See README.md "Our Hyperparameters" for the exact commands and the "Demo" section for a short CO3D-subset smoke run.

The `--model`, `--train_dataset`, `--test_dataset`, `--train_criterion`, `--test_criterion` flags all take **Python expression strings** that get `eval()`'d. For datasets the syntax is `"N @ DatasetCls(args)"` (sample-count weighting) combined with `+`, e.g. `"100_000 @ Co3d(split='train', resolution=224) + 100_000 @ ScanNetpp(split='train', resolution=224)"`. Available dataset classes are re-exported from `dust3r/datasets/__init__.py`. All loss classes are exposed via `from dust3r.losses import *` inside `training.py`, so any name there is valid in `--train_criterion`.

### Visual localization
```bash
python3 visloc.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt \
  --dataset "VislocAachenDayNight('/path/to/data', subscene='day', pairsfile='fire_top50', topk=20)" \
  --pnp_mode poselib --reprojection_error_diag_ratio 0.008 \
  --output_dir /path/to/output
```
See `dust3r_visloc/README.md` for the per-dataset directory layouts (Aachen-Day-Night, InLoc, 7-Scenes, Cambridge Landmarks) and exact commands.

### Dataset preprocessing
Each supported dataset has its own preprocessor under `datasets_preprocess/` (e.g. `preprocess_co3d.py`, `preprocess_scannetpp.py`). You must download raw datasets + the published pair files yourself (links in README) and then run the corresponding script to produce the directory layout the training `Dataset` classes expect.

## Architecture

### Model: `dust3r/model.py`
`AsymmetricCroCo3DStereo` extends `CroCoNet` (from the croco submodule) with:
- a **second decoder stack** (`dec_blocks2 = deepcopy(self.dec_blocks)`) so view1 and view2 are decoded asymmetrically — both outputs live in **view1's coordinate frame**, hence the name;
- a **downstream head** (`linear` or `dpt`) wired in by `set_downstream_head` and built via `dust3r/heads/__init__.py::head_factory`; the head predicts `pts3d` + a confidence map;
- a configurable patch embedder (`PatchEmbedDust3R` for fixed shape, `ManyAR_PatchEmbed` for mixed aspect ratios during training) selected by `patch_embed_cls`.

`load_model()` does some checkpoint-string rewriting: it replaces `ManyAR_PatchEmbed` with `PatchEmbedDust3R` at inference time and forces `landscape_only=False`. When loading via HuggingFace Hub (`from_pretrained("naver/...")`), the model auto-downloads.

### Inference path
1. `dust3r.utils.image.load_images(paths_or_dir, size=224|512)` → list of view dicts.
2. `dust3r.image_pairs.make_pairs(images, scene_graph, symmetrize=True)` → list of `(view1, view2)` pairs. When `symmetrize=True`, both `(A,B)` and `(B,A)` are included, doubling the batch.
3. `dust3r.inference.inference(pairs, model, device, batch_size)` → dict with `view1`, `view2`, `pred1`, `pred2`. `pred1['pts3d']` is in view1 space; `pred2['pts3d_in_other_view']` is also in view1 space.
4. (Optional) `dust3r.cloud_opt.global_aligner(output, device, mode=...)` — produces a `scene` object exposing `get_focals()`, `get_im_poses()`, `get_pts3d()`, `get_masks()`, and `show()`.

### Global alignment: `dust3r/cloud_opt/`
Three modes selected via `GlobalAlignerMode`:
- `PointCloudOptimizer` (default) — full optimization over per-image poses, focals, and depthmaps. Call `scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)` before reading results.
- `ModularPointCloudOptimizer` — variant with more knobs for partial optimization.
- `PairViewer` — no optimization; just unpacks the raw pairwise prediction. Use this when you have exactly 2 images (the demo auto-selects this for ≤2 images).

`base_opt.BasePCOptimizer` holds the shared optimization scaffolding; `init_im_poses.py` provides the MST-based initialization.

### Training pipeline: `dust3r/training.py` and `dust3r/losses.py`
`training.train(args)` is a fairly standard MAE/DeiT-style loop, but be aware:
- `--model` / dataset / loss strings are passed through `eval()`, so every symbol used in them must be imported (or wildcard-imported) inside `training.py`. That's why it does `from dust3r.losses import *` and `from dust3r.model import AsymmetricCroCo3DStereo, inf`.
- Standard loss: `ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)` — confidence-weighted 3D regression. Eval loss is typically `Regr3D_ScaleShiftInv(L21, gt_scale=True)`.
- Distributed init goes through `croco.utils.misc.init_distributed_mode`. `--accum_iter` increases effective batch size when GPU memory is tight.
- Checkpointing produces `checkpoint-last.pth` (every `--save_freq`), `checkpoint-N.pth` (every `--keep_freq`), and `checkpoint-best.pth` (best eval). Multi-stage training warm-starts from these.

### Heads: `dust3r/heads/`
- `linear_head.py::LinearPts3d` — single linear projection from decoder tokens to a `H×W×(3+1)` map (xyz + confidence). Light-weight; used by the 224 model.
- `dpt_head.py::PixelwiseTaskWithDPT` — dense DPT decoder for higher-resolution 512 checkpoints. Heavier and slower.
- `postprocess.py::postprocess` — applies the `depth_mode` (`exp`/`sigmoid`/`linear`) and `conf_mode` activations on raw head outputs.

### Datasets: `dust3r/datasets/`
All training datasets inherit from `base/base_stereo_view_dataset.BaseStereoViewDataset`, which yields pairs of view dicts with `img`, `depthmap`, `camera_pose`, `camera_intrinsics`, etc. `base/batched_sampler.BatchedRandomSampler` groups same-resolution samples into a batch (needed because `ManyAR_PatchEmbed` requires homogeneous shapes within a batch). `utils/cropping.py` handles aspect-ratio-aware crops; `utils/transforms.py` defines `ColorJitter` etc. that show up in `--train_dataset` strings.

## Conventions and gotchas

- **CroCo paths**: any file that uses croco code must `import dust3r.utils.path_to_croco` first (often with `# noqa: F401`). Don't remove these — they look unused but they mutate `sys.path`.
- **Symmetrize doubles work**: with `symmetrize=True`, batches are `2N`. Memory accounting in scripts assumes this.
- **`landscape_only`**: training-time flag; at inference, `load_model` forces it off. Don't toggle it on for inference.
- **HF Hub model IDs** are `naver/<Modelname>` (e.g. `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`); the bare `<Modelname>` form in `--model_name` is prefixed automatically in `demo.py` / `visloc.py`.
- **No CI / no tests**: changes are validated by running the demo / a short training smoke run. Don't claim a refactor "passes tests" — there aren't any.

## License

Code is CC BY-NC-SA 4.0 (non-commercial). The standard Naver copyright header appears at the top of every file; preserve it when editing.
