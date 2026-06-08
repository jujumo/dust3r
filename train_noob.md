# Training on the Slurm cluster + what to watch in TensorBoard

## Launching the training on Slurm

```bash
bash docker/slurm_train_Co3d.sh
# override resources via env vars, e.g.:
PARTITION=gpu GPUS=2 TIME=4:00:00 bash docker/slurm_train_Co3d.sh
```

It `sbatch`es an `apptainer exec --nv` job that runs
`docker/files/train_Co3d_entrypoint.sh`. That entrypoint writes checkpoints **and**
the TensorBoard event files into `args.output_dir`, which is currently
`checkpoints/dust3r_demo_224` (set on the last line of `train_Co3d_entrypoint.sh`).
Because the host repo is bind-mounted into the container, those event files
land on your NFS home and are visible from anywhere.

## Watching the curves

TensorBoard runs in its own container, pointed at the same dir:

```bash
bash docker/tensorboard.sh                 # serves checkpoints/ on :6006
# from your laptop: ssh -fNL 6006:localhost:6006 chaos-09
```

`--logdir` defaults to `/dust3r/checkpoints`, so it picks up `dust3r_demo_224`
(and any other output_dir) automatically, reloading every 5s as the job writes.

## What you can actually monitor

The logging lives in `dust3r/training.py`. The x-axis is `epoch * 1000`
("epoch_1000x") so curves stay comparable if you change batch size.

### Training scalars (`train_one_epoch`, written every `--print_freq` iters)

| Tag | Meaning |
|---|---|
| `train_loss` | total training loss, the value being optimized — this is `ConfLoss = conf_loss1 + conf_loss2` (`losses.py:238`), DDP-reduced across ranks |
| `train_lr` | learning rate — you'll see the 1-epoch warmup ramp then the cosine decay toward `--min_lr` |
| `train_iter` | just epoch_1000x plotted against itself — a sanity diagonal, ignore it |
| `train_conf_loss_1` / `train_conf_loss2` | the two halves of the confidence-weighted loss (view1 and view2 sides) |
| `train_Regr3D_pts3d_1` / `train_Regr3D_pts3d_2` | the **raw** L21 3D-regression error per view, *before* confidence weighting — these bubble up from `Regr3D` (`losses.py:193`). Most interpretable "how good are the pointmaps" signal during training |

### Eval scalars (`test_one_epoch`, once per `--eval_freq` epoch)

Prefixed by the test-dataset name (e.g. `Co3d_…`). The eval criterion is
`Regr3D_ScaleShiftInv(L21, gt_scale=True)`, and each metric is logged with both
an average and a median aggregation:

| Tag pattern | Meaning |
|---|---|
| `<DS>_loss_avg` / `<DS>_loss_med` | scale/shift-invariant 3D regression error on the test set |
| `<DS>_Regr3D_ScaleShiftInv_pts3d_1_{avg,med}` | per-view (view1) eval error |
| `<DS>_Regr3D_ScaleShiftInv_pts3d_2_{avg,med}` | per-view (view2) eval error |

The `loss_med` (median) is what drives `checkpoint-best.pth` selection
(`training.py:210`), so that's the curve to watch for "is this run improving."

## Practical notes

- Only the **All Scalars** tab is populated — there are no image/histogram/graph
  summaries logged, so don't expect to see pointmaps or weight distributions in
  TensorBoard.
- With the default smoke config (`--epochs 10`, `1000 @ Co3d` train / `100 @ Co3d`
  test, single GPU since `train_Co3d_entrypoint.sh` runs a plain `python train.py` not
  `torchrun`), the curves are short. For a real run you'd edit the entrypoint's
  dataset/epoch flags (and switch to `torchrun --nproc_per_node=$GPUS` for
  multi-GPU, per the note in `slurm_train_Co3d.sh:34`).
- Healthy run = `train_loss` and the `Regr3D` curves trending down, `train_lr`
  showing warmup-then-cosine. If `train_loss` goes non-finite the job exits
  immediately (`training.py:304`) — you'll see it in
  `slurm-logs/dust3r-train-<jobid>.out` rather than TensorBoard.
