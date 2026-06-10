#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Headless visual check for the OverMaps loader. Writes files you can open in
# any image / 3D viewer (no live OpenGL window needed -- works over ssh):
#
#   <out>_montage.png   per view: RGB | depth (turbo, invalid=black) | valid mask
#   <out>_cloud.glb     both views' point clouds in WORLD frame, RGB-colored;
#                       if depth+pose+intrinsics are right, they overlap.
#
# Usage (from repo root, in the dust3r conda env):
#   python datasets_preprocess/viz_overmaps.py --index 0
#   python datasets_preprocess/viz_overmaps.py --split test --random --out /tmp/om
# --------------------------------------------------------
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import trimesh

import path_to_root  # noqa: F401
from dust3r.datasets import OverMaps
from dust3r.utils.image import rgb


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/overmaps_processed")
    p.add_argument("--split", default="train", choices=["train", "test", "val"])
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--index", type=int, default=0, help="pair index (ignored with --random)")
    p.add_argument("--random", action="store_true")
    p.add_argument("--out", default="/tmp/overmaps_viz")
    return p


def main():
    args = get_parser().parse_args()
    dataset = OverMaps(split=args.split, ROOT=args.root, resolution=args.resolution)
    idx = np.random.randint(len(dataset)) if args.random else args.index
    views = dataset[idx]
    print(f"[{args.split}] idx={idx}/{len(dataset)}  {views[0]['label']}  <->  {views[1]['label']}")

    # ---- 2D montage: RGB | depth | valid mask, one row per view ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r, v in enumerate(views):
        img = rgb(v["img"])
        dm = v["depthmap"]
        vm = v["valid_mask"]
        dvis = np.ma.masked_where(dm <= 0, dm)  # invalid -> black
        axes[r, 0].imshow(img);                         axes[r, 0].set_title(f"{v['label']}  RGB")
        im = axes[r, 1].imshow(dvis, cmap="turbo");     axes[r, 1].set_title("depth (m)")
        fig.colorbar(im, ax=axes[r, 1], fraction=0.046)
        axes[r, 2].imshow(vm, cmap="gray");             axes[r, 2].set_title(f"valid ({vm.mean():.1%})")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    montage = args.out + "_montage.png"
    fig.savefig(montage, dpi=110)
    print("wrote", montage)

    # ---- 3D: both world-frame point clouds in one GLB ----
    scene = trimesh.Scene()
    for v in views:
        pts = v["pts3d"][v["valid_mask"]]                       # (N,3) world coords
        col = (rgb(v["img"])[v["valid_mask"]] * 255).astype(np.uint8)
        scene.add_geometry(trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3)))
    glb = args.out + "_cloud.glb"
    scene.export(glb)
    print("wrote", glb, "(open in MeshLab / CloudCompare / VS Code 3D viewer)")


if __name__ == "__main__":
    main()
