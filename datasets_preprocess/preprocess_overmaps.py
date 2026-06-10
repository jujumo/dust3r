#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the OverMaps dataset.
#
# Raw layout (input):
#   <overmaps_dir>/sparse/<scene>/0/{cameras,images,points3D}.bin   COLMAP model
#   <overmaps_dir>/images/<scene>/<stem>.jpg                        RGB (portrait)
#   <overmaps_dir>/depths/<scene>/<stem>_DS.exr                     metric depth (Y, landscape)
#   <overmaps_dir>/masks_images/<scene>/<stem>.png                  binary mask (0/255)
#
# Processed layout (output) -- intentionally human-readable, no monolithic blob:
#   <output_dir>/<scene>/images/<stem>.jpg     coarse-cropped RGB
#   <output_dir>/<scene>/depth/<stem>.png      uint16 depth in MILLIMETERS, 0 = invalid
#   <output_dir>/<scene>/poses/<stem>.json     per-image intrinsics + cam2world
#   <output_dir>/<scene>/pairs.json            co-visibility pairs (by stem) + overlap + split
#
# Usage:
#   python3 datasets_preprocess/preprocess_overmaps.py \
#       --overmaps_dir data/OverMaps-1K --output_dir data/overmaps_processed
# --------------------------------------------------------

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # must precede cv2 import (EXR depth)

import argparse
import csv
import itertools
import json
import os.path as osp
from collections import defaultdict

import numpy as np
import cv2
import PIL.Image
import pycolmap
from tqdm import tqdm

import path_to_root  # noqa: F401  (puts the repo root on sys.path)
import dust3r.datasets.utils.cropping as cropping  # noqa
import dust3r.utils.geometry as geometry  # noqa
from dust3r.utils.image import imread_cv2  # noqa


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overmaps_dir", required=True,
                        help="raw OverMaps root (contains sparse/, images/, depths/, masks_images/)")
    parser.add_argument("--output_dir", default="data/overmaps_processed")
    parser.add_argument("--manifest", default=None,
                        help="dataset_manifest.csv (default: <overmaps_dir>/dataset_manifest.csv). "
                             "Scenes are filtered to those with lidar depth, i.e. a non-empty "
                             "depths_path. If no manifest is found, every scene under sparse/ is used.")
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess a scene even if its output pairs.json already exists "
                             "(default: skip done scenes, so the job is resumable)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="lower dimension will be >= img_size*3/4, max dimension >= img_size")
    parser.add_argument("--far_thresh", type=float, default=24.0,
                        help="depth beyond this many meters is treated as invalid")
    parser.add_argument("--min_overlap", type=float, default=0.10,
                        help="minimum co-visibility overlap ratio for a training pair")
    parser.add_argument("--max_overlap", type=float, default=0.90,
                        help="maximum co-visibility overlap ratio for a training pair")
    parser.add_argument("--top_k", type=int, default=10,
                        help="keep at most this many co-visible neighbours per image")
    parser.add_argument("--test_frac", type=float, default=0.05,
                        help="fraction of pairs held out for the 'test' split")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ------------------------------------------------------------------ COLMAP --

def _camera_model_name(cam):
    # pycolmap 4.x exposes cam.model_name; 3.x uses cam.model.name
    return getattr(cam, "model_name", None) or cam.model.name


def intrinsics_from_camera(cam):
    """COLMAP PINHOLE camera -> OpenCV 3x3 K (float64)."""
    name = _camera_model_name(cam)
    assert name == "PINHOLE", f"only PINHOLE cameras are supported, got {name}"
    fx, fy, cx, cy = cam.params  # PINHOLE order
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return geometry.colmap_to_opencv_intrinsics(K)


def cam2world_from_image(img):
    """COLMAP image -> 4x4 cam2world (float64). cam_from_world is world->cam [R|t]."""
    # pycolmap 4.x: cam_from_world is a method; 3.x: a property. Both -> Rigid3d.
    cfw = img.cam_from_world
    if callable(cfw):
        cfw = cfw()
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = np.asarray(cfw.matrix(), dtype=np.float64)
    return np.linalg.inv(w2c)


def compute_covisibility(rec, min_overlap, max_overlap, top_k):
    """Track-based co-visibility. Returns dict {(img_id_a<img_id_b): overlap}."""
    # observed-point count per image
    obs_count = {iid: sum(1 for p2d in im.points2D if p2d.has_point3D())
                 for iid, im in rec.images.items()}

    # accumulate shared-point counts over 3D point tracks (cheap vs. O(N^2))
    covis = defaultdict(int)
    for p3 in rec.points3D.values():
        iids = sorted({el.image_id for el in p3.track.elements})
        for a, b in itertools.combinations(iids, 2):
            covis[(a, b)] += 1  # a < b because iids is sorted

    # overlap ratio + band filter
    pair_ov = {}
    for (a, b), c in covis.items():
        denom = min(obs_count[a], obs_count[b])
        if denom == 0:
            continue
        ov = c / denom
        if min_overlap <= ov <= max_overlap:
            pair_ov[(a, b)] = ov

    # cap at top_k neighbours per image (by descending overlap), then dedupe
    adj = defaultdict(list)
    for (a, b), ov in pair_ov.items():
        adj[a].append((ov, b))
        adj[b].append((ov, a))
    kept = set()
    for i, lst in adj.items():
        for ov, j in sorted(lst, reverse=True)[:top_k]:
            kept.add((min(i, j), max(i, j)))
    return {pair: pair_ov[pair] for pair in kept}


# --------------------------------------------------------------- per-frame --

def process_frame(stem, K, cam2world, overmaps_dir, scene, scene_out, img_size, far_thresh):
    """Load + align + crop one frame; write jpg/png/json. Returns True on success."""
    impath = osp.join(overmaps_dir, "images", scene, stem + ".jpg")
    depthpath = osp.join(overmaps_dir, "depths", scene, stem + "_DS.exr")
    maskpath = osp.join(overmaps_dir, "masks_images", scene, stem + ".png")
    if not (osp.isfile(impath) and osp.isfile(depthpath) and osp.isfile(maskpath)):
        return False

    rgb = PIL.Image.open(impath).convert("RGB")
    W, H = rgb.size

    # depth: landscape EXR -> rot90 CCW -> portrait -> upscale (NEAREST) to RGB size
    depth = imread_cv2(depthpath)                       # float32, (144, 256)
    depth = np.ascontiguousarray(np.rot90(depth, 1))    # CCW -> (256, 144)
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # mask: binary, at RGB resolution
    mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[..., 0]

    # validity = finite & positive & not-too-far & inside mask  -> invalid set to 0
    valid = np.isfinite(depth) & (depth > 0) & (depth <= far_thresh) & (mask > 127)
    depth = np.where(valid, depth, 0.0).astype(np.float32)

    K = K.astype(np.float64)
    # coarse principal-point-centered crop (keeps cx,cy centered)
    cx, cy = K[:2, 2].round().astype(int)
    mx, my = min(cx, W - cx), min(cy, H - cy)
    rgb, depth, K = cropping.crop_image_depthmap(
        rgb, depth, K, (cx - mx, cy - my, cx + mx, cy + my))

    # rescale so the lower dim is ~ img_size*3/4 (mirror preprocess_co3d.py)
    W, H = rgb.size
    scale = ((img_size * 3 // 4) / min(H, W)) + 1e-8
    out_res = np.floor(np.array([W, H]) * scale).astype(int)
    if max(out_res) < img_size:
        scale = (img_size / max(H, W)) + 1e-8
        out_res = np.floor(np.array([W, H]) * scale).astype(int)
    rgb, depth, K = cropping.rescale_image_depthmap(rgb, depth, K, out_res)

    # depth -> uint16 millimeters (0 stays invalid)
    depth_mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)

    rgb.save(osp.join(scene_out, "images", stem + ".jpg"), quality=95)
    cv2.imwrite(osp.join(scene_out, "depth", stem + ".png"), depth_mm)
    with open(osp.join(scene_out, "poses", stem + ".json"), "w") as f:
        json.dump(dict(
            image=f"images/{stem}.jpg",
            depth=f"depth/{stem}.png",
            width=int(rgb.size[0]), height=int(rgb.size[1]),
            depth_unit_to_meters=0.001,
            camera_model="PINHOLE",
            intrinsics=K.astype(np.float64).tolist(),
            cam2world=cam2world.astype(np.float64).tolist(),
        ), f, indent=2)
    return True


# -------------------------------------------------------------------- main --

def process_scene(scene, overmaps_dir, output_dir, args, rng):
    rec = pycolmap.Reconstruction(osp.join(overmaps_dir, "sparse", scene, "0"))

    scene_out = osp.join(output_dir, scene)
    for sub in ("images", "depth", "poses"):
        os.makedirs(osp.join(scene_out, sub), exist_ok=True)

    # per-frame export
    id2stem = {iid: osp.splitext(im.name)[0] for iid, im in rec.images.items()}
    saved = set()
    for iid, im in tqdm(sorted(rec.images.items()), desc=scene, leave=False):
        cam = rec.cameras[im.camera_id]
        stem = id2stem[iid]
        ok = process_frame(stem, intrinsics_from_camera(cam), cam2world_from_image(im),
                            overmaps_dir, scene, scene_out, args.img_size, args.far_thresh)
        if ok:
            saved.add(iid)

    # co-visibility pairs, restricted to successfully-exported frames
    pair_ov = compute_covisibility(rec, args.min_overlap, args.max_overlap, args.top_k)
    pairs = []
    n_test = 0
    for (a, b), ov in sorted(pair_ov.items()):
        if a not in saved or b not in saved:
            continue
        split = "test" if rng.random() < args.test_frac else "train"
        n_test += split == "test"
        pairs.append([id2stem[a], id2stem[b], round(float(ov), 4), split])
    with open(osp.join(scene_out, "pairs.json"), "w") as f:
        json.dump(pairs, f, indent=1)

    print(f"  {scene}: {len(saved)} frames, {len(pairs)} pairs ({n_test} test)")
    return len(saved), len(pairs)


def list_lidar_scenes(args):
    """Scene ids to process: those with lidar depth (non-empty depths_path in the
    manifest), or, if no manifest is found, every scene under sparse/."""
    manifest = args.manifest or osp.join(args.overmaps_dir, "dataset_manifest.csv")
    if osp.isfile(manifest):
        scenes = []
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("depths_path", "").strip():  # has lidar depth
                    scenes.append(row["mapping_id"])
        print(f"manifest {manifest}: {len(scenes)} scene(s) with lidar depth")
        return sorted(scenes)
    sparse_root = osp.join(args.overmaps_dir, "sparse")
    scenes = sorted(d for d in os.listdir(sparse_root)
                    if osp.isdir(osp.join(sparse_root, d)))
    print(f"no manifest; using all {len(scenes)} scene(s) under {sparse_root}")
    return scenes


def scene_inputs_ready(overmaps_dir, scene):
    """True iff every raw modality the preprocessor reads is materialized on disk."""
    needed = [osp.join("sparse", scene, "0"), osp.join("images", scene),
              osp.join("depths", scene), osp.join("masks_images", scene)]
    return all(osp.isdir(osp.join(overmaps_dir, p)) for p in needed)


def main():
    args = get_parser().parse_args()
    assert args.overmaps_dir != args.output_dir
    scenes = list_lidar_scenes(args)
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tot_frames = tot_pairs = 0
    n_done = n_skipped_existing = n_skipped_missing = 0
    for scene in scenes:
        if not args.overwrite and osp.isfile(osp.join(args.output_dir, scene, "pairs.json")):
            n_skipped_existing += 1
            continue
        if not scene_inputs_ready(args.overmaps_dir, scene):
            n_skipped_missing += 1
            continue
        nf, npairs = process_scene(scene, args.overmaps_dir, args.output_dir, args, rng)
        tot_frames += nf
        tot_pairs += npairs
        n_done += 1
    print(f"DONE: processed {n_done} scene(s) ({tot_frames} frames, {tot_pairs} pairs); "
          f"skipped {n_skipped_existing} already-done, {n_skipped_missing} not-yet-materialized "
          f"(missing images/depths/masks/sparse) -> {args.output_dir}")
    if n_skipped_missing:
        print(f"NOTE: {n_skipped_missing} lidar scene(s) lack materialized inputs (e.g. images/); "
              f"re-run this command after materializing them to pick them up incrementally.")


if __name__ == "__main__":
    main()
