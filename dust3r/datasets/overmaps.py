# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for the preprocessed OverMaps dataset.
# See datasets_preprocess/preprocess_overmaps.py for the on-disk layout:
#   <ROOT>/<scene>/images/<stem>.jpg     coarse-cropped RGB
#   <ROOT>/<scene>/depth/<stem>.png      uint16 depth in mm, 0 = invalid
#   <ROOT>/<scene>/poses/<stem>.json     per-image intrinsics + cam2world
#   <ROOT>/<scene>/pairs.json            [[stem_i, stem_j, overlap, split], ...]
# --------------------------------------------------------
import os
import os.path as osp
import json
from collections import deque

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class OverMaps(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_label = 'OverMaps'
        self._load_data()

    def _load_data(self):
        # which pairs split to keep
        if self.split is None:
            keep = None
        elif self.split == 'train':
            keep = {'train'}
        elif self.split in ('test', 'val'):
            keep = {'test'}
        else:
            raise ValueError(f"unknown split {self.split!r} (expected train/test/val/None)")

        scenes = sorted(d for d in os.listdir(self.ROOT)
                        if osp.isfile(osp.join(self.ROOT, d, 'pairs.json')))

        self.samples = []          # (scene, stem1, stem2)
        self.scene_stems = {}      # scene -> sorted list of available stems (for retry)
        for scene in scenes:
            with open(osp.join(self.ROOT, scene, 'pairs.json')) as f:
                pairs = json.load(f)
            for stem1, stem2, _overlap, split in pairs:
                if keep is None or split in keep:
                    self.samples.append((scene, stem1, stem2))
            poses_dir = osp.join(self.ROOT, scene, 'poses')
            self.scene_stems[scene] = sorted(osp.splitext(f)[0] for f in os.listdir(poses_dir)
                                             if f.endswith('.json'))

        # {resolution: {(scene, stem): bool}} -- frames with no valid depth at a resolution
        self.invalidate = {}

    def __len__(self):
        return len(self.samples)

    def _read_frame(self, scene, stem):
        scene_dir = osp.join(self.ROOT, scene)
        with open(osp.join(scene_dir, 'poses', stem + '.json')) as f:
            meta = json.load(f)
        intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
        camera_pose = np.array(meta['cam2world'], dtype=np.float32)
        rgb_image = imread_cv2(osp.join(scene_dir, meta['image']))
        depthmap = imread_cv2(osp.join(scene_dir, meta['depth']), cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) * meta['depth_unit_to_meters']  # -> meters, 0 = invalid
        return rgb_image, depthmap, intrinsics, camera_pose

    def _get_views(self, idx, resolution, rng):
        scene, stem1, stem2 = self.samples[idx]
        if resolution not in self.invalidate:
            self.invalidate[resolution] = {}
        inval = self.invalidate[resolution]
        stems = self.scene_stems[scene]

        views = []
        queue = deque([stem2, stem1])  # pop order yields [stem1, stem2]
        while queue:
            stem = queue.pop()

            if inval.get((scene, stem), False):
                # this frame had no valid depth at this resolution: walk to a neighbour
                pos = stems.index(stem)
                direction = 2 * rng.choice(2) - 1
                for off in range(1, len(stems)):
                    cand = stems[(pos + direction * off) % len(stems)]
                    if not inval.get((scene, cand), False):
                        stem = cand
                        break

            rgb_image, depthmap, intrinsics, camera_pose = self._read_frame(scene, stem)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=(scene, stem))

            if (depthmap > 0.0).sum() == 0:
                # no valid depth -> invalidate this frame and retry with a neighbour
                inval[(scene, stem)] = True
                queue.append(stem)
                continue

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=scene + '/' + stem,
                instance=f'{idx}_{stem}',
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = OverMaps(split='train', ROOT="data/overmaps_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
