"""Dataset for arbitrary .bin point clouds (no labels). For custom inference."""

import os
from glob import glob
import numpy as np
from .base import Base

# VLP-32C / CADC: 32 beams, -25° to +15°. Use this for Velodyne 32-beam (GLYD, CADC).
# WADS uses 64 beams; if your sensor is 64-beam, pass inc=WADS.INC from caller.
from .cadc import INC, WIDTH


class CustomBin(Base):
    def __init__(self, data_dir, name="CustomBin", inc=INC, width=WIDTH, training=False, skip=1, adaptive=False, **kwargs):
        super().__init__(
            data_dir, name=name, inc=inc, width=width, training=training, skip=skip, **kwargs
        )
        self._data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self._adaptive = adaptive

    @staticmethod
    def get_file_id(file_name):
        return file_name

    def read_file_list(self, data_dir):
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        paths = glob(os.path.join(data_dir, "**", "*.bin"), recursive=True)
        return sorted(paths)

    def points2image(self, points, labels, interleave=True):
        """Override: when adaptive=True, map elevation to rows by frame min/max so no points are lost to extrapolation."""
        if not self._adaptive:
            return super().points2image(points, labels, interleave)

        depth = np.linalg.norm(points[:, :3], axis=-1)
        order = np.argsort(depth)[::-1]
        if interleave:
            num_split = self.num_beams * 4
            order = np.hstack([order[k::num_split] for k in range(num_split)])
        points = points[order, :]
        labels = labels[order]
        depth = depth[order]

        depth_safe = np.maximum(depth, 1e-6)
        ratio = np.clip(points[:, 2] / depth_safe, -1.0, 1.0)
        inclination = np.arcsin(ratio)
        azimuth = np.arctan2(points[:, 1], points[:, 0])

        inc_min, inc_max = inclination.min(), inclination.max()
        if inc_max - inc_min < 1e-9:
            ring = np.zeros_like(inclination, dtype=np.int32)
        else:
            ring = ((inclination - inc_min) / (inc_max - inc_min + 1e-9) * (self.num_beams - 1)).astype(np.int32)
        ring = np.clip(ring, 0, self.num_beams - 1)
        i0 = (self.num_beams - 1) - ring
        i1 = np.floor((1 - 0.5 * (azimuth / np.pi + 1)) * self.width).astype(np.int32)
        i1 = np.clip(i1, 0, self.width - 1)

        idx_ok = np.isfinite(i0) & np.isfinite(i1)
        i0, i1 = i0[idx_ok], i1[idx_ok]
        depth, points, labels = depth[idx_ok], points[idx_ok, :], labels[idx_ok]

        range_img = np.full([2, self.num_beams, self.width], -1, dtype=points.dtype)
        range_img[0, i0, i1] = self.shrink(depth)
        range_img[1, i0, i1] = self.shrink(points[:, 3])
        xyz_img = np.full([3, self.num_beams, self.width], -np.inf, dtype=points.dtype)
        for c in range(3):
            xyz_img[c, i0, i1] = points[:, c]
        lbl_img = np.full([self.num_beams, self.width], -1, dtype=labels.dtype)
        lbl_img[i0, i1] = labels
        lbl_img = np.expand_dims(lbl_img, 0)
        return range_img, xyz_img, lbl_img
