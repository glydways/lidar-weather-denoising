#!/usr/bin/env python3
"""Convert Velodyne-style .bin point clouds to HDF5 range images (DENSE/visu compatible).

Output HDF5 can be used for WeatherNet training (WeatherNet/PointCloudDenoisingTraining/train_dense.py).
The training code does not hardcode rows/cols. It loads whatever shape is in the HDF5 (see weathernet/datasets/dense.py).
Larger rows/cols retain more points and give finer angular resolution, at the cost of higher memory per sample (smaller batch size).
Default 32×400 matches original DENSE; use e.g. --cols 2048 for higher azimuth resolution.
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np


def _natural_sort_key(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


# VLP-32C / CADC / GLYD elevation range (rad). Use for fixed row mapping so row index = same beam across frames.
# Note: WADS .bin stores intensity 0–255; this script does not scale to 0–1. For WeatherNet you may need to normalize in the dataset or train with WADS-specific mean/std.
PHI_RANGE_VLP32 = (np.deg2rad(-25.0), np.deg2rad(15.0))


def bin_to_range_image(pts_xyzi, labels=None, rows=32, cols=400, phi_range=None):
    """Project (N,4) x,y,z,intensity to (rows,cols) range image; keep nearest per pixel.
    labels: if provided (length N, int32), labels_img gets the winning point's label; -1 for empty pixels.
    phi_range: if provided (phi_min_rad, phi_max_rad), map elevation to rows using this fixed range (row = same beam across frames).
               Points outside range are skipped. If None, use this frame's phi.min()/phi.max() (row meaning varies per frame)."""
    x, y, z = pts_xyzi[:, 0], pts_xyzi[:, 1], pts_xyzi[:, 2]
    intensity = pts_xyzi[:, 3] if pts_xyzi.shape[1] >= 4 else np.zeros_like(x)
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x * x + y * y))

    col = np.clip(((theta + np.pi) / (2 * np.pi) * cols).astype(np.int32), 0, cols - 1)
    if phi_range is not None:
        phi_min, phi_max = phi_range[0], phi_range[1]
        row_float = (phi - phi_min) / (phi_max - phi_min + 1e-9) * (rows - 1)
        row = np.clip(row_float.round().astype(np.int32), 0, rows - 1)
        valid_row = (phi >= phi_min) & (phi <= phi_max)
    else:
        phi_min, phi_max = phi.min(), phi.max()
        if phi_max - phi_min < 1e-6:
            row = np.zeros_like(col, dtype=np.int32)
            valid_row = np.ones_like(col, dtype=bool)
        else:
            row = np.clip(
                ((phi - phi_min) / (phi_max - phi_min + 1e-9) * (rows - 1)).astype(np.int32),
                0, rows - 1,
            )
            valid_row = np.ones_like(col, dtype=bool)

    dist_img = np.full((rows, cols), np.nan, dtype=np.float32)
    int_img = np.zeros((rows, cols), dtype=np.float32)
    sx_img = np.zeros((rows, cols), dtype=np.float32)
    sy_img = np.zeros((rows, cols), dtype=np.float32)
    sz_img = np.zeros((rows, cols), dtype=np.float32)
    labels_img = np.full((rows, cols), -1, dtype=np.int32)
    order = np.argsort(r)
    for idx in order:
        if not valid_row[idx]:
            continue
        ri, ci = row[idx], col[idx]
        if np.isnan(dist_img[ri, ci]) or r[idx] < dist_img[ri, ci]:
            dist_img[ri, ci] = r[idx]
            int_img[ri, ci] = intensity[idx]
            sx_img[ri, ci] = x[idx]
            sy_img[ri, ci] = y[idx]
            sz_img[ri, ci] = z[idx]
            if labels is not None:
                labels_img[ri, ci] = labels[idx]
    n_kept = int(np.isfinite(dist_img).sum())
    dist_img = np.nan_to_num(dist_img, nan=0.0)
    return dist_img, int_img, labels_img, sx_img, sy_img, sz_img, n_kept


def convert(bin_dir, out_dir, rows=32, cols=400, prefix="test", label_dir=None, phi_range=None):
    """Convert .bin point clouds to HDF5 range images (DENSE/visu format).
    rows, cols: range image shape (rows = beams, cols = azimuth bins). Default 32×400.
    phi_range: (phi_min_rad, phi_max_rad) for fixed elevation mapping (e.g. PHI_RANGE_VLP32); None = per-frame min/max.
    label_dir: if set, look for .label next to each .bin (same basename); or infer from bin_path (velodyne->labels, .bin->.label)."""
    os.makedirs(out_dir, exist_ok=True)
    raw_bins = glob.glob(os.path.join(os.path.expanduser(bin_dir), "*.bin"))
    bin_list = sorted(raw_bins, key=_natural_sort_key)
    n = len(bin_list)
    if n == 0:
        print(f"No .bin files in {bin_dir}")
        return

    total_orig, total_kept = 0, 0
    out_idx = 0
    for i, bin_path in enumerate(bin_list):
        raw = np.fromfile(bin_path, dtype=np.float32)
        # Currently assume (N, 4) per point: x, y, z, intensity. (Some formats use (N, 5) with 5th column e.g. ring/beam id; not supported here.)
        if len(raw) % 4 != 0:
            print(f"Skip {os.path.basename(bin_path)}: size {len(raw)} not divisible by 4")
            continue
        pts = raw.reshape(-1, 4)
        labels_1d = None
        if label_dir is not None:
            label_path = os.path.join(os.path.expanduser(label_dir), os.path.basename(bin_path).replace(".bin", ".label"))
        else:
            label_path = bin_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.isfile(label_path):
            labels_raw = np.fromfile(label_path, dtype=np.int32)
            if len(labels_raw) == len(pts):
                pts, idx_unique = np.unique(pts, axis=0, return_index=True)
                labels_1d = labels_raw[idx_unique]
            else:
                labels_1d = None
        n_orig = len(pts)
        dist, inty, labels_img, sx, sy, sz, n_kept = bin_to_range_image(
            pts, labels=labels_1d, rows=rows, cols=cols, phi_range=phi_range
        )
        total_orig += n_orig
        total_kept += n_kept
        n_dropped = n_orig - n_kept
        print(f"  {os.path.basename(bin_path)}: {n_orig} pts -> {n_kept} kept, {n_dropped} dropped")
        out_path = os.path.join(out_dir, f"{prefix}_{out_idx:06d}.hdf5")
        out_idx += 1
        with h5py.File(out_path, "w") as f:
            f.create_dataset("distance_m_1", data=dist)
            f.create_dataset("intensity_1", data=inty)
            f.create_dataset("labels_1", data=labels_img)
            f.create_dataset("sensorX_1", data=sx)
            f.create_dataset("sensorY_1", data=sy)
            f.create_dataset("sensorZ_1", data=sz)
        if (i + 1) % 500 == 0:
            print(f"Written {i + 1}/{n}")
    total_dropped = total_orig - total_kept
    print(f"Done: {n} files -> {out_dir}")
    print(f"Total: {total_orig} pts -> {total_kept} kept ({total_dropped} dropped)")


def main():
    p = argparse.ArgumentParser(description="Convert Velodyne .bin to HDF5 (DENSE/visu format).")
    p.add_argument("--input-dir", "-i", required=True, help="Directory containing *.bin files")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory for *.hdf5")
    p.add_argument("--label-dir", "-l", default=None, help="Directory for *.label (optional). If omitted, infer from input path: velodyne->labels, .bin->.label")
    p.add_argument("--rows", type=int, default=32, help="Range image rows (beams). Default 32 (DENSE/CADC-style).")
    p.add_argument("--cols", type=int, default=400, help="Range image cols (azimuth bins). Default 400. Use e.g. 2048 for finer resolution (more points retained).")
    p.add_argument("--prefix", type=str, default="test", help="Output filename prefix (default: test)")
    p.add_argument("--phi-range", choices=("vlp32", "auto"), default="vlp32",
                   help="Elevation (phi) mapping: vlp32 = fixed -25° to +15° (VLP-32C/CADC/GLYD, row = same beam across frames); auto = per-frame min/max (default: vlp32)")
    args = p.parse_args()
    phi_range = PHI_RANGE_VLP32 if args.phi_range == "vlp32" else None
    convert(args.input_dir, args.output_dir, rows=args.rows, cols=args.cols, prefix=args.prefix, label_dir=args.label_dir, phi_range=phi_range)


if __name__ == "__main__":
    main()
