#!/usr/bin/env python3
"""
Convert raw + denoised .bin point clouds to HDF5 with point-wise labels.
Points present in denoised -> label 100 (valid); points only in raw -> label 110 (snow).
Output format compatible with visu_ros2.py.

Example (GLYD):
  python scripts/raw_denoised_to_labeled_hdf5.py \\
    -r dataset/glyd/glyd_snow_highlight_bin \\
    -d lisnownet/out/glyd_snow_denoised \\
    -o lisnownet/out/glyd_snow_labelled
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Label IDs (match WADS / visu convention)
LABEL_VALID = 100
LABEL_SNOW = 110


def _natural_sort_key(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _path_to_key(path: str, root: str) -> str:
    """Extract stable key: date/drive/frame.bin (CADC), seq/frame.bin (WADS), or basename (GLYD flat)."""
    rel = os.path.relpath(path, root)
    parts = rel.replace("\\", "/").split("/")
    key_parts = []
    for i, p in enumerate(parts):
        if re.match(r"\d{4}_\d{2}_\d{2}", p):  # date
            key_parts = [p]
            if i + 1 < len(parts) and re.match(r"\d+", parts[i + 1]):
                key_parts.append(parts[i + 1])  # drive
            break
        if re.match(r"\d{2}", p) and len(p) <= 3:  # WADS seq e.g. 11, 12
            key_parts = [p]
            break
    key_parts.append(parts[-1])  # frame.bin (e.g. 000357.bin for GLYD)
    return "/".join(key_parts)


def _collect_bins_by_key(root: str) -> dict:
    """Recursively find all .bin, return {key: fullpath}. Key = date/drive/frame or seq/frame."""
    root = os.path.abspath(os.path.expanduser(root))
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*.bin"), recursive=True):
        p = os.path.abspath(p)
        key = _path_to_key(p, root)
        if key in out:
            if len(p) < len(out[key]):
                out[key] = p
        else:
            out[key] = p
    return out


def process_frame(raw_path: str, denoised_path: str, match_radius: float = 0.2) -> dict:
    """
    Label raw points: 100=valid (in denoised), 110=snow (only in raw).
    For each raw point, if a denoised point exists within match_radius (m) -> 100 (kept), else 110 (removed).
    Returns dict with sensorX_1, sensorY_1, sensorZ_1, distance_m_1, intensity_1, labels_1.
    """
    raw = np.fromfile(raw_path, dtype=np.float32)
    denoised = np.fromfile(denoised_path, dtype=np.float32)

    for arr, name in [(raw, "raw"), (denoised, "denoised")]:
        if len(arr) % 4 != 0 and len(arr) % 5 != 0:
            raise ValueError(f"{name} bad size {len(arr)}")
    cols_raw = 4 if len(raw) % 4 == 0 else 5
    cols_denoised = 4 if len(denoised) % 4 == 0 else 5

    pts_raw = raw.reshape(-1, cols_raw)[:, :4]  # x,y,z,i
    pts_denoised = denoised.reshape(-1, cols_denoised)[:, :4]

    xyz_raw = pts_raw[:, :3]
    xyz_denoised = pts_denoised[:, :3]

    tree = cKDTree(xyz_denoised)
    distances, _ = tree.query(xyz_raw, k=1, distance_upper_bound=match_radius)
    # query returns inf when no point within radius
    labels = np.where(np.isfinite(distances), LABEL_VALID, LABEL_SNOW).astype(np.float32)

    dist = np.sqrt((xyz_raw ** 2).sum(axis=1)).astype(np.float32)
    intensity = pts_raw[:, 3].astype(np.float32)

    return {
        "sensorX_1": xyz_raw[:, 0].astype(np.float32),
        "sensorY_1": xyz_raw[:, 1].astype(np.float32),
        "sensorZ_1": xyz_raw[:, 2].astype(np.float32),
        "distance_m_1": dist,
        "intensity_1": intensity,
        "labels_1": labels,
    }


def convert(raw_root: str, denoised_root: str, out_dir: str, prefix: str = "labeled", match_radius: float = 0.2):
    raw_map = _collect_bins_by_key(raw_root)
    denoised_map = _collect_bins_by_key(denoised_root)

    common = sorted(set(raw_map) & set(denoised_map), key=lambda k: (_natural_sort_key(k), k))
    if not common:
        print("No matching .bin files between raw and denoised roots.")
        return

    os.makedirs(out_dir, exist_ok=True)
    for i, base in enumerate(common):
        try:
            data = process_frame(raw_map[base], denoised_map[base], match_radius=match_radius)
            out_path = os.path.join(out_dir, f"{prefix}_{i:06d}.hdf5")
            with h5py.File(out_path, "w") as f:
                for k, v in data.items():
                    f.create_dataset(k, data=v)
        except Exception as e:
            print(f"Skip {base}: {e}")
            continue
        if (i + 1) % 100 == 0:
            print(f"Written {i + 1}/{len(common)}")
    print(f"Done: {len(common)} files -> {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Raw + denoised .bin -> HDF5 with point-wise labels (100=valid, 110=snow)."
    )
    p.add_argument("--raw-dir", "-r", required=True, help="Root dir for raw .bin (searched recursively)")
    p.add_argument("--denoised-dir", "-d", required=True, help="Root dir for denoised .bin (from eval)")
    p.add_argument("--output-dir", "-o", required=True, help="Output dir for .hdf5")
    p.add_argument("--prefix", type=str, default="labeled")
    p.add_argument("--match-radius", type=float, default=0.2, help="Max distance (m) to match raw point to denoised; default 0.2 (lidar projection merges nearby points)")
    args = p.parse_args()
    convert(args.raw_dir, args.denoised_dir, args.output_dir, args.prefix, args.match_radius)


if __name__ == "__main__":
    main()
