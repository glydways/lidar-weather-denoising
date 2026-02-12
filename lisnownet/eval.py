#!/usr/bin/env python3
import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from glob import glob
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.datasets.wads import WADS
from tools.datasets.cadc import CADC
from tools.datasets.custom_bin import CustomBin
from tools.models import LiSnowNet
from tools.utils import image2points


def _draw_bev(fname_png, xyzi, idx_pr, figure_id, zmin=-2, zmax=6, axlim=30):
    """Draw Raw vs Denoised BEV. Left: all points. Right: kept (gray) + removed noise (pink). idx_pr=True=snow."""
    fig = plt.figure(figure_id, figsize=(8, 4.5), tight_layout=True)
    axes = [fig.add_subplot(1, 2, k + 1) for k in range(2)]
    kept = xyzi[~idx_pr, :] if xyzi.size > 0 else np.zeros((0, 4))
    noise = xyzi[idx_pr, :] if xyzi.size > 0 else np.zeros((0, 4))

    for idx, ax in enumerate(axes):
        if idx:
            n_kept, n_noise = kept.shape[0], noise.shape[0]
            ax.set_title(f'Denoised (kept={n_kept} gray, removed={n_noise} pink)')
            ax.set_yticklabels([])
            if kept.shape[0] > 0:
                ax.scatter(kept[:, 0], kept[:, 1], c='#555555', s=0.5, alpha=0.75, marker=',', label='kept')
            if noise.shape[0] > 0:
                ax.scatter(noise[:, 0], noise[:, 1], c='#E91E8C', s=1.0, alpha=0.9, marker=',', label='removed')
            if kept.shape[0] == 0 and noise.shape[0] == 0:
                ax.text(0.5, 0.5, 'No points', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.set_title('Raw')
            ax.set_ylabel(r'$y$ [m]')
            pts = xyzi if xyzi.size > 0 else np.zeros((0, 4))
            if pts.shape[0] > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.07, vmin=zmin, vmax=zmax, alpha=0.9, marker=',')
            else:
                ax.text(0.5, 0.5, 'No points', ha='center', va='center', transform=ax.transAxes)
        ax.axis('scaled')
        ax.set_xlim(-axlim, axlim)
        ax.set_ylim(-axlim, axlim)
        ax.set_xlabel(r'$x$ [m]')
    fig.savefig(fname_png, dpi=240)
    plt.close(fig)


def save_results(frame, log_dir, save_bev=True, zmin=-2, zmax=6, axlim=30):
    # points: [x, y, z, i, delta_d, delta_i, gt, pr]
    fid, points = frame
    points = points[::-1, :]
    print(f'\t\t\t{fid}', end='\r')

    xyzi, res = points[:, :4], points[:, 4:6]
    idx_pr = points[:, 7].astype(bool)

    # save filtered points
    fname_xyz = os.path.join(
        log_dir,
        os.path.dirname(fid),
        'velodyne',
        os.path.basename(fid)
    )
    os.makedirs(os.path.dirname(fname_xyz), exist_ok=True)
    xyzi[~idx_pr, :].tofile(fname_xyz)

    if not save_bev:
        return

    fname_png = fname_xyz.replace('velodyne', 'bev').replace('.bin', '.png')
    os.makedirs(os.path.dirname(fname_png), exist_ok=True)

    fid_list = fid.replace('.bin', '').replace('\\', '/').split('/')
    if len(fid_list) == 2:
        drive_id, frame_id = int(fid_list[0]), int(fid_list[1])
        figure_id = (drive_id << 4) + frame_id
    elif len(fid_list) == 4:
        _, drive_id, _, frame_id = fid_list
        figure_id = (int(drive_id) << 4) + int(frame_id)
    else:
        figure_id = hash(fid) % (2 ** 16)

    _draw_bev(fname_png, xyzi, idx_pr, figure_id, zmin, zmax, axlim)


def save_results_custom(frame, input_dir, output_dir, save_bev=True, figure_id=0, zmin=-2, zmax=6, axlim=30):
    """Save denoised .bin (and optionally BEV) to output_dir in custom mode. Does NOT touch input_dir."""
    fid, points = frame
    points = points[::-1, :]

    xyzi = points[:, :4]
    idx_pr = points[:, 7].astype(bool)
    denoised = xyzi[~idx_pr, :]

    rel = os.path.relpath(fid, input_dir)
    rel = rel.replace('\\', '/')
    out_path = os.path.join(output_dir, rel)
    print(f'\t\t\t-> {out_path}', end='\r')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    denoised.astype(np.float32).tofile(out_path)

    if not save_bev:
        return

    # BEV: same structure as save_results -> dirname(rel)/bev/basename.png
    rel_dir, rel_base = os.path.split(rel)
    bev_rel = os.path.join(rel_dir, 'bev', os.path.splitext(rel_base)[0] + '.png')
    fname_png = os.path.join(output_dir, bev_rel)
    os.makedirs(os.path.dirname(fname_png), exist_ok=True)
    _draw_bev(fname_png, xyzi, idx_pr, figure_id, zmin=-10, zmax=10, axlim=100)


def _run_bev_only(raw_dir, denoised_dir, match_radius=1e-4, zmin=-2, zmax=6, axlim=30):
    """Regenerate BEV from existing raw + denoised .bin (no model). Match by basename."""
    plt.rcParams.update({'text.usetex': False, 'font.family': 'sans-serif', 'font.size': 10})

    raw_bins = {os.path.basename(p): p for p in glob(os.path.join(raw_dir, '**', '*.bin'), recursive=True)}
    denoised_bins = {}
    for p in glob(os.path.join(denoised_dir, '**', '*.bin'), recursive=True):
        if 'bev' in p.replace('\\', '/').split('/'):
            continue
        denoised_bins[os.path.basename(p)] = p

    common = sorted(set(raw_bins) & set(denoised_bins))
    if not common:
        print('No matching .bin between raw and denoised dirs.')
        return

    bev_dir = os.path.join(denoised_dir, 'bev')
    os.makedirs(bev_dir, exist_ok=True)
    for idx, base in enumerate(common):
        raw_path = raw_bins[base]
        denoised_path = denoised_bins[base]
        raw_pts = np.fromfile(raw_path, dtype=np.float32)
        denoised_pts = np.fromfile(denoised_path, dtype=np.float32)
        if len(raw_pts) % 4 != 0:
            continue
        if len(denoised_pts) == 0:
            xyzi = raw_pts.reshape(-1, 4)
            idx_pr = np.ones(xyzi.shape[0], dtype=bool)
        else:
            raw_pts = raw_pts.reshape(-1, 4)[:, :4]
            denoised_pts = denoised_pts.reshape(-1, 4)[:, :4]
            xyzi = raw_pts
            tree = cKDTree(denoised_pts[:, :3])
            dist, _ = tree.query(raw_pts[:, :3], k=1, distance_upper_bound=match_radius)
            idx_pr = ~np.isfinite(dist)
        fname_png = os.path.join(bev_dir, os.path.splitext(base)[0] + '.png')
        _draw_bev(fname_png, xyzi, idx_pr, idx % (2 ** 16), zmin, zmax, axlim)
        if (idx + 1) % 100 == 0:
            print(f'BEV {idx + 1}/{len(common)}', end='\r')
    print(f'Done. BEV saved to {bev_dir}')


def benchmark(frame):
    # frame[0] is frame_id, frame[1] is the points with GT and PR
    points = frame[1]
    idx_gt, idx_pr = points[:, 6].astype(bool), points[:, 7].astype(bool)

    tp = (idx_pr & idx_gt).sum()
    fp = (idx_pr & ~idx_gt).sum()
    fn = (~idx_pr & idx_gt).sum()
    denom_p = tp + fp
    denom_r = tp + fn
    union = (idx_pr | idx_gt).sum()
    precision = tp / denom_p if denom_p > 0 else (0.0 if tp == 0 else np.nan)
    recall = tp / denom_r if denom_r > 0 else (0.0 if tp == 0 else np.nan)
    iou = tp / union if union > 0 else 0.0

    return precision, recall, iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=1.2e-2)
    parser.add_argument('--z_ground', type=float, default=-1.8)
    parser.add_argument('--snow_id', type=int, default=110)
    parser.add_argument('--dataset', type=str, default='cadc', choices=['cadc', 'wads'])
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root dir for datasets (contains cadcd/ and wads/)')
    parser.add_argument('--cadc_dir', type=str, default='./data/cadcd',
                        help='Path to CADC dataset root')
    parser.add_argument('--wads_dir', type=str, default='./data/wads',
                        help='Path to WADS dataset root')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--input_dir', '-i', type=str, default=None,
                        help='Custom mode: root dir of .bin (with --output_dir)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='Custom mode: output dir for denoised .bin')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Custom mode: path to .pth (required when using --input_dir)')
    parser.add_argument('--no_bev', action='store_true',
                        help='Do not save BEV images (Raw vs Denoised)')
    parser.add_argument('--bev_only', action='store_true',
                        help='Only (re)generate BEV from existing raw + denoised .bin; -i raw dir, -o denoised dir')
    parser.add_argument('--lidar', type=str, default='cadc32',
                        help='Custom mode: lidar preset (e.g. cadc32, wads64, vlp16, ouster64). Use --list_lidar to show all.')
    parser.add_argument('--list_lidar', action='store_true',
                        help='Print available --lidar presets and exit.')
    parser.add_argument('--adaptive', action='store_true',
                        help='Custom mode: map elevation by frame min/max (no fixed INC).')
    config = parser.parse_args()

    if config.list_lidar:
        from tools.datasets.lidar_presets import list_presets
        print('Lidar presets (--lidar <name>):')
        list_presets()
        sys.exit(0)

    config.save_bev = not config.no_bev

    bev_only = config.bev_only
    custom_mode = config.input_dir is not None and config.output_dir is not None

    if bev_only:
        if not custom_mode:
            raise ValueError('--bev_only requires -i (raw .bin dir) and -o (denoised .bin dir)')
        config.input_dir = os.path.abspath(os.path.expanduser(config.input_dir))
        config.output_dir = os.path.abspath(os.path.expanduser(config.output_dir))
        if not os.path.isdir(config.input_dir):
            raise FileNotFoundError(f'Input (raw) dir not found: {config.input_dir}')
        if not os.path.isdir(config.output_dir):
            raise FileNotFoundError(f'Output (denoised) dir not found: {config.output_dir}')
        _run_bev_only(config.input_dir, config.output_dir, match_radius=1e-4)
        sys.exit(0)

    if custom_mode and config.checkpoint is None:
        raise ValueError('Custom mode (--input_dir + --output_dir) requires --checkpoint')
    if custom_mode:
        config.input_dir = os.path.abspath(os.path.expanduser(config.input_dir))
        config.output_dir = os.path.abspath(os.path.expanduser(config.output_dir))
        if not os.path.isdir(config.input_dir):
            raise FileNotFoundError(f'Input dir not found: {config.input_dir}')
        n_bin = len(glob(os.path.join(config.input_dir, '**', '*.bin'), recursive=True))
        if n_bin == 0:
            raise FileNotFoundError(
                f'No .bin files in {config.input_dir}\n'
                'Put Velodyne-style .bin (x,y,z,intensity) under the input dir.'
            )
        os.makedirs(config.output_dir, exist_ok=True)

    if config.data_dir is not None:
        config.cadc_dir = os.path.join(config.data_dir, 'cadcd')
        config.wads_dir = os.path.join(config.data_dir, 'wads')

    config.tag = config.tag.split('/')[-1]
    log_dir = os.path.join(config.log_dir, config.tag)

    if not custom_mode:
        plt.rcParams.update({
        'text.usetex': False,  # avoid LaTeX dependency
        'font.family': 'sans-serif',
        'font.size': 10
        })

    device = torch.device('cuda')

    d_thresh, i_thresh = 2.5, 2 / 255

    if custom_mode:
        from tools.datasets.cadc import WIDTH as W
        from tools.datasets.lidar_presets import get_inc
        inc = get_inc(config.lidar)
        dataset = CustomBin(data_dir=config.input_dir, inc=inc, width=W, training=False, skip=1, adaptive=config.adaptive)
    elif config.dataset == 'cadc':
        dataset = CADC(data_dir=config.cadc_dir, training=False, skip=1)
    elif config.dataset == 'wads':
        dataset = WADS(data_dir=config.wads_dir, training=False, skip=1)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=cpu_count() // 2,
        pin_memory=False,
        drop_last=False
    )

    # Using multiple GPUs
    model = nn.DataParallel(
        LiSnowNet(),
        device_ids=range(torch.cuda.device_count())
    ).to(device)

    if custom_mode:
        ckpt = config.checkpoint
    else:
        checkpoints = sorted(glob(os.path.join(log_dir, '*.pth')))
        ckpt = checkpoints[-1] if checkpoints else None

    if ckpt and os.path.isfile(ckpt):
        print(f'\nLoading the last checkpoint {ckpt:s}')
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')

    model.eval()

    runtime, frames = [], []
    for index, (fid, range_img, xyz_img, lbl_img) in enumerate(loader):
        range_img = range_img.to(device)
        xyz_img, lbl_img = xyz_img.to(device), lbl_img.to(device)

        # Model expects 32 or 64 rows; upsample 16-beam to 32 to avoid padding crash
        h = range_img.shape[2]
        if custom_mode and h < 32:
            target_h = 32
            range_img = F.interpolate(range_img, size=(target_h, range_img.shape[3]), mode='nearest')
            xyz_img = F.interpolate(xyz_img, size=(target_h, xyz_img.shape[3]), mode='nearest')
            lbl_img = F.interpolate(lbl_img.float(), size=(target_h, lbl_img.shape[3]), mode='nearest').long()

        # Forward
        t0 = time.time()

        idx_valid, y = model(range_img)
        residual_img = (y - range_img) * idx_valid
        delta_d, delta_i = [residual_img[:, k, :, :] for k in range(2)]

        # Downsample back to original H if we upsampled 16->32
        if custom_mode and h < 32:
            range_img = range_img[:, :, 0::2, :]
            xyz_img = xyz_img[:, :, 0::2, :]
            lbl_img = lbl_img[:, :, 0::2, :]
            idx_valid = idx_valid[:, :, 0::2, :]
            residual_img = residual_img[:, :, 0::2, :]
            delta_d, delta_i = delta_d[:, 0::2, :], delta_i[:, 0::2, :]

        # convert back to actual readings
        range_img = range_img.pow(3)

        # predictions
        pr_img = delta_d * delta_i.pow(3) > config.threshold
        # snowflakes are higher than the ground plane
        pr_img &= xyz_img[:, 2, :, :] > config.z_ground
        # snowflakes are very dark
        pr_img &= range_img[:, 1, :, :] < i_thresh
        # points within a small distance are 100% snowflakes
        pr_img |= range_img[:, 0, :, :] < d_thresh

        runtime.append((time.time() - t0) / range_img.shape[0])

        # results to be saved
        pr_img = pr_img.unsqueeze(1)
        gt_img = (lbl_img == config.snow_id) if not custom_mode else torch.zeros_like(pr_img, dtype=torch.bool)
        output_img = torch.cat([
            xyz_img,
            range_img[:, 1, :, :].unsqueeze(1),
            residual_img,
            gt_img,
            pr_img
        ], dim=1)
        idx_valid = idx_valid[:, 0, :, :].unsqueeze(1).expand_as(output_img)
        output_img[~idx_valid] = -1

        p_out = image2points(output_img)
        p_out = p_out.detach().cpu().numpy()
        for _fid, p1 in zip(fid, p_out):
            p1 = p1[np.isfinite(p1).all(axis=-1), :]

            n_denoised = int((p1[:, 7] < 0.5).sum())
            n_total = p1.shape[0]
            print(', '.join([
                f'[{index + 1:4d}/{len(loader):4d}] {_fid}',
                f'FPS = {1 / np.median(runtime):.4f}',
                f'kept/total = {n_denoised}/{n_total}'
            ]), end='\n')

            frames.append((_fid, p1))

    print('')
    num_proc = min(3 * cpu_count() // 4, 64)

    if custom_mode:
        n_empty = sum(1 for _, p in frames if p.shape[0] == 0)
        if n_empty > 0:
            print(f'>>> Warning: {n_empty}/{len(frames)} frames have 0 points (empty .bin / BEV). '
                  f'Try another --lidar preset (see --list_lidar) or check .bin format.')
        n_kept = sum(int((p[:, 7] < 0.5).sum()) for _, p in frames)
        n_removed = sum(int((p[:, 7] >= 0.5).sum()) for _, p in frames)
        n_total = sum(p.shape[0] for _, p in frames)
        print(f'>>> Inference time:\t{1000 * np.median(runtime):.2f} ms/frame  ({1 / np.median(runtime):.1f} FPS)')
        print(f'>>> Summary:\t{n_removed:,} removed / {n_total:,} total points  ({100 * n_removed / max(n_total, 1):.2f}% removed)')
        print('Saving results ... ', end='\r')
        for idx, f in enumerate(frames):
            save_results_custom(f, config.input_dir, config.output_dir, save_bev=config.save_bev, figure_id=idx)
        print(f'\nDone. Output: {config.output_dir}')
    else:
        if config.dataset == 'wads':
            with Pool(num_proc) as pool:
                out = pool.map(benchmark, frames)

            precision = np.array([o[0] for o in out])
            recall = np.array([o[1] for o in out])
            iou = np.array([o[2] for o in out])
            score = precision * recall * iou

            print('\n'.join([
                f'>>> Precision:\t{np.mean(precision):.4f} +/- {np.std(precision):.4f}',
                f'>>> Recall:\t{np.mean(recall):.4f} +/- {np.std(recall):.4f}',
                f'>>> IOU:\t{np.mean(iou):.4f} +/- {np.std(iou):.4f}',
                f'>>> Score:\t{np.mean(score):.4f} +/- {np.std(score):.4f}',
                f'>>> Runtime:\t{1000 * np.median(runtime):.4f} ms'
            ]))
        else:
            print(f'No GT point-wise labels for {dataset.name:s}. Skipping the de-noising benchmark.')
            print(f'>>> Inference time:\t{1000 * np.median(runtime):.2f} ms/frame  ({1 / np.median(runtime):.1f} FPS)')

        print('Saving results ... ', end='\r')
        with Pool(num_proc) as pool:
            pool.map(partial(save_results, log_dir=log_dir, save_bev=config.save_bev), frames)

        print('\nDone.')
