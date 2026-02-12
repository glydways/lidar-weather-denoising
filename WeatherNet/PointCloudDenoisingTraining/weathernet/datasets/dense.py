"""
HDF5 range-image datasets for WeatherNet (train_01/val_01 layout, optional label remap).
Base: RangeImageHDF5. DENSE and WADS are thin subclasses with their own classes/label_mapping.
"""
import os
import random
import re
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import h5py

from lilanet.datasets.transforms import Compose, RandomHorizontalFlip, Normalize

Class = namedtuple('Class', ['name', 'id', 'color'])

__all__ = ['RangeImageHDF5', 'DENSE', 'WADS', 'Class', 'Normalize', 'Compose', 'RandomHorizontalFlip']


class RangeImageHDF5(data.Dataset):
    """Base dataset: HDF5 range images under root/train_01 or root/val_01, optional label_mapping."""

    Class = Class
    classes = [
        Class('nolabel', 0, (0, 0, 0)),
        Class('clear', 100, (0, 0, 142)),
        Class('rain', 101, (220, 20, 60)),
        Class('fog', 102, (119, 11, 32)),
    ]
    # Subclasses override for their semantics (e.g. WADS: snow instead of rain).
    label_mapping = None  # optional {source_id: target_id}; target in {0, 100, 101, 102}; missing -> 100

    def __init__(self, root, split='train', transform=None, direct_dir=False, label_mapping=None,
                 sequence_split_val_seqs=None):
        """
        sequence_split_val_seqs: if set (e.g. {15, 22, 30}), scan root for all .hdf5, split train/val by
        sequence (infer seq from parent dir name like 11_20211109... -> 11). No train_01/val_01 needed.
        """
        self.root = os.path.expanduser(root)
        self.lidar_path = os.path.join(self.root, 'lidar_2d')
        self.transform = transform
        self._label_mapping = label_mapping if label_mapping is not None else self.__class__.label_mapping
        self.lidar = []

        if split not in ['train', 'val', 'test', 'all']:
            raise ValueError('Invalid split! Use split="train", split="val" or split="all"')

        if direct_dir:
            self.split = self.root
            if not os.path.isdir(self.split):
                raise FileNotFoundError('Dataset root not found: {}'.format(self.split))
            all_files = [os.path.join(r, file) for r, d, f in os.walk(self.split) for file in f]
            hdf5_files = [p for p in all_files if p.endswith('.hdf5')]
            def _natural_key(p):
                nums = re.findall(r'\d+', os.path.basename(p))
                return (int(nums[-1]) if nums else 0, os.path.basename(p))
            self.lidar = sorted(hdf5_files, key=_natural_key)
        elif sequence_split_val_seqs is not None:
            # WADS-style: root = wads_hdf5 (dirs 11_xxx, 12_xxx), no train_01/val_01
            self.split = self.root
            if not os.path.isdir(self.root):
                raise FileNotFoundError('Dataset root not found: {}'.format(self.root))
            val_seqs = set(sequence_split_val_seqs)
            all_files = [os.path.join(r, file) for r, d, f in os.walk(self.root) for file in f]
            hdf5_files = [p for p in all_files if p.endswith('.hdf5')]
            def _seq_from_path(p):
                parent = os.path.basename(os.path.dirname(p))
                m = re.match(r'^(\d+)', parent)
                return int(m.group(1)) if m else -1
            if split == 'val':
                self.lidar = sorted([p for p in hdf5_files if _seq_from_path(p) in val_seqs])
            else:
                self.lidar = sorted([p for p in hdf5_files if _seq_from_path(p) not in val_seqs])
        else:
            self.split = os.path.join(self.root, '{}_01'.format(split))
            if not os.path.isdir(self.split):
                raise FileNotFoundError(
                    'Dataset split dir not found: {}. Use train_01/val_01 or for WADS pass sequence_split_val_seqs '
                    'and use root=wads_hdf5.'.format(self.split)
                )
            all_files = [os.path.join(r, file) for r, d, f in os.walk(self.split) for file in f]
            hdf5_files = [p for p in all_files if p.endswith('.hdf5')]
            self.lidar = sorted(hdf5_files)

        if len(self.lidar) == 0:
            raise ValueError(
                'No .hdf5 files for split={} under {}.'.format(split, self.split)
            )

    def __getitem__(self, index):
        with h5py.File(self.lidar[index], "r", driver='core') as hdf5:
            distance_1 = hdf5.get('distance_m_1')[()]
            reflectivity_1 = hdf5.get('intensity_1')[()]
            label_1 = hdf5.get('labels_1')[()]

            if self._label_mapping is not None:
                label_1 = np.vectorize(lambda x: self._label_mapping.get(int(x), 100))(label_1)

            sensorX = hdf5.get('sensorX_1')[()]
            sensorY = hdf5.get('sensorY_1')[()]
            sensorZ = hdf5.get('sensorZ_1')[()]

            label_dict = {0: 0, 100: 1, 101: 2, 102: 3}
            label_1 = np.vectorize(lambda x: label_dict.get(x, 0))(label_1)

        distance = torch.as_tensor(distance_1.astype(np.float32, copy=False)).contiguous()
        reflectivity = torch.as_tensor(reflectivity_1.astype(np.float32, copy=False)).contiguous()
        label = torch.as_tensor(label_1.astype(np.float32, copy=False)).contiguous()

        if self.transform:
            distance, reflectivity, label = self.transform(distance, reflectivity, label)

        return distance, reflectivity, label, sensorX, sensorY, sensorZ

    def __len__(self):
        return len(self.lidar)

    @classmethod
    def num_classes(cls):
        return len(cls.classes)

    @staticmethod
    def mean():
        return [0.21, 12.12]

    @staticmethod
    def std():
        return [0.16, 12.32]

    @staticmethod
    def class_weights():
        return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])

    @classmethod
    def get_colormap(cls):
        cmap = torch.zeros([256, 3], dtype=torch.uint8)
        for c in cls.classes:
            cmap[c.id, :] = torch.tensor(c.color, dtype=torch.uint8)
        return cmap


class DENSE(RangeImageHDF5):
    """DENSE LiDAR dataset (clear / rain / fog). No label remapping."""

    classes = [
        Class('nolabel', 0, (0, 0, 0)),
        Class('clear', 100, (0, 0, 142)),
        Class('rain', 101, (220, 20, 60)),
        Class('fog', 102, (119, 11, 32)),
    ]
    label_mapping = None


class WADS(RangeImageHDF5):
    """WADS: same HDF5 layout, labels remapped to snow (110→101) vs clear (100). Use root=wads_hdf5 + sequence_split_val_seqs (no train_01/val_01)."""

    VAL_SEQUENCES = {15, 22, 30}  # validation sequences when using sequence_split_val_seqs
    classes = [
        Class('nolabel', 0, (0, 0, 0)),
        Class('clear', 100, (0, 0, 142)),
        Class('snow', 101, (220, 20, 60)),   # 101 reused for snow (was rain in DENSE)
        Class('fog', 102, (119, 11, 32)),   # unused in WADS but keep 4 classes for model
    ]
    label_mapping = {0: 0, 110: 101}  # falling_snow -> 101 (snow); 111 and rest -> 100 (clear)


if __name__ == '__main__':
    joint_transforms = Compose([
        RandomHorizontalFlip(),
        Normalize(mean=DENSE.mean(), std=DENSE.std())
    ])

    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    def visualize_seg(label_map, dataset_cls=DENSE, one_hot=False):
        if one_hot:
            label_map = np.argmax(label_map, axis=-1)
        out = np.zeros((label_map.shape[0], label_map.shape[1], 3))
        for l in range(1, dataset_cls.num_classes()):
            mask = label_map == l
            out[mask, 0] = np.array(dataset_cls.classes[l].color[1])
            out[mask, 1] = np.array(dataset_cls.classes[l].color[0])
            out[mask, 2] = np.array(dataset_cls.classes[l].color[2])
        return out

    dataset = DENSE('../../data/DENSE', transform=joint_transforms)
    distance, reflectivity, label, *_ = random.choice(dataset)

    print('Distance size: ', distance.size())
    print('Reflectivity size: ', reflectivity.size())
    print('Label size: ', label.size())

    import matplotlib.pyplot as plt
    distance_map = Image.fromarray((255 * _normalize(distance.numpy())).astype(np.uint8))
    reflectivity_map = Image.fromarray((255 * _normalize(reflectivity.numpy())).astype(np.uint8))
    label_map = Image.fromarray((255 * visualize_seg(label.numpy())).astype(np.uint8))
    blend_map = Image.blend(distance_map.convert('RGBA'), label_map.convert('RGBA'), alpha=0.4)

    plt.figure(figsize=(10, 5))
    plt.subplot(221); plt.title("Distance"); plt.imshow(distance_map)
    plt.subplot(222); plt.title("Reflectivity"); plt.imshow(reflectivity_map)
    plt.subplot(223); plt.title("Label"); plt.imshow(label_map)
    plt.subplot(224); plt.title("Result"); plt.imshow(blend_map)
    plt.show()
