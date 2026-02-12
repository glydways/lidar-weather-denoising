import os
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from weathernet.datasets import DENSE, Normalize, Compose, RandomHorizontalFlip
from weathernet.datasets.transforms import ToTensor
from weathernet.model import WeatherNet
from weathernet.utils import save

import time
import numpy as np
import h5py


def get_data_loaders(data_dir, batch_size=None, num_workers=None, input_dir=None):
    """If input_dir is set, load HDF5 files directly from that folder (single scenario)."""
    normalize = Normalize(mean=DENSE.mean(), std=DENSE.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])
    if input_dir is not None:
        dataset = DENSE(root=input_dir, split='test', transform=transforms, direct_dir=True)
        # No shuffle so output order = input file order (test0 = 1st file, test1 = 2nd, ...) for visu.py comparison
        shuffle = False
    else:
        dataset = DENSE(root=data_dir, split='test', transform=transforms)
        shuffle = True
    test_loader = DataLoader(dataset,
                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return test_loader

def run(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = WeatherNet(num_classes)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size
        args.val_batch_size = device_count * args.val_batch_size

    model = model.to(device)

    test_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.num_workers, input_dir=args.input_dir)


    # Init and load model from args
    model.load_state_dict(torch.load(args.model, map_location=device))

    n_batches = len(test_loader)
    print(n_batches)

    testnum = 0
    inference_times = []

    model.eval()
    for data in test_loader:
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = model(data[0].to(device), data[1].to(device))
            pred = torch.argmax(pred, dim=1, keepdim=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_size = data[0].shape[0]
            inference_times.append((t1 - t0) / batch_size)
            #print(pred.shape)

            #print(data[0].shape, data[1].shape)


            #save predictions back into hdf5
            distance_1 = torch.squeeze(data[0]).cpu().numpy()
            reflectivity_1 = torch.squeeze(data[1]).cpu().numpy()
            label_1 = torch.squeeze(pred).cpu().numpy()
            sensorX = torch.squeeze(data[3]).numpy()
            sensorY = torch.squeeze(data[4]).numpy()
            sensorZ = torch.squeeze(data[5]).numpy()

            label_dict= {0:0, 1:100, 2:101, 3:102}
            label_1 = np.vectorize(label_dict.get)(label_1)

            #TODO: Remove non-valid points


            # HDF5 keys compatible with PointCloudDeNoisingVisual/src/visu.py
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, 'test' + str(testnum) + '.hdf5')
            hf = h5py.File(out_path, 'w')
            hf.create_dataset('distance_m_1', data=distance_1)
            hf.create_dataset('intensity_1', data=reflectivity_1)
            hf.create_dataset('labels_1', data=label_1)
            hf.create_dataset('sensorX_1', data=sensorX)
            hf.create_dataset('sensorY_1', data=sensorY)
            hf.create_dataset('sensorZ_1', data=sensorZ)
            hf.close()
            testnum = testnum + 1

    if inference_times:
        mean_ms = sum(inference_times) / len(inference_times) * 1000
        print(f"Inference latency: {mean_ms:.2f} ms per frame (mean over {len(inference_times)} batches)")
        print(f"Throughput: {1000.0 / mean_ms:.1f} fps")

if __name__ == '__main__':
    parser = ArgumentParser('WeatherNet with PyTorch')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size; use 1 with --input-dir for one output per input (easy visu.py comparison)')
    parser.add_argument('--val-batch-size', type=int, default=10,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument("--dataset-dir", type=str, default="data/DENSE",
                        help="location of the full dataset (used when --input-dir is not set)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="direct folder of HDF5 files (e.g. one scenario); overrides dataset-dir, no test_01 subdir")
    parser.add_argument("--model", type=str, default="checkpoints/model_epoch7_mIoU=75.5.pth",
                        help="path to model checkpoint for inference")
    parser.add_argument("--output-dir", type=str, default="processed_data",
                        help="directory to save inference results as HDF5 (visu.py compatible). With --input-dir and --batch-size 1: test0=1st input, test1=2nd, ...")

    run(parser.parse_args())
