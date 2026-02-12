# LiDAR Rain/Snow Denoise via Segmentation and Filtering (WIP)

This repo contains ML-based LiDAR denoising experiments to remove weather clutter (rain, snow, fog).

**Datasets:** [DENSE](https://www.uni-ulm.de/en/in/driveu/projects/dense-dataset/) (rain), [CADC](http://cadcd.uwaterloo.ca/) (winter), [WADS](https://digitalcommons.mtu.edu/wads/) (snow). See [scripts/data_playground.ipynb](scripts/data_playground.ipynb) for exploration.
For evaluation, we use snow machine data collected in NYNJ at 2026/1/20 ([catalog link](https://catalog.tbdrobotics.com/artifact/2a0c8569-8a4c-4dd7-836f-d7f66631c2e8)).

We use the following `dataset/` layout (local only; not in repo due to size; download them with the link above):

```
dataset/
├── cadc/
│   └── cadcd/               # 2018_03_06/, 2018_03_07/, 2019_02_27/, ... (date/route → labeled/, lidar_points/, etc.)
├── dense/
│   ├── train_01/
│   ├── train_02/
│   ├── val_01/
│   ├── test_01/
│   ├── test_road_01/
│   ├── test_road_02/
│   └── ...                  # DENSE HDF5 (native for WeatherNet)
├── glyd/
│   └── glyd_snow_bin/       # Bin file generated from autolabeling tool
└── wads/
    ├── wads_hdf5/           # 11_20211109T212945Z_001/, 15_..., 22_..., 30_..., ... (WeatherNet)
    ├── wads_original_bin/   # same seq names; .bin + .label (flatten for LiSnowNet)
    └── ...
```

**Data format note:** WeatherNet uses range images in HDF5; LiSnowNet uses raw Velodyne `.bin`. bin→HDF5: [scripts/velodyne_bin_to_hdf5.py](scripts/velodyne_bin_to_hdf5.py).

---

## Setup

**Conda envs** (from exported env files):

```bash
# Create envs/ first if not exists: mkdir -p envs
# Export (run once): conda activate <env> && conda env export --no-builds > envs/<env>-environment.yml

conda env create -f envs/weathernet-environment.yml
conda env create -f envs/lisnownet-environment.yml
```

Or create from scratch via pip:

```bash
# WeatherNet
conda create -n weathernet python=3.10 -y && conda activate weathernet
pip install -r WeatherNet/PointCloudDenoisingTraining/requirements.txt

# LiSnowNet
conda create -n lisnownet python=3.10 -y && conda activate lisnownet
pip install -r lisnownet/requirements.txt
```

---

## WeatherNet pipeline

### DENSE (native format)

DENSE dataset is already in HDF5; use as-is.

```bash
conda activate weathernet
cd WeatherNet/PointCloudDenoisingTraining

python train_dense.py \
  --dataset-dir data/DENSE \
  --output-dir checkpoints_dense \
  --log-dir logs
```

### WADS (needs bin→HDF5 conversion)

WADS raw data is `.bin`. We provide [scripts/velodyne_bin_to_hdf5.py](scripts/velodyne_bin_to_hdf5.py) to convert first, then train:

```bash
# 0. Convert WADS bin → HDF5 first (one-time)
python scripts/velodyne_bin_to_hdf5.py -i dataset/wads/wads_bin_flat -o dataset/wads/wads_hdf5

# 1. train
conda activate weathernet
cd WeatherNet/PointCloudDenoisingTraining
python train_dense.py \
  --dataset-dir dataset/wads/wads_hdf5 \
  --label-map wads \
  --output-dir checkpoints_wads \
  --log-dir logs

# 2. eval → labelled hdf5
python test.py \
  --model checkpoints_wads/model_epoch143_mIoU=60.7.pth \
  --input-dir dataset/glyd/glyd_snow_1run_hdf5 \
  --output-dir out/glyd_snow_wads_out \
  --batch-size 1

# 3. visualize (need ROS)
source /opt/ros/humble/setup.bash
cd ../PointCloudDenoisingVisual/src
python visu_ros2.py --path /path/to/out --hz 5
```

---

## LiSnowNet pipeline

```bash
conda activate lisnownet
cd lisnownet

# 1. train (alpha 5.5 = more aggressive snow removal; lower = more conservative)
python train.py --dataset wads --wads_dir dataset/wads/wads_bin_flat \
  --log_dir ./logs --tag wads --alpha 5.5 --num_epochs 30 --batch_size 32

# 2. eval → denoised .bin (custom any .bin folder, use --lidar wads64 for WADS 64-beam)
python eval.py -i dataset/glyd/glyd_snow_1run_bin -o ./out/glyd_snow_denoised \
  -c ./logs/wads_alpha=5.5/0020.pth --threshold 8e-3 --batch_size 32 --lidar wads64

# 3. denoised bin → labelled hdf5 (for visu)
python scripts/raw_denoised_to_labeled_hdf5.py -r dataset/glyd/glyd_snow_1run_bin \
  -d lisnownet/out/glyd_snow_denoised -o lisnownet/out/glyd_snow_labelled

# 4. visualize (need ROS; visu lives in WeatherNet)
cd ../WeatherNet/PointCloudDenoisingVisual/src
python visu_ros2.py --path ../../../lisnownet/out/glyd_snow_labelled --hz 5
```

---

**More detail:** [WeatherNet](WeatherNet/PointCloudDenoisingTraining/README.md) | [LiSnowNet](lisnownet/README.md) | [PointCloudDenoisingVisual](WeatherNet/PointCloudDenoisingVisual/README.md)