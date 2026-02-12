# ROS 2 setup (Ubuntu 22.04)

Run the point-cloud denoiser with **ROS 2 Humble**. Use **Chrome Remote Desktop (CRD)** to open the desktop and run RViz for visualization.

**Note:** The denoiser node (`./run_ros2_node.sh`) runs with no terminal output until it receives point clouds on `/lidar/parent/points_raw` — that is normal.

---

## 1. Install ROS 2 Humble

```bash
sudo apt update && sudo apt install -y locales curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop

# Add to your shell (zsh → ~/.zshrc, bash → ~/.bashrc)
echo "source /opt/ros/humble/setup.bash" >> ~/.zshrc
source /opt/ros/humble/setup.bash
```

---

## 2. Python (conda env, Python 3.10)

Use one env for both training and ROS 2. Python must be **3.10** (required by `ros2-numpy`).

```bash
conda activate pointcloud-denoise   # or create: conda create -n pointcloud-denoise python=3.10
pip install ros2-numpy --no-deps
pip install transformations pybase64
```

If you already ran `pip install ros2-numpy` and numpy was downgraded: `pip install "numpy>=1.26.4"`.

---

## 3. Run the denoiser node

**Always use the run script** — it sources ROS 2 and sets `PYTHONPATH` / `LD_LIBRARY_PATH` so conda’s Python finds `rclpy` and shared libs (e.g. `librcl_action.so`).

```bash
conda activate pointcloud-denoise
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_ros2_node.sh
```

The env must have: Python 3.10, torch, weathernet, ros2-numpy, h5py. If you see `No module named 'h5py'`, run `pip install h5py`.

---

## 4. Visualize (RViz)

Connect with CRD, then in a terminal. If `source /opt/ros/humble/setup.bash` fails with "no such file or directory: .../setup.sh", use the script (it unsets the env that points at this project):

```bash
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_rviz2.sh
```

Or manually (run all in the same terminal):

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
rviz2
```

In RViz2:

1. Set **Fixed Frame** to `velodyne` (left panel).
2. **Add** → **By topic** → `/PCD_points` → **PointCloud2** (去噪後的點雲).
3. Optional: add `/lidar/parent/points_raw` as another PointCloud2 to see the raw input.

If you see no points:

- Restart both the denoiser and the test publisher so they use the same QoS (sensor_data). The log must not show "incompatible QoS".
- Confirm all three are running: `run_tf_velodyne.sh`, `run_ros2_node.sh`, `run_publish_test_points.sh`.
- If `/lidar/parent/points_raw` shows points but `/PCD_points` is empty, the model may have classified all points as non-clear (rain/fog); that can happen with synthetic test data.

---

## How to see lidar points (step by step)

RViz shows "Frame [velodyne] does not exist" and no points until (1) the frame exists and (2) someone publishes point clouds. Do these in order:

**Terminal 1 – make frame "velodyne" exist (leave running):**
```bash
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_tf_velodyne.sh
```

**Terminal 2 – denoiser (leave running):**
```bash
conda activate pointcloud-denoise
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_ros2_node.sh
```

**Terminal 3 – test point cloud (no real lidar needed; leave running):**
```bash
conda activate pointcloud-denoise
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_publish_test_points.sh
```
This publishes a small grid of points to `/lidar/parent/points_raw` every 0.5 s. The denoiser will process them and publish to `/PCD_points`.

**Terminal 4 (or CRD desktop) – RViz:**
```bash
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_rviz2.sh
```
In RViz: **Global Options** → **Fixed Frame** = `velodyne`. **Add** → **By topic** → `/PCD_points` and `/lidar/parent/points_raw` → **PointCloud2**. You should see a grid of points.

If you have **real lidar**, skip Terminal 3 and run your lidar driver so it publishes to `/lidar/parent/points_raw` (or change the topic name in `ros_test_ros2.py` to match your driver).

---

## Topics

| Topic                     | Type       |
|---------------------------|------------|
| `/lidar/parent/points_raw` | PointCloud2 (input)  |
| `/PCD_points`             | PointCloud2 (output) |

To use another model, edit `ckpt_path` in `ros_test_ros2.py` (default: `checkpoints/model_epoch197_mIoU=78.8.pth`).

---

## Run ROS 2 commands (topic echo, list, etc.)

**If you use zsh:** `source /opt/ros/humble/setup.bash` fails in zsh (setup.bash is for bash). Use the scripts below — they run in bash so source works. Do **not** run `source run_ros2.sh` (that runs in zsh); run `./run_ros2.sh` so the script runs in bash.

**List topics:**
```bash
cd /home/ubuntu/Downloads/cnn_denoising/PointCloudDenoising
./run_ros2.sh topic list
```

**See one message and get frame_id (for RViz Fixed Frame):** Use `run_ros2_echo.sh` — it has an 8s timeout so it won't hang if no one is publishing. If nothing is publishing to the topic, it will exit after 8s with no output.
```bash
./run_ros2_echo.sh /PCD_points | head -25
./run_ros2_echo.sh /lidar/parent/points_raw | head -25
```
Look for `frame_id: '...'` in the output; use that as **Fixed Frame** in RViz. If you see no output, no node is publishing to that topic yet (start the denoiser and/or your lidar driver first).
