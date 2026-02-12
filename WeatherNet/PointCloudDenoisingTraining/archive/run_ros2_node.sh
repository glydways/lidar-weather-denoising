#!/usr/bin/env bash
# Run the ROS 2 denoiser node.
# 1. Activate your conda env first: conda activate pointcloud-denoise
# 2. Then run: ./run_ros2_node.sh
# (Env must have Python 3.10, torch, weathernet, ros2-numpy, h5py.)

set -e
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash

# Conda Python cannot see ROS 2 packages or .so files without these:
# local/ = rclpy, sensor_msgs; lib/.../site-packages = rpyutils and other ROS 2 Python deps
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}

cd "$(dirname "$0")"
exec python3 ros_test_ros2.py "$@"
