#!/usr/bin/env bash
# Publish test point cloud to /lidar/parent/points_raw (no real lidar needed).
# Run run_tf_velodyne.sh and ./run_ros2_node.sh first, then this script.
# 1. Activate conda: conda activate pointcloud-denoise
# 2. Run: ./run_publish_test_points.sh
set -e
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}
cd "$(dirname "$0")"
exec python3 publish_test_points_ros2.py "$@"
