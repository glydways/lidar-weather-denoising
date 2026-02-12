#!/usr/bin/env bash
# Publish a static transform so the frame "velodyne" exists. RViz needs this to show point clouds.
# Run this in a terminal and leave it running (or run in background).
# Usage: ./run_tf_velodyne.sh
set -e
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
exec ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map velodyne
