#!/usr/bin/env bash
# Echo one message from a topic (with 8s timeout so it doesn't hang if no publisher).
# Usage: ./run_ros2_echo.sh <topic>   e.g.  ./run_ros2_echo.sh /PCD_points
# Use this to see frame_id: run  ./run_ros2_echo.sh /PCD_points  | head -25
set -e
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
timeout 8 ros2 topic echo "$1" --once || true
