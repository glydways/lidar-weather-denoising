#!/usr/bin/env bash
# Run any ROS 2 command. Must run as ./run_ros2.sh (not "source run_ros2.sh") so it runs in bash.
# In zsh, "source setup.bash" fails; this script runs in bash so source works.
# Usage: ./run_ros2.sh <command and args>
# Example: ./run_ros2.sh topic list
# Example: ./run_ros2.sh topic echo /PCD_points --once
set -e
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
exec ros2 "$@"
