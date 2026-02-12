#!/usr/bin/env bash
# Launch RViz2 with ROS 2 Humble. Use this if "source ... setup.bash" fails with "setup.sh not found".
# Run from any directory (e.g. after connecting with CRD).

set -e
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
exec rviz2 "$@"
