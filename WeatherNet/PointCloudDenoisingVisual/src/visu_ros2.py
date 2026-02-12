#!/usr/bin/env python3
"""
" author: Robin Heinzler (original ROS1 version)
" project: point cloud de-noising
" info: ROS2 port for visualization with rviz2
"""

import glob
import os
import re
import sys
import threading
import time
from dataclasses import dataclass

try:
    import termios
    import tty
    _HAVE_TERMIOS = True
except ImportError:
    _HAVE_TERMIOS = False

import h5py
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
except ImportError:
    rclpy = None


# settings
# change input path here
PATH = "~/PointCloudDeNoising/test_01/2018-11-29_104141_Static2-FogB/"
COLOR_LABEL_MAPPING = {
    0: [0, 0, 0],
    100: [158, 158, 158],   # valid / kept
    101: [0, 153, 153],
    102: [115, 0, 230],
    110: [255, 105, 180],   # snow / removed (pink, matches BEV)
    111: [250, 50, 50],   # accumulated snow / removed (red, matches BEV)
}
MAX_FRAMES = 10000


@dataclass(frozen=True)
class Hdf5Frame:
    sensorX_1: np.ndarray
    sensorY_1: np.ndarray
    sensorZ_1: np.ndarray
    distance_m_1: np.ndarray
    intensity_1: np.ndarray
    labels_1: np.ndarray


def load_hdf5_file(filename: str) -> Hdf5Frame:
    """Load one single hdf5 file with point cloud data."""
    with h5py.File(filename, "r", driver="core") as hdf5:
        return Hdf5Frame(
            sensorX_1=hdf5.get("sensorX_1")[()],
            sensorY_1=hdf5.get("sensorY_1")[()],
            sensorZ_1=hdf5.get("sensorZ_1")[()],
            distance_m_1=hdf5.get("distance_m_1")[()],
            intensity_1=hdf5.get("intensity_1")[()],
            labels_1=hdf5.get("labels_1")[()],
        )


def get_rgb(labels: np.ndarray, color_label_mapping=COLOR_LABEL_MAPPING):
    """Return color coding according to input labels."""
    r = g = b = np.zeros_like(labels, dtype=np.float32)
    for label_id, color in color_label_mapping.items():
        r = np.where(labels == label_id, color[0] / 255.0, r)
        g = np.where(labels == label_id, color[1] / 255.0, g)
        b = np.where(labels == label_id, color[2] / 255.0, b)
    return r, g, b


class Ros2Publisher(Node):
    def __init__(self, node_name: str = "pointcloud_denoising_visu"):
        super().__init__(node_name)
        self.publisher = self.create_publisher(PointCloud2, "pointcloud", 10)

    def publish_frame(self, frame: Hdf5Frame, frame_id: str = "base"):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        labels = frame.labels_1.flatten()
        r, g, b = get_rgb(labels)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="distance", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="r", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="g", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="b", offset=28, datatype=PointField.FLOAT32, count=1),
        ]

        points = list(
            zip(
                frame.sensorX_1.flatten(),
                frame.sensorY_1.flatten(),
                frame.sensorZ_1.flatten(),
                frame.distance_m_1.flatten(),
                frame.intensity_1.flatten(),
                r,
                g,
                b,
            )
        )

        cloud = point_cloud2.create_cloud(header, fields, points)
        self.publisher.publish(cloud)


def _natural_sort_key(path: str):
    """Natural sort: extract all numbers from basename, sort by tuple of ints. test2 < test10, labeled_001 < labeled_002."""
    base = os.path.basename(path)
    numbers = re.findall(r"\d+", base)
    return (tuple(int(x) for x in numbers) if numbers else (0,), path)


def _get_key_no_enter():
    """Read one key without Enter (Unix). Fallback: return None and use line input."""
    if not _HAVE_TERMIOS or not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    except (OSError, termios.error):
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (OSError, termios.error):
            pass


def main(
    path: str = PATH,
    max_frames: int = MAX_FRAMES,
    publish_hz: float = 2.0,
    loop: bool = True,
    interactive: bool = True,
    step_mode: bool = False,
):
    print("### start PointCloudDeNoising visualization (ROS2)")

    expanded_path = os.path.expanduser(str(path))
    if not expanded_path.endswith("/"):
        expanded_path += "/"

    raw_files = glob.glob(expanded_path + "*.hdf5")
    files = sorted(raw_files, key=_natural_sort_key)
    print(f"Directory {expanded_path} contains {len(files)} hdf5-files")

    if len(files) == 0:
        print(f"Please check the input dir {expanded_path}. Could not find any hdf5-file")
        return

    if rclpy is None:
        print("ImportError rclpy / ROS2!")
        print("No visualization; dumping first frame arrays to terminal.")
        frame0 = load_hdf5_file(files[0])
        print("distance_m_1", frame0.distance_m_1.flatten())
        print("intensity_1", frame0.intensity_1.flatten())
        print("labels_1", frame0.labels_1.flatten())
        return

    rclpy.init()
    node = Ros2Publisher()

    print("In rviz2: Add display 'PointCloud2', Topic = /pointcloud, Fixed Frame = base")
    frames_to_play = files[:max_frames] if max_frames > 0 else files
    n_frames = len(frames_to_play)

    if step_mode:
        # Step mode: no auto-play; a=prev, d=next, q=quit (single key, no Enter when termios available)
        use_raw = _HAVE_TERMIOS and sys.stdin.isatty()
        print(f"Step mode: {n_frames} frames. [a] prev [d] next [q] quit" + (" (no Enter)" if use_raw else " then Enter"))
        idx = 0
        frame = load_hdf5_file(frames_to_play[idx])
        node.publish_frame(frame)
        node.get_logger().info(f"{idx:04d} / {n_frames}  {os.path.basename(frames_to_play[idx])}")
        try:
            while True:
                key = _get_key_no_enter() if use_raw else None
                if key is None:
                    line = input(" [a] prev [d] next [q] quit: ").strip().lower()
                    key = line[:1] if line else ""
                if key == "q":
                    break
                if key == "a":
                    idx = max(0, idx - 1)
                elif key == "d":
                    idx = min(n_frames - 1, idx + 1)
                else:
                    continue
                frame = load_hdf5_file(frames_to_play[idx])
                node.publish_frame(frame)
                rclpy.spin_once(node, timeout_sec=0.0)
                node.get_logger().info(f"{idx:04d} / {n_frames}  {os.path.basename(frames_to_play[idx])}")
        except (KeyboardInterrupt, EOFError):
            pass
        node.destroy_node()
        rclpy.shutdown()
        print("### End of PointCloudDeNoising visualization (ROS2)")
        return

    delay_s = 1.0 / float(publish_hz) if publish_hz > 0 else 0.0
    print(f"Publishing {len(frames_to_play)} frames at ~{publish_hz} Hz (loop={loop}). Ctrl+C to stop.")
    if interactive:
        print("CLI: [p] + Enter = pause/play, [q] + Enter = quit")

    state = {"paused": False, "running": True}

    def input_thread():
        while state["running"]:
            try:
                line = sys.stdin.readline().strip().lower()
                if not line:
                    continue
                if "q" in line:
                    state["running"] = False
                    print("Quit requested.")
                    break
                if "p" in line:
                    state["paused"] = not state["paused"]
                    print("Paused." if state["paused"] else "Playing.")
            except (EOFError, OSError):
                break

    if interactive:
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()

    try:
        while state["running"]:
            for frame_idx, filename in enumerate(frames_to_play):
                if not state["running"]:
                    break
                frame = load_hdf5_file(filename)
                while state["paused"] and state["running"]:
                    node.publish_frame(frame)
                    rclpy.spin_once(node, timeout_sec=0.0)
                    time.sleep(0.05)
                if not state["running"]:
                    break
                node.get_logger().info(f"{frame_idx:04d} / {os.path.basename(filename)}")
                node.publish_frame(frame)
                rclpy.spin_once(node, timeout_sec=0.0)
                if delay_s > 0:
                    time.sleep(delay_s)
            if not loop or not state["running"]:
                break
            node.get_logger().info("Loop: replaying from start...")
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        node.destroy_node()
        rclpy.shutdown()

    print("### End of PointCloudDeNoising visualization (ROS2)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Visualize point cloud HDF5 with ROS2 / rviz2")
    ap.add_argument("--path", type=str, default=PATH, help="Directory containing *.hdf5 files")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Max frames per loop (0 = all)")
    ap.add_argument("--hz", type=float, default=2.0, help="Publish rate (Hz); lower = easier to see")
    ap.add_argument("--no-loop", action="store_true", help="Play once and exit (default: loop)")
    ap.add_argument("--no-input", action="store_true", help="Disable CLI pause/play (p/q)")
    ap.add_argument("--step", action="store_true", help="Step mode: no auto-play; [a] prev [d] next [q] quit (single key, no Enter on Unix)")
    args = ap.parse_args()
    main(
        path=args.path,
        max_frames=args.max_frames,
        publish_hz=args.hz,
        loop=not args.no_loop,
        interactive=not args.no_input,
        step_mode=args.step,
    )

