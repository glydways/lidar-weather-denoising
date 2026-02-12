"""
ROS 2 version of the point-cloud denoiser node.
Use this on Ubuntu 22.04 with ROS 2 Humble.

Requires: pip install ros2-numpy  (for PointCloud2 <-> numpy conversion)
"""

import math
import os
import struct
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
import torch

from weathernet.datasets import DENSE
from weathernet.model import WeatherNet

try:
    import ros2_numpy as rnp
    HAS_ROS2_NUMPY = True
except ImportError:
    HAS_ROS2_NUMPY = False


def denoise(model, device, distance_1, reflectivity_1):
    """Run model on distance and reflectivity; return per-point labels."""
    distance = torch.as_tensor(distance_1.astype(np.float32, copy=False)).contiguous()
    reflectivity = torch.as_tensor(reflectivity_1.astype(np.float32, copy=False)).contiguous()

    # Model expects 4D (B, C, H, W); first conv has kernel (7,3) so H>=7, W>=3.
    # Reshape 1D (N,) -> (1, 1, H, W), then cat -> (1, 2, H, W).
    n_orig = None
    if distance.dim() == 1:
        n = distance.numel()
        n_orig = n
        min_h, min_w = 7, 3
        h = max(min_h, int(math.ceil(math.sqrt(n))))
        w = max(min_w, int(math.ceil(n / h)))
        need = h * w
        if need > n:
            distance = torch.nn.functional.pad(distance.unsqueeze(0), (0, need - n), value=0).squeeze(0)
            reflectivity = torch.nn.functional.pad(reflectivity.unsqueeze(0), (0, need - n), value=0).squeeze(0)
        distance = distance[:need].view(h, w).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        reflectivity = reflectivity[:need].view(h, w).unsqueeze(0).unsqueeze(0)
    elif distance.dim() == 2:
        distance = distance.unsqueeze(0).unsqueeze(0)
        reflectivity = reflectivity.unsqueeze(0).unsqueeze(0)
    else:
        distance = distance.unsqueeze(0).unsqueeze(0)
        reflectivity = reflectivity.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(distance.to(device), reflectivity.to(device))

    pred = torch.argmax(pred, dim=1, keepdim=True)
    labels = torch.squeeze(pred).cpu().numpy()
    if labels.ndim > 1:
        labels = labels.flatten()
    if n_orig is not None and labels.size > n_orig:
        labels = labels[:n_orig]

    label_dict = {0: 0, 1: 100, 2: 101, 3: 102}
    labels = np.vectorize(label_dict.get)(labels)
    return labels


class PC_DenoiserNode(Node):
    """ROS 2 node: subscribe to PointCloud2, run denoiser, publish filtered cloud."""

    def __init__(self, model, device):
        super().__init__("PC_Denoiser")
        self.model = model
        self.device = device
        self._save_dir = os.environ.get("PCD_SAVE_DIR")
        if self._save_dir:
            os.makedirs(self._save_dir, exist_ok=True)

        # Publish RELIABLE so RViz (default RELIABLE) can subscribe.
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Subscribe RELIABLE so it matches the test publisher and RViz defaults.
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(PointCloud2, "PCD_points", pub_qos)
        self.sub = self.create_subscription(
            PointCloud2,
            "/lidar/parent/points_raw",
            self.callback,
            sub_qos,
        )

    def callback(self, msg):
        if not HAS_ROS2_NUMPY:
            self.get_logger().error("ros2_numpy not installed. Run: pip install ros2-numpy")
            return

        try:
            pc = rnp.numpify(msg)
        except Exception as e:
            self.get_logger().error("point_cloud2 numpify failed: %s" % e)
            return

        # ros2_numpy returns dict with "xyz" (n,3) and optionally "intensity" / "rgb"
        if isinstance(pc, dict):
            points = np.asarray(pc["xyz"], dtype=np.float32)
            n = points.shape[0]
            if "intensity" in pc:
                reflect = np.asarray(pc["intensity"]).flatten().astype(np.float32)
                if reflect.size != n:
                    reflect = np.broadcast_to(reflect.ravel()[:1], (n,)).astype(np.float32)
            elif "reflectivity" in pc:
                reflect = np.asarray(pc["reflectivity"]).flatten().astype(np.float32)
            else:
                reflect = np.zeros(n, dtype=np.float32)
        else:
            # Structured array with x, y, z fields
            x = np.asarray(pc["x"]).flatten()
            y = np.asarray(pc["y"]).flatten()
            z = np.asarray(pc["z"]).flatten()
            n = len(x)
            points = np.column_stack([x, y, z]).astype(np.float32)
            if "reflectivity" in pc.dtype.names:
                reflect = np.asarray(pc["reflectivity"]).flatten().astype(np.float32)
            elif "intensity" in pc.dtype.names:
                reflect = np.asarray(pc["intensity"]).flatten().astype(np.float32)
            else:
                reflect = np.zeros(n, dtype=np.float32)

        distance_1 = np.sqrt(np.sum(points ** 2, axis=1))

        labels = denoise(self.model, self.device, distance_1, reflect)

        # Keep only points with label 100 (clear); avoid NaNs for ros2_numpy
        keep = labels == 100
        points_out = np.asarray(points[keep], dtype=np.float32)
        reflect_out = np.asarray(reflect[keep], dtype=np.float32)
        if points_out.size == 0:
            return

        # Optional export (set env var): PCD_SAVE_DIR=/tmp/pcd_exports
        if self._save_dir:
            stamp = msg.header.stamp
            fname = f"{stamp.sec}.{stamp.nanosec:09d}.npz"
            np.savez_compressed(
                os.path.join(self._save_dir, fname),
                frame_id=str(msg.header.frame_id),
                stamp_sec=int(stamp.sec),
                stamp_nanosec=int(stamp.nanosec),
                points_raw=points.astype(np.float32, copy=False),
                intensity_raw=reflect.astype(np.float32, copy=False),
                labels=labels.astype(np.int32, copy=False),
                points_denoised=points_out,
                intensity_denoised=reflect_out,
            )

        # Build PointCloud2 manually (ros2_numpy msgify has np.hstack bug with bytes)
        n = points_out.shape[0]
        point_step = 16  # x,y,z float32 + intensity float32
        data = bytearray(n * point_step)
        for i in range(n):
            base = i * point_step
            struct.pack_into("fff", data, base, float(points_out[i, 0]), float(points_out[i, 1]), float(points_out[i, 2]))
            struct.pack_into("f", data, base + 12, float(reflect_out[i]))
        out_msg = PointCloud2()
        out_msg.header = msg.header
        out_msg.height = 1
        out_msg.width = n
        out_msg.is_dense = True
        out_msg.is_bigendian = False
        out_msg.point_step = point_step
        out_msg.row_step = n * point_step
        out_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out_msg.data = bytes(data)
        self.pub.publish(out_msg)


def main():
    rclpy.init()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = DENSE.num_classes()
    model = WeatherNet(num_classes)
    model = model.to(device)

    # Load best model (change path if needed)
    ckpt_path = "checkpoints/model_epoch197_mIoU=78.8.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    node = PC_DenoiserNode(model, device)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
