"""
Publish a small test point cloud to /lidar/parent/points_raw so you can see something in RViz
without real lidar. Run run_tf_velodyne.sh and the denoiser first, then this script.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
import struct


def make_pointcloud2_msg(clock, frame_id="velodyne"):
    """Build a PointCloud2 (x, y, z, intensity) for testing."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx) + 1.0
    n = xx.size
    intensity = np.ones(n, dtype=np.float32) * 100.0

    point_step = 16
    data = bytearray(n * point_step)
    for i in range(n):
        base = i * point_step
        struct.pack_into("fff", data, base, float(xx.ravel()[i]), float(yy.ravel()[i]), float(zz.ravel()[i]))
        struct.pack_into("f", data, base + 12, float(intensity[i]))

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = clock.now().to_msg()
    msg.height = 1
    msg.width = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = n * point_step
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = bytes(data)
    return msg


class TestPublisher(Node):
    def __init__(self):
        super().__init__("test_pointcloud_publisher")
        # Publish RELIABLE so RViz (default RELIABLE) can subscribe.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(PointCloud2, "/lidar/parent/points_raw", qos)
        self.timer = self.create_timer(0.5, self.timer_cb)

    def timer_cb(self):
        msg = make_pointcloud2_msg(self.get_clock())
        self.pub.publish(msg)
        self.get_logger().info("Published test point cloud (%d points)" % msg.width, throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = TestPublisher()
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
