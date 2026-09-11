"""
scripts/test_detector_synthetic.py

合成图像测试：不接真实相机也能跑通 object_detector + depth_estimator。

生成一张带红色方块 + 深度纹理的合成 RGB-D 图像，
通过 ROS 2 发布到 /camera/rgb/image_raw 和 /camera/depth/image_raw，
观察 /objects 和 /object_3d 话题输出。

用法：
  1. 容器里另开终端，跑 object_detector + depth_estimator
  2. 本脚本发布一帧合成图像
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class SyntheticPublisher(Node):
    def __init__(self):
        super().__init__("synthetic_pub")
        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.timer = self.create_timer(1.0, self.publish)   # 1Hz
        self.count = 0
        self.get_logger().info("SyntheticPublisher ready, will publish @1Hz")

    def publish(self):
        self.count += 1
        rgb = self._make_rgb()
        depth = self._make_depth(rgb)

        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        img_msg.header.frame_id = "camera_link"
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_pub.publish(img_msg)

        # 深度图：mm 单位，32FC1
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.frame_id = "camera_link"
        depth_msg.header.stamp = img_msg.header.stamp
        self.depth_pub.publish(depth_msg)

        self.get_logger().info(
            f"[{self.count}] published rgb={rgb.shape} depth={depth.shape}, "
            f"block at (400,300) size 100x100, z=0.5m")

    def _make_rgb(self):
        """生成一张 640x480 灰底图 + 中央红色方块。"""
        h, w = 480, 640
        img = np.full((h, w, 3), (120, 120, 120), dtype=np.uint8)
        # 红色方块（BGR：蓝色 0，绿色 0，红色 255）
        x, y, sz = 400, 300, 100
        img[y:y + sz, x:x + sz] = (0, 0, 255)
        # 边界画个黑框让目标更明显
        cv2.rectangle(img, (x, y), (x + sz, y + sz), (0, 0, 0), 2)
        # 标签
        cv2.putText(img, "RED", (x + 30, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return img

    def _make_depth(self, rgb):
        """生成匹配的深度图：背景 1000mm，方块 500mm。"""
        h, w = rgb.shape[:2]
        depth = np.full((h, w), 1000.0, dtype=np.float32)   # 背景 1m
        x, y, sz = 400, 300, 100
        depth[y:y + sz, x:x + sz] = 500.0                    # 方块 0.5m
        # 加一点高斯噪声模拟真实传感器
        noise = np.random.normal(0, 5, depth.shape).astype(np.float32)
        depth = np.clip(depth + noise, 10, 2000)
        return depth


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
