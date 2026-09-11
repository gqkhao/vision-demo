"""
nodes/depth_estimator.py

订阅 /objects (bbox JSON) 和 /camera/depth/image_raw，
把 2D bbox 中心点映射到 3D 相机坐标系下的位姿。

依赖 RealSense 的相机内参（/camera/depth/camera_info 或默认 D435i 参数）。

输出：
  /object_3d:  geometry_msgs/PoseStamped (frame_id=camera_link)
    position = 相机坐标系下目标中心
    orientation = 目标朝向 (简化：假设 z 朝上)
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
from cv_bridge import CvBridge
import numpy as np


# ============ 配置 ============
TOPIC_DEPTH = "/camera/depth/image_raw"
TOPIC_OBJECTS = "/objects"
TOPIC_OUT = "/object_3d"

# RealSense D435i 默认内参（focal, principal point, baseline）
# 实际相机请用 /camera/depth/camera_info 获取
FOCAL_X = 460.0
FOCAL_Y = 460.0
PRINCIPAL_X = 310.0
PRINCIPAL_Y = 240.0
DEPTH_UNIT_MM = 1000.0          # ROS 2 深度图默认 mm


class DepthEstimator(Node):
    def __init__(self):
        super().__init__("depth_estimator")
        self.bridge = CvBridge()

        self.depth_sub = self.create_subscription(
            Image, TOPIC_DEPTH, self.depth_cb, 10)
        self.obj_sub = self.create_subscription(
            String, TOPIC_OBJECTS, self.obj_cb, 10)
        self.out_pub = self.create_publisher(PoseStamped, TOPIC_OUT, 10)

        self.latest_depth = None       # numpy float32, shape=(H,W)
        self.latest_obj = None         # dict 或 list

        self.get_logger().info("DepthEstimator ready")

    def depth_cb(self, msg: Image):
        """缓存最新深度图。"""
        try:
            # 深度图默认 32FC1（单位 mm）
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.latest_depth = cv_img
        except Exception as e:
            self.get_logger().warn(f"depth imgmsg_to_cv2 failed: {e}")

    def obj_cb(self, msg: String):
        """收到检测结果，若深度图就绪则计算 3D 位姿。"""
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"bad json: {e}")
            return

        objects = data.get("objects", [])
        if not objects:
            return

        # 取置信度最高的
        best = max(objects, key=lambda o: o.get("conf", 0))
        bbox = best["bbox"]
        label = best["label"]

        if self.latest_depth is None:
            self.get_logger().warn("no depth yet, skip")
            return

        depth = self.latest_depth
        H, W = depth.shape
        x1, y1, x2, y2 = bbox

        # 缩放到深度图尺寸（detector 输出可能已 resize 过）
        sx, sy = W / 640.0, H / 480.0
        bx1, by1 = int(x1 * sx), int(y1 * sy)
        bx2, by2 = int(x2 * sx), int(y2 * sy)
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        # 采样 bbox 中心 3x3 邻域，取中位深度（抗噪）
        roi = depth[
            max(0, int(bcy) - 1):int(bcy) + 2,
            max(0, int(bcx) - 1):int(bcx) + 2,
        ]
        valid = roi[np.isfinite(roi)]
        if valid.size == 0:
            self.get_logger().warn("no valid depth in bbox, skip")
            return

        z_mm = float(np.median(valid))
        z = z_mm / DEPTH_UNIT_MM       # 米

        # 反投影到相机坐标系 (x 右, y 下, z 前)
        x = (bcx - PRINCIPAL_X) * z / FOCAL_X
        y = (bcy - PRINCIPAL_Y) * z / FOCAL_Y

        # 发布 PoseStamped
        pose = PoseStamped()
        pose.header.frame_id = "camera_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        # 简化：四元数 (1,0,0,0) = 单位旋转
        pose.pose.orientation.w = 1.0

        self.out_pub.publish(pose)
        self.get_logger().info(
            f"{label}: x={x:.3f} y={y:.3f} z={z:.3f} m  "
            f"(bbox center=({bcx:.0f},{bcy:.0f}), depth={z_mm:.0f}mm)")


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
