"""
nodes/grasp_planner.py

订阅 /object_3d (PoseStamped in camera_link)
用 tf2 转到机械臂基坐标系（frame_target）
调用 MoveIt2 MoveGroup 规划抓取轨迹并发布。

教学用简化版：
  1. 只算笛卡尔终点（不做起姿、不做姿态优化）
  2. 直接执行 pre_grasp + grasp 两个 waypoint
  3. 不做碰撞检查（生产环境请开启 planning_scene）

换机械臂时改顶部常量：
  - Panda:      GROUP_NAME="panda_arm"      FRAME_TARGET="base_link"
  - UR5:        GROUP_NAME="manipulator"    FRAME_TARGET="base_link"
  - Hiwonder:   GROUP_NAME="arm"            FRAME_TARGET="base"
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from std_msgs.msg import Header
import numpy as np

from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK

# 用 ROS 2 官方 moveit_py / moveit_ros_planning_interface（可选）
try:
    from moveit_core.robot_state import RobotState as MC_RS  # 若装了
    HAVE_MOVEIT_CORE = True
except Exception:
    HAVE_MOVEIT_CORE = False


# ============ 配置 ============
TOPIC_IN = "/object_3d"
FRAME_SOURCE = "camera_link"        # 相机坐标系（depth_estimator 输出）
FRAME_TARGET = "base_link"          # 机械臂基坐标系（改这里！）
GROUP_NAME = "panda_arm"            # MoveIt2 planning group（改这里！）
PRE_GRASP_OFFSET_Z = 0.20           # 预抓取点离目标高度（米）
GRASP_TIMEOUT = 3.0                 # IK 求解超时（秒）

# 抓取姿态：工具 Z 轴朝下（对准目标顶部）
# 四元数 (x, y, z, w)：绕 X 轴旋转 90°
GRASP_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class GraspPlanner(Node):
    def __init__(self):
        super().__init__("grasp_planner")
        self.in_sub = self.create_subscription(
            PoseStamped, TOPIC_IN, self.pose_cb, 10)

        # IK 服务客户端
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        while not self.ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for /compute_ik service...")

        self.get_logger().info(
            f"GraspPlanner ready | group={GROUP_NAME} "
            f"target_frame={FRAME_TARGET}")

    def pose_cb(self, msg: PoseStamped):
        """
        msg: 目标在相机坐标系下的位姿
        策略：
          1. 直接用 msg.header.frame_id 转成 FRAME_TARGET
             (tf2 会自动处理，只要 TF 树连通)
          2. 生成 pre_grasp 和 grasp 两个笛卡尔目标
          3. 分别调 IK 求关节角
          4. 打印关节角结果（生产代码应发 MoveAction 给 move_group）
        """
        target = msg.pose.position
        self.get_logger().info(
            f"target in {msg.header.frame_id}: "
            f"x={target.x:.3f} y={target.y:.3f} z={target.z:.3f}")

        # 简化：直接把相机坐标当作 FRAME_TARGET 坐标（假设 TF 树里
        # camera_link -> base_link 平移/旋转已按机械臂+相机标定算好）
        # 生产环境应用 tf2_py.buffer.transform 做坐标变换

        pre = self._make_pose(target.x, target.y, target.z + PRE_GRASP_OFFSET_Z)
        grasp = self._make_pose(target.x, target.y, target.z - 0.02)

        pre_j = self._ik(pre)
        grasp_j = self._ik(grasp)

        if pre_j is None or grasp_j is None:
            self.get_logger().warn("IK failed, skip execution")
            return

        self.get_logger().info(
            "=== GRASP PLAN READY ===")
        self.get_logger().info(
            f"pre_grasp  joints: {self._fmt(pre_j)}")
        self.get_logger().info(
            f"grasp      joints: {self._fmt(grasp_j)}")
        self.get_logger().info(
            "=== 下一步：调用 MoveAction 让 move_group 执行 ===")

    def _make_pose(self, x, y, z):
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        p.orientation.x = GRASP_QUAT[0]
        p.orientation.y = GRASP_QUAT[1]
        p.orientation.z = GRASP_QUAT[2]
        p.orientation.w = GRASP_QUAT[3]
        return p

    def _ik(self, pose):
        """调 /compute_ik 服务求关节角。"""
        req = GetPositionIK.Request()
        req.ik_request.header.frame_id = FRAME_TARGET
        req.ik_request.pose_stamped.header = Header()
        req.ik_request.pose_stamped.header.frame_id = FRAME_TARGET
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose = pose
        req.ik_request.robot_state = RobotState()
        req.ik_request.robot_state.joint_state.header.frame_id = FRAME_TARGET
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.timeout.sec = int(GRASP_TIMEOUT)
        req.ik_request.avoid_collision = True

        if not self.ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("compute_ik service unavailable")
            return None

        fut = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=GRASP_TIMEOUT + 1)
        if not fut.done():
            self.get_logger().warn("compute_ik timeout")
            return None

        try:
            resp = fut.result()
        except Exception as e:
            self.get_logger().warn(f"compute_ik error: {e}")
            return None

        if resp.error_code.val != resp.error_code.SUCCESS:
            self.get_logger().warn(
                f"IK failed: {resp.error_code.val} "
                f"({resp.error_code.msg})")
            return None

        return resp.solution.joint_state.position

    @staticmethod
    def _fmt(joints):
        return " ".join(f"{v:+.3f}" for v in joints)


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
