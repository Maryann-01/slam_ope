import math
import os
import random

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class ResetPlugin:
    def init(self, webots_node, properties):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("reset_plugin")
        self.ready = False

        self.robot = webots_node.robot
        self.robot_def = properties.get("robotDef", "TurtleBot3Burger")
        self.noise_xy = float(properties.get("startNoiseXY", "0.10"))
        self.noise_yaw = float(properties.get("startNoiseYaw", "0.10"))
        # FIXED nominal start pose (world frame). Never read from the robot,
        # because the robot ends each episode somewhere else.
        self.nominal_x = float(properties.get("startX", "0.7584"))
        self.nominal_y = float(properties.get("startY", "2.0718"))
        self.nominal_yaw = float(properties.get("startYaw", "0.8204"))

        self.robot_node = self.robot.getFromDef(self.robot_def)
        if self.robot_node is None:
            self.node.get_logger().error(f"Could not find DEF {self.robot_def}")
            return

        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        self.nominal_z = self.translation_field.getSFVec3f()[2]

        seed_env = os.environ.get("OPE_EPISODE_SEED")
        self.rng = random.Random(int(seed_env) if seed_env is not None else None)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gt_pub = self.node.create_publisher(PoseStamped, "/ground_truth_pose", 10)
        self.start_pub = self.node.create_publisher(
            Float64MultiArray, "/episode_start_pose", latched)
        self.service = self.node.create_service(
            Trigger, "/reset_robot_pose", self.reset_robot_pose)

        x, y, yaw = self._teleport_with_noise()
        msg = Float64MultiArray()
        msg.data = [x, y, yaw]
        self.start_pub.publish(msg)

        self.ready = True
        self.node.get_logger().info(
            f"Reset plugin ready. Episode start pose (seed={seed_env}): "
            f"x={x:.3f} y={y:.3f} yaw={yaw:.3f}")

    def _current_yaw(self):
        m = self.robot_node.getOrientation()
        return math.atan2(m[3], m[0])

    def _zero_wheel_joints(self):
        """Zero the wheel encoders so the new diff-drive controller starts at odom (0,0,0)."""
        for name in ("LEFT_JOINT", "RIGHT_JOINT"):
            try:
                joint = self.robot_node.getFromProtoDef(name)
                params = joint.getField("jointParameters").getSFNode()
                params.getField("position").setSFFloat(0.0)
            except Exception as exc:
                self.node.get_logger().warn(f"Could not zero {name}: {exc}")

    def _teleport_with_noise(self):
        x = self.nominal_x + self.rng.gauss(0.0, self.noise_xy)
        y = self.nominal_y + self.rng.gauss(0.0, self.noise_xy)
        yaw = self.nominal_yaw + self.rng.gauss(0.0, self.noise_yaw)
        self.translation_field.setSFVec3f([x, y, self.nominal_z])
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, yaw])
        # (wheel zeroing not permitted on PROTO internals; policy anchors to first SLAM pose instead)
        self.robot_node.resetPhysics()
        return x, y, yaw

    def reset_robot_pose(self, request, response):
        if not self.ready:
            response.success = False
            response.message = "Reset plugin is not ready."
            return response
        x, y, yaw = self._teleport_with_noise()
        response.success = True
        response.message = f"{x:.4f},{y:.4f},{yaw:.4f}"
        self.node.get_logger().info(f"Robot teleported to {response.message}")
        return response

    def step(self):
        if not hasattr(self, "node"):
            return
        if self.ready:
            pos = self.robot_node.getPosition()
            yaw = self._current_yaw()
            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)
            self.gt_pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0)
