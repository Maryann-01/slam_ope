import csv
import math
import os

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_quaternion

GOALS = [(3.0, 1.5), (-2.0, 2.0), (3.5, -2.5), (-3.0, 0.5), (0.0, -3.0)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class BehaviourPolicy(Node):
    COLUMNS = [
        "episode", "timestep", "wall_time",
        "goal_x", "goal_y",
        "start_x", "start_y", "start_yaw",
        "slam_x", "slam_y", "slam_yaw", "cov_xx", "cov_yy", "cov_xy", "cov_yaw",
        "est_x", "est_y", "est_yaw", "pose_source",
        "true_x", "true_y", "true_yaw",
        "odom_x", "odom_y", "odom_yaw",
        "front_dist", "left_dist", "right_dist",
        "est_dist_goal", "true_dist_goal",
        "est_heading_err", "true_heading_err",
        "action_linear", "action_angular", "action_angular_mean", "sigma",
        "log_prob", "mode",
        "reward", "terminated", "truncated", "term_reason", "belief_goal_reached",
        "next_slam_x", "next_slam_y", "next_slam_yaw",
        "next_cov_xx", "next_cov_yy", "next_cov_xy", "next_cov_yaw",
        "next_est_x", "next_est_y", "next_est_yaw",
        "next_true_x", "next_true_y", "next_true_yaw",
        "next_odom_x", "next_odom_y", "next_odom_yaw",
        "next_front_dist", "next_left_dist", "next_right_dist",
        "next_est_dist_goal", "next_true_dist_goal",
        "next_est_heading_err",
    ]

    def __init__(self):
        super().__init__("behaviour_policy")

        p = self.declare_parameters("", [
            ("episode", 1),
            ("seed", 0),
            ("output_dir", os.path.expanduser("~/ope_data")),
            ("policy_name", "behaviour_a"),
            # Stochastic policy, required for importance sampling. Same in both policies.
            ("sigma", 0.3),
            ("max_timesteps", 500),
            ("goal_radius", 0.75),
            ("collision_dist", 0.15),
            ("startup_timeout_s", 90.0),
            ("speed_fast", 0.20),
            ("speed_slow", 0.12),
            ("gain", 1.5),
        ])
        g = {x.name: x.value for x in p}
        self.episode = int(g["episode"])
        self.seed = int(g["seed"])
        self.output_dir = os.path.expanduser(str(g["output_dir"]))
        self.policy_name = str(g["policy_name"])
        self.sigma = float(g["sigma"])
        self.max_timesteps = int(g["max_timesteps"])
        self.goal_radius = float(g["goal_radius"])
        self.collision_dist = float(g["collision_dist"])
        self.startup_timeout_s = float(g["startup_timeout_s"])
        self.speed_fast = float(g["speed_fast"])
        self.speed_slow = float(g["speed_slow"])
        self.gain = float(g["gain"])

        self.rng = np.random.default_rng(self.seed)
        self.goal = GOALS[int(self.rng.integers(len(GOALS)))]

        self.scan = None
        self.slam_cov = None
        self.slam_topic_pose = None
        self.truth = None
        self.odom = None
        self.start = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/pose", self.pose_cb, 10)
        self.create_subscription(PoseStamped, "/ground_truth_pose", self.truth_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(Float64MultiArray, "/episode_start_pose",
                                 self.start_cb, latched)
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        self.timestep = 0
        self.pending = None
        self.finished = False
        self.started = False
        self.belief_reached = 0
        self.start_wall = self.get_clock().now()

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(
            self.output_dir, f"{self.policy_name}_ep{self.episode:04d}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(self.COLUMNS)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            f"Episode {self.episode} (seed {self.seed}) goal={self.goal} "
            f"-> {self.csv_path}")

    def scan_cb(self, msg):
        r = np.asarray(msg.ranges, dtype=float)
        r = np.where(np.isfinite(r) & (r > 0.0), r, msg.range_max)
        n = len(r)
        if n == 0:
            return

        def sector_min(center_angle, width_deg=50):
            c = int(round((center_angle - msg.angle_min) / msg.angle_increment)) % n
            h = max(1, int(round(math.radians(width_deg) / 2 / abs(msg.angle_increment))))
            idx = [(c + k) % n for k in range(-h, h + 1)]
            return float(np.min(r[idx]))

        self.scan = (sector_min(0.0), sector_min(math.pi / 2), sector_min(-math.pi / 2))

    def pose_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.slam_topic_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        c = msg.pose.covariance
        self.slam_cov = (c[0], c[7], c[1], c[35])

    def truth_cb(self, msg):
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.truth = (msg.pose.position.x, msg.pose.position.y, yaw)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def start_cb(self, msg):
        if len(msg.data) >= 3:
            self.start = (msg.data[0], msg.data[1], msg.data[2])

    def slam_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            q = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return (t.transform.translation.x, t.transform.translation.y, yaw), "tf"
        except Exception:
            if self.slam_topic_pose is not None:
                return self.slam_topic_pose, "topic"
            return None, "none"

    def to_world(self, pose):
        sx, sy, syaw = self.start
        x, y, yaw = pose
        c, s = math.cos(syaw), math.sin(syaw)
        return (sx + c * x - s * y, sy + s * x + c * y, wrap(syaw + yaw))

    # Odometry starts far from the origin because wheel encoders accumulate across runs.
    def anchored(self, slam):
        if not hasattr(self, "slam0"):
            self.slam0 = slam
        x0, y0, yaw0 = self.slam0
        dx, dy = slam[0] - x0, slam[1] - y0
        c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
        rel = (c0 * dx - s0 * dy, s0 * dx + c0 * dy, wrap(slam[2] - yaw0))
        return self.to_world(rel)

    def dist_goal(self, pose):
        return math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])

    def heading_err(self, pose):
        return wrap(math.atan2(self.goal[1] - pose[1],
                               self.goal[0] - pose[0]) - pose[2])

    def choose_action(self, front, left, right, est):
        # Gain unused here, so both policies match and the importance ratio is 1.
        if front < 0.45:
            linear = 0.0
            mean = 0.8 if left > right else -0.8
            mode = "avoid"
        else:
            mean = float(np.clip(self.gain * self.heading_err(est), -1.0, 1.0))
            if left < 0.55:
                mean -= 0.5
            if right < 0.55:
                mean += 0.5
            linear = self.speed_fast if front > 0.6 else self.speed_slow
            mode = "goal_seek"

        angular = float(np.clip(mean + self.rng.normal(0.0, self.sigma), -0.9, 0.9))
        log_prob = (-0.5 * ((angular - mean) / self.sigma) ** 2
                    - math.log(self.sigma * math.sqrt(2 * math.pi)))
        return linear, mean, angular, log_prob, mode

    def publish_cmd(self, linear, angular):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x = float(linear)
        m.twist.angular.z = float(angular)
        self.cmd_pub.publish(m)

    def control_loop(self):
        if self.finished:
            return

        slam, source = self.slam_pose()
        ready = (self.scan is not None and self.truth is not None
                 and self.start is not None and slam is not None
                 and self.slam_cov is not None and self.odom is not None)

        if not ready:
            self.publish_cmd(0.0, 0.0)
            waited = (self.get_clock().now() - self.start_wall).nanoseconds * 1e-9
            if int(waited) % 5 == 0 and abs(waited - round(waited)) < 0.06:
                self.get_logger().info(
                    f"waiting: scan={self.scan is not None} truth={self.truth is not None} "
                    f"start={self.start is not None} slam={slam is not None} "
                    f"cov={self.slam_cov is not None} odom={self.odom is not None}")
            if waited > self.startup_timeout_s:
                self.get_logger().error("Startup timeout; aborting episode.")
                self.finish(abort=True)
            return

        if not self.started:
            self.started = True
            self.get_logger().info("All inputs present. Episode running.")

        est = self.anchored(slam)
        front, left, right = self.scan
        truth, odom, cov = self.truth, self.odom, self.slam_cov
        min_dist = min(front, left, right)
        true_d = self.dist_goal(truth)
        est_d = self.dist_goal(est)

        if est_d < self.goal_radius:
            self.belief_reached = 1

        if self.pending is not None:
            row, prev_true_d = self.pending
            terminated, reason = 0, ""
            reward = 10.0 * (prev_true_d - true_d) - 0.01
            if min_dist < 0.30:
                reward -= 0.05
            if min_dist < self.collision_dist:
                reward -= 10.0
                terminated, reason = 1, "collision"
            elif true_d < self.goal_radius:
                reward += 10.0
                terminated, reason = 1, "goal"
            # Separate from terminated: a time-limit cutoff still has future value.
            truncated = int(not terminated and self.timestep >= self.max_timesteps)

            row += [reward, terminated, truncated, reason, self.belief_reached,
                    slam[0], slam[1], slam[2], cov[0], cov[1], cov[2], cov[3],
                    est[0], est[1], est[2], truth[0], truth[1], truth[2],
                    odom[0], odom[1], odom[2], front, left, right,
                    est_d, true_d, self.heading_err(est)]
            self.writer.writerow(row)
            self.csv_file.flush()
            self.pending = None

            if terminated or truncated:
                self.get_logger().info(
                    f"Episode {self.episode} ended: "
                    f"{reason or 'time limit'} at step {self.timestep}")
                self.finish()
                return

        linear, mean, angular, log_prob, mode = self.choose_action(
            front, left, right, est)
        self.publish_cmd(linear, angular)

        self.pending = ([
            self.episode, self.timestep,
            self.get_clock().now().nanoseconds * 1e-9,
            self.goal[0], self.goal[1],
            self.start[0], self.start[1], self.start[2],
            slam[0], slam[1], slam[2], cov[0], cov[1], cov[2], cov[3],
            est[0], est[1], est[2], source,
            truth[0], truth[1], truth[2],
            odom[0], odom[1], odom[2],
            front, left, right,
            est_d, true_d,
            self.heading_err(est), self.heading_err(truth),
            linear, angular, mean, self.sigma, log_prob, mode,
        ], true_d)
        self.timestep += 1

    def finish(self, abort=False):
        self.finished = True
        self.timer.cancel()
        for _ in range(5):
            self.publish_cmd(0.0, 0.0)
        self.csv_file.flush()
        self.csv_file.close()
        if abort:
            os.replace(self.csv_path, self.csv_path + ".aborted")
        self.get_logger().info(f"Saved {self.csv_path}{'.aborted' if abort else ''}")


def main(args=None):
    rclpy.init(args=args)
    node = BehaviourPolicy()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.csv_file.closed:
            node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
