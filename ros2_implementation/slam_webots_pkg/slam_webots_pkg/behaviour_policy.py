import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import csv
from tf_transformations import euler_from_quaternion
from std_srvs.srv import Empty

SIGMA = 0.3
POLICY_NAME = 'behaviour_a'

class BehaviourPolicy(Node):
    def __init__(self):
        super().__init__('behaviour_policy')
        self.front_dist = 10.0
        self.left_dist = 10.0
        self.right_dist = 10.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.cov_xx = 0.0
        self.cov_yy = 0.0
        self.cov_yaw = 0.0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.current_reward = 0.0
        self.episode_number = 1
        self.timestep = 0
        self.max_timesteps = 900
        self.last_chosen_dir = None
        self.time_in_area = 0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        self.slam_reset_client = self.create_client(Empty, '/slam_toolbox/clear')
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.csv_file = open(f'{POLICY_NAME}_sigma{SIGMA}.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestep', 'episode',
            'x', 'y', 'yaw',
            'cov_xx', 'cov_yy', 'cov_yaw',
            'front_dist', 'left_dist', 'right_dist',
            'action_linear', 'action_angular',
            'reward'
        ])

    def get_sector_min(self, ranges, center_idx, width=60):
        half = width // 2
        n = len(ranges)
        start = (center_idx - half) % n
        end = (center_idx + half) % n
        if start < end:
            sector = ranges[start:end]
        else:
            sector = np.concatenate([ranges[start:], ranges[:end]])
        return float(np.nanmin(sector)) if len(sector) > 0 else 10.0

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.front_dist = self.get_sector_min(ranges, 180, width=50)
        #self.left_dist = self.get_sector_min(ranges, 270, width=60)
        #self.right_dist = self.get_sector_min(ranges, 90, width=60)
        self.left_dist = self.get_sector_min(ranges, 90, width=60)
        self.right_dist = self.get_sector_min(ranges, 270, width=60)
    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.cov_xx = msg.pose.covariance[0]
        self.cov_yy = msg.pose.covariance[7]
        self.cov_yaw = msg.pose.covariance[35]

    def compute_reward(self):
        min_dist = min(self.front_dist, self.left_dist, self.right_dist)
        progress = 0.8 * self.current_linear_vel
        if min_dist < 0.20:
            safety = -20.0
        elif min_dist < 0.40:
            safety = -5.0
        else:
            safety = 0.0
        smoothness = -0.02 * abs(self.current_angular_vel)
        return progress + safety + smoothness

    def control_loop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        if self.front_dist < 0.25:
            cmd.twist.linear.x = -0.1
            cmd.twist.angular.z = 0.3 if self.left_dist > self.right_dist else -0.3
            self.last_chosen_dir = None
            self.publisher.publish(cmd)
            self.current_linear_vel = cmd.twist.linear.x
            self.current_angular_vel = cmd.twist.angular.z
            self.current_reward = self.compute_reward()
            self.log_row()
            self.timestep += 1
            if self.timestep >= self.max_timesteps:
                self.reset_episode()
            return

        scores = {
            'front': self.front_dist,
            'left': self.left_dist,
            'right': self.right_dist
        }
        if self.last_chosen_dir in scores:
            scores[self.last_chosen_dir] += 0.15 * max(scores.values())
        best_dir = max(scores, key=scores.get)
        noise = np.random.normal(0, SIGMA)

        if best_dir == 'front' and self.front_dist > 0.6:
            cmd.twist.linear.x = 0.2
            cmd.twist.angular.z = noise * 0.1
        elif best_dir == 'left':
            cmd.twist.linear.x = 0.1
            cmd.twist.angular.z = 0.5 + noise * 0.1
        else:
            cmd.twist.linear.x = 0.1
            cmd.twist.angular.z = -0.5 + noise * 0.1

        self.last_chosen_dir = best_dir
        self.time_in_area += 1
        if self.time_in_area > 80:
            cmd.twist.angular.z = float(np.random.uniform(1.0, 2.0)) * int(np.random.choice([-1, 1]))
            cmd.twist.linear.x = 0.0
            self.time_in_area = 0
            self.last_chosen_dir = None

        self.publisher.publish(cmd)
        self.current_linear_vel = cmd.twist.linear.x
        self.current_angular_vel = cmd.twist.angular.z
        self.current_reward = self.compute_reward()
        self.log_row()
        self.timestep += 1
        if self.timestep >= self.max_timesteps:
            self.reset_episode()

    def log_row(self):
        self.csv_writer.writerow([
            self.timestep,
            self.episode_number,
            self.current_x,
            self.current_y,
            self.current_yaw,
            self.cov_xx,
            self.cov_yy,
            self.cov_yaw,
            self.front_dist,
            self.left_dist,
            self.right_dist,
            self.current_linear_vel,
            self.current_angular_vel,
            self.current_reward
        ])

    def reset_episode(self):
        if self.episode_number >= 50:
            self.get_logger().info('All 50 episodes complete. Shutting down.')
            self.csv_file.close()
            rclpy.shutdown()
            return
        self.episode_number += 1
        self.timestep = 0
        self.last_chosen_dir = None
        self.time_in_area = 0
        self.get_logger().info(
            f'Episode {self.episode_number - 1} complete. Starting episode {self.episode_number}'
        )
        if self.slam_reset_client.wait_for_service(timeout_sec=1.0):
            self.slam_reset_client.call_async(Empty.Request())
            self.get_logger().info('SLAM map cleared')
        else:
            self.get_logger().warn('SLAM reset service not available')

def main(args=None):
    rclpy.init(args=args)
    node = BehaviourPolicy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
