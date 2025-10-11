from controller import Supervisor
import math
import numpy as np
import csv
import os
from collections import defaultdict, deque
import random

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')
left_encoder.enable(timestep)
right_encoder.enable(timestep)
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

camera = robot.getDevice('camera')
camera.enable(timestep)
if camera.hasRecognition():
    camera.recognitionEnable(timestep)
else:
    print("Camera recognition not available. SLAM will rely on just odometry")

robot_node = robot.getSelf()
MY_RANDOM_SEED = 42
np.random.seed(MY_RANDOM_SEED)
random.seed(MY_RANDOM_SEED)

WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160
MAX_SPEED = 6.28

STATE_SIZE = 3
LM_SIZE = 2
DT = timestep / 1000.0

R_measurement = np.diag([0.10**2, (np.deg2rad(6.0))**2]) 
Q_process = np.diag([0.08**2, 0.08**2, (np.deg2rad(10.0))**2])
MAHALANOBIS_THRESHOLD = 5.99  

FORWARD_SPEED = 0.5
TURN_SPEED = 0.5
SAFE_FRONT = 0.6
SAFE_SIDE = 0.4
EMERGENCY_DISTANCE = 0.3

LANDMARK_CONFIDENCE_THRESHOLD = 3
MAX_LANDMARK_RANGE = 2.8

STUCK_VELOCITY_THRESHOLD = 0.10
STUCK_TIME_THRESHOLD = 8.0
POSITION_HISTORY_SIZE = 20
RECOVERY_BACKUP_TIME = 2.0
RECOVERY_TURN_ANGLE = 90

SIGMA_POLICY = 0.5  
POLICY_TYPE = "logging" if SIGMA_POLICY >= 1.0 else "evaluation"
print(f"RUNNING {POLICY_TYPE.upper()} POLICY (σ = {SIGMA_POLICY})")

COLLISION_DISTANCE_THRESHOLD = 0.25  
REWARD_COLLISION = -10.0 
REWARD_SAFE_MOVE=0.1
REWARD_SAFE = 0.0  


TURN_DIRECTION = 1
position_history = deque(maxlen=POSITION_HISTORY_SIZE)
stuck_timer = 0.0
recovery_mode = False
recovery_start_time = 0.0
recovery_stage = 0


LOG_FILE = f'slam_data_{POLICY_TYPE}_sigma{SIGMA_POLICY}.csv'

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'time',
            'est_x', 'est_y', 'est_yaw',
            'true_x', 'true_y', 'true_yaw',
            'pos_error', 'yaw_error',
            'pos_uncertainty', 'yaw_uncertainty',
            'P00', 'P01', 'P11',
            'normalized_pos_error', 'normalized_yaw_error',
            'landmarks_seen', 'landmarks_confident',
            # Controls (AFTER noise is applied)
            'v_cmd', 'omega_cmd',
            # Policy parameters
            'policy_type', 'sigma_policy',
            'nav_mode', 'min_distance', 'is_stuck', 'recovery_mode',
            # EKF consistency
            'avg_nis', 'nis_consistent',
            'reward', 'cumulative_reward',
            'landmark_updates'
        ]
        writer.writerow(header)
    print(f"Created log file: {LOG_FILE}")


def tidy(v, eps=1e-9):
    return 0.0 if abs(v) < eps else float(v)


def get_true_pose(robot_node):
    pos = robot_node.getPosition()
    orient = robot_node.getOrientation()
    up_vec = np.array([orient[2], orient[5], orient[8]])
    up_axis = int(np.argmax(np.abs(up_vec)))

    if up_axis == 1:
        true_x = pos[0]
        true_y = pos[2]
    else:
        true_x = pos[0]
        true_y = pos[1]

    forward = np.array([orient[0], orient[3], orient[6]])
    forward_proj = forward.copy()
    forward_proj[up_axis] = 0.0
    nx, ny, nz = forward_proj
    if up_axis == 1:
        yaw = math.atan2(nz, nx)
    else:
        yaw = math.atan2(ny, nx)

    return tidy(true_x), tidy(true_y), float(yaw)


class CorrectedEKFSLAM:
    def __init__(self):
        self.x = np.zeros((STATE_SIZE, 1))
        self.P = np.diag([0.001, 0.001, 0.001])
        self.landmarks = {}
        self.landmark_positions = {}
        self.observation_counts = defaultdict(int)
        self.total_updates = 0
        self.rejected_updates = 0
        self.nis_history = []
        self.current_timestep_updates = []  # Track updates in current timestep

    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    def predict(self, v, omega, dt):
        if dt <= 0 or np.any(np.isnan([v, omega])):
            return

        theta = float(self.x[2, 0])
        dx = v * dt * math.cos(theta)
        dy = v * dt * math.sin(theta)
        dtheta = omega * dt

        self.x[0, 0] += dx
        self.x[1, 0] += dy
        self.x[2, 0] = self.normalize_angle(self.x[2, 0] + dtheta)

        n = self.x.shape[0]
        F = np.eye(n)
        if n >= 3:
            F[0, 2] = -v * dt * math.sin(theta)
            F[1, 2] = v * dt * math.cos(theta)

        G_u = np.array([
            [dt * math.cos(theta), -0.5 * v * dt**2 * math.sin(theta)],
            [dt * math.sin(theta), 0.5 * v * dt**2 * math.cos(theta)],
            [0.0, dt]
        ])

        a1 = 0.30
        a2 = 0.30
        a3 = 0.40
        a4 = 0.60

        Q_v_omega_diag = [
            a1 * v**2 + a2 * omega**2,
            a3 * v**2 + a4 * omega**2
        ]
        Q_v_omega = np.diag(Q_v_omega_diag)
        Q_state = G_u @ Q_v_omega @ G_u.T
        Q = np.zeros((n, n))
        Q[:STATE_SIZE, :STATE_SIZE] = Q_state

        self.P[:n, :n] = F @ self.P[:n, :n] @ F.T + Q
        
        MIN_COV_POSE = 1e-6
        pose_indices = range(STATE_SIZE)
        np.fill_diagonal(
            self.P[np.ix_(pose_indices, pose_indices)],
            np.maximum(np.diag(self.P[np.ix_(pose_indices, pose_indices)]), MIN_COV_POSE)
        )

    def _ensure_positive_definite(self):
        self.P = (self.P + self.P.T) / 2.0
        try:
            eigvals = np.linalg.eigvalsh(self.P)
            min_eig = np.min(eigvals)
            if min_eig < -1e-6:
                shift = -min_eig + 1e-6
                self.P += np.eye(self.P.shape[0]) * shift
        except:
            pass

    def add_landmark(self, landmark_id, measurement):
        r, bearing = float(measurement[0]), float(measurement[1])
        x_r, y_r, theta_r = self.x[0, 0], self.x[1, 0], self.x[2, 0]

        lm_x = x_r + r * math.cos(theta_r + bearing)
        lm_y = y_r + r * math.sin(theta_r + bearing)

        idx = len(self.landmarks)
        self.landmarks[landmark_id] = idx
        self.landmark_positions[landmark_id] = np.array([lm_x, lm_y])

        self.x = np.vstack([self.x, np.array([[lm_x], [lm_y]])])

        J_x = np.array([
            [1.0, 0.0, -r * math.sin(theta_r + bearing)],
            [0.0, 1.0, r * math.cos(theta_r + bearing)]
        ])
        J_z = np.array([
            [math.cos(theta_r + bearing), -r * math.sin(theta_r + bearing)],
            [math.sin(theta_r + bearing), r * math.cos(theta_r + bearing)]
        ])
        P_rr = self.P[:STATE_SIZE, :STATE_SIZE]
        P_ll = J_x @ P_rr @ J_x.T + J_z @ R_measurement @ J_z.T

        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[:n_old, :n_old] = self.P
        P_new[n_old:n_old + 2, n_old:n_old + 2] = P_ll

        P_cross = P_rr @ J_x.T
        P_new[:STATE_SIZE, n_old:n_old + 2] = P_cross
        P_new[n_old:n_old + 2, :STATE_SIZE] = P_cross.T

        self.P = P_new
        self._ensure_positive_definite()
        # print(f"Added landmark {landmark_id} at ({lm_x:.2f}, {lm_y:.2f})")

    def get_landmark_state(self, landmark_id):
        if landmark_id not in self.landmarks:
            return None
        idx = self.landmarks[landmark_id]
        state_idx = STATE_SIZE + idx * LM_SIZE
        if state_idx + 1 >= self.x.shape[0]:
            return None
        return self.x[state_idx:state_idx + 2, 0]

    def calculate_expected_measurement(self, landmark_pos):
        dx = landmark_pos[0] - self.x[0, 0]
        dy = landmark_pos[1] - self.x[1, 0]
        r = math.sqrt(dx * dx + dy * dy)
        bearing = self.normalize_angle(math.atan2(dy, dx) - self.x[2, 0])
        return np.array([r, bearing])

    def get_measurement_jacobian(self, landmark_id):
        if landmark_id not in self.landmarks:
            return None
        landmark_pos = self.get_landmark_state(landmark_id)
        if landmark_pos is None:
            return None

        dx = landmark_pos[0] - self.x[0, 0]
        dy = landmark_pos[1] - self.x[1, 0]
        q = dx * dx + dy * dy
        if q < 1e-6:
            return None
        sqrt_q = math.sqrt(q)
        n = self.x.shape[0]
        H = np.zeros((2, n))

        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0.0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        lm_idx = self.landmarks[landmark_id]
        state_idx = STATE_SIZE + lm_idx * LM_SIZE
        if state_idx + 1 < n:
            H[0, state_idx] = dx / sqrt_q
            H[0, state_idx + 1] = dy / sqrt_q
            H[1, state_idx] = -dy / q
            H[1, state_idx + 1] = dx / q

        return H

    def reset_timestep_updates(self):
        self.current_timestep_updates = []

    def update_with_landmark(self, landmark_id, measurement, current_time):
        self.observation_counts[landmark_id] += 1

        if landmark_id not in self.landmarks:
            if measurement[0] < MAX_LANDMARK_RANGE:
                self.add_landmark(landmark_id, measurement)
            return None

        if self.observation_counts[landmark_id] < LANDMARK_CONFIDENCE_THRESHOLD:
            return None

        landmark_pos = self.get_landmark_state(landmark_id)
        if landmark_pos is None:
            return None

        z_expected = self.calculate_expected_measurement(landmark_pos)
        z_actual = np.array(measurement)
        innovation = z_actual - z_expected
        innovation[1] = self.normalize_angle(innovation[1])

        H = self.get_measurement_jacobian(landmark_id)
        if H is None:
            return None

        try:
            S = H @ self.P @ H.T + R_measurement
            nis = float(innovation.T @ np.linalg.inv(S) @ innovation)

            if nis > MAHALANOBIS_THRESHOLD:
                self.rejected_updates += 1
                self.current_timestep_updates.append({
                    'landmark_id': landmark_id,
                    'nis': nis,
                    'accepted': False
                })
                return nis

            K = self.P @ H.T @ np.linalg.inv(S)
            self.x += K @ innovation.reshape(-1, 1)
            self.x[2, 0] = self.normalize_angle(self.x[2, 0])

            I = np.eye(self.P.shape[0])
            I_KH = I - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R_measurement @ K.T
            self._ensure_positive_definite()

            self.total_updates += 1
            self.nis_history.append(nis)
            self.current_timestep_updates.append({
                'landmark_id': landmark_id,
                'nis': nis,
                'accepted': True
            })
            return nis
        except (np.linalg.LinAlgError, ValueError) as e:
            self.rejected_updates += 1
            return None

    def get_pose(self):
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

    def get_uncertainty(self):
        pos_cov = self.P[:2, :2]
        eigvals, _ = np.linalg.eigh(pos_cov)
        pos_uncertainty = math.sqrt(max(eigvals))
        yaw_uncertainty = math.sqrt(max(self.P[2, 2], 0.0))
        return pos_uncertainty, yaw_uncertainty

    def get_landmark_counts(self):
        total_seen = len(self.observation_counts)
        confident = sum(1 for c in self.observation_counts.values() 
                       if c >= LANDMARK_CONFIDENCE_THRESHOLD)
        return total_seen, confident

    def get_consistency_metrics(self):
        if len(self.nis_history) >= 5:
            recent = self.nis_history[-50:]
            avg_nis = float(np.mean(recent))
            nis_consistent = 1.0 <= avg_nis <= 2.5
        else:
            avg_nis = 0.0
            nis_consistent = True
        return avg_nis, nis_consistent


# Reward function
def calculate_reward(min_distance, nav_mode, is_stuck):
    if min_distance < COLLISION_DISTANCE_THRESHOLD:
        return REWARD_COLLISION
    
    if nav_mode == "emergency_stop" or is_stuck or "recovery" in nav_mode:
        return REWARD_COLLISION
        
    if nav_mode == "exploring" or "avoiding" in nav_mode:
        return REWARD_SAFE_MOVE 

    return REWARD_SAFE


# NAVIGATION 
def is_robot_stuck(current_pos, dt):
    global position_history, stuck_timer
    position_history.append(current_pos)
    if len(position_history) < 5:
        return False
    recent_positions = list(position_history)[-10:]
    max_distance = 0.0
    for i in range(len(recent_positions)):
        for j in range(i + 1, len(recent_positions)):
            d = math.hypot(recent_positions[i][0] - recent_positions[j][0],
                          recent_positions[i][1] - recent_positions[j][1])
            max_distance = max(max_distance, d)
    if max_distance < STUCK_VELOCITY_THRESHOLD * len(recent_positions) * dt:
        stuck_timer += dt
    else:
        stuck_timer = 0.0
    return stuck_timer > STUCK_TIME_THRESHOLD


def get_wheel_odometry(left_enc, right_enc, prev_left_enc, prev_right_enc):
    if prev_left_enc is None or prev_right_enc is None:
        return 0.0, 0.0
    d_left = (left_enc - prev_left_enc) * WHEEL_RADIUS
    d_right = (right_enc - prev_right_enc) * WHEEL_RADIUS
    dt = DT
    if dt == 0:
        return 0.0, 0.0
    v = (d_right + d_left) / (2.0 * dt)
    omega = (d_right - d_left) / (WHEEL_BASE * dt)
    v = np.clip(v, -3.0, 3.0)
    omega = np.clip(omega, -5.0, 5.0)
    return v, omega


def detect_landmarks(camera):
    observations = []
    if not camera.hasRecognition():
        return observations
    robot_node = robot.getSelf()
    n_objects = camera.getRecognitionNumberOfObjects()
    for i in range(n_objects):
        obj = camera.getRecognitionObjects()[i]
        landmark_id = obj.getId()
        lm_node = robot.getFromId(landmark_id)
        if lm_node is None:
            continue
        lm_pose = lm_node.getPose(robot_node)
        rel_x = lm_pose[3]
        rel_y = lm_pose[7]
        r = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        bearing = math.atan2(rel_y, rel_x)
        if 0.1 < r < MAX_LANDMARK_RANGE:
            r_noisy = r + np.random.normal(0, 0.10)
            bearing_noisy = bearing + np.random.normal(0, np.deg2rad(6.0))
            observations.append((landmark_id, np.array([r_noisy, bearing_noisy])))
    return observations


def recovery_behavior(current_time):
    global recovery_mode, recovery_start_time, recovery_stage, stuck_timer
    if not recovery_mode:
        recovery_mode = True
        recovery_start_time = current_time
        recovery_stage = 0
        print("Stuck detected, starting recovery")
        return -0.5, 0.0, "recovery_backup"
    
    elapsed = current_time - recovery_start_time
    if recovery_stage == 0:
        if elapsed < RECOVERY_BACKUP_TIME:
            return -0.5, 0.0, "recovery_backup"
        else:
            recovery_stage = 1
            print("Recovery:turning")
            return 0.0, TURN_SPEED, "recovery_turn"
    elif recovery_stage == 1:
        turn_time = math.radians(RECOVERY_TURN_ANGLE) / TURN_SPEED
        if elapsed < RECOVERY_BACKUP_TIME + turn_time:
            return 0.0, TURN_SPEED, "recovery_turn"
        else:
            recovery_stage = 2
            print("Recovery complete")
            recovery_mode = False
            stuck_timer = 0.0
            position_history.clear()
            return 0.2, 0.0, "recovery_resume"
    return 0.0, 0.0, "recovery_complete"


def collision_safe_navigation(lidar_ranges, current_pos, current_time):
    global TURN_DIRECTION, recovery_mode
    is_stuck_flag = is_robot_stuck(current_pos, DT)
    if is_stuck_flag or recovery_mode:
        return recovery_behavior(current_time)
    
    if len(lidar_ranges) == 0:
        return 0.0, 0.0, "no_lidar"
    
    n = len(lidar_ranges)
    front_wide = lidar_ranges[max(n // 2 - 15, 0):min(n // 2 + 15, n)]
    front_narrow = lidar_ranges[max(n // 2 - 5, 0):min(n // 2 + 5, n)]
    left_sector = lidar_ranges[min(n // 2 + 15, n - 1):min(n // 2 + 40, n)]
    right_sector = lidar_ranges[max(n // 2 - 40, 0):max(n // 2 - 15, 1)]
    
    front_wide_dist = min(front_wide) if len(front_wide) > 0 else float('inf')
    front_narrow_dist = min(front_narrow) if len(front_narrow) > 0 else float('inf')
    left_dist = min(left_sector) if len(left_sector) > 0 else float('inf')
    right_dist = min(right_sector) if len(right_sector) > 0 else float('inf')
    
    if front_narrow_dist < EMERGENCY_DISTANCE:
        return 0.0, 0.0, "emergency_stop"
    
    if front_wide_dist < SAFE_FRONT:
        v = 0.0
        TURN_DIRECTION = 1 if right_dist > left_dist else -1
        omega = TURN_DIRECTION * TURN_SPEED
        mode = "turning"
    else:
        v = FORWARD_SPEED
        omega = 0.0
        if left_dist < SAFE_SIDE:
            omega = -TURN_SPEED * 0.5
            mode = "avoiding_left"
        elif right_dist < SAFE_SIDE:
            omega = TURN_SPEED * 0.5
            mode = "avoiding_right"
        else:
            mode = "exploring"
    
    return v, omega, mode

robot_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])
robot_node.getField('rotation').setSFRotation([0, 1, 0, 0])
robot.step(timestep)

slam = CorrectedEKFSLAM()
prev_left_enc = None
prev_right_enc = None
start_time = robot.getTime()
cumulative_reward = 0.0

print(f"\nStarting EKF-SLAM with {POLICY_TYPE} policy (σ={SIGMA_POLICY})")


while robot.step(timestep) != -1:
    current_time = robot.getTime() - start_time
    slam.reset_timestep_updates()

    left_enc = left_encoder.getValue()
    right_enc = right_encoder.getValue()
    lidar_ranges = np.array(lidar.getRangeImage())

    v_odom, omega_odom = get_wheel_odometry(left_enc, right_enc, 
                                            prev_left_enc, prev_right_enc)
    prev_left_enc, prev_right_enc = left_enc, right_enc

    # EKF-SLAM predict step
    slam.predict(v_odom, omega_odom, DT)

    # EKF-SLAM update step (landmarks)
    landmark_observations = detect_landmarks(camera)
    for lid, meas in landmark_observations:
        slam.update_with_landmark(lid, meas, current_time)

    #current estimate
    est_x, est_y, est_yaw = slam.get_pose()
    current_pos = (est_x, est_y)

    v_base, omega_base, nav_mode = collision_safe_navigation(
        lidar_ranges, current_pos, current_time
    )

    # Applying the policy noise:
    v_cmd = v_base + np.random.normal(0, SIGMA_POLICY)
    omega_cmd = omega_base + np.random.normal(0, SIGMA_POLICY)

    v_left = (2 * v_cmd - omega_cmd * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    v_right = (2 * v_cmd + omega_cmd * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    v_left = max(min(v_left, MAX_SPEED), -MAX_SPEED)
    v_right = max(min(v_right, MAX_SPEED), -MAX_SPEED)
    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)

    true_x, true_y, true_yaw = get_true_pose(robot_node)

    pos_error = math.hypot(est_x - true_x, est_y - true_y)
    yaw_diff = slam.normalize_angle(est_yaw - true_yaw)
    yaw_error = abs(yaw_diff)

    pos_uncertainty, yaw_uncertainty = slam.get_uncertainty()
    landmarks_seen, landmarks_confident = slam.get_landmark_counts()
    avg_nis, nis_consistent = slam.get_consistency_metrics()

    norm_pos_err = pos_error / max(pos_uncertainty, 1e-6)
    norm_yaw_err = yaw_error / max(yaw_uncertainty, 1e-6)

    min_distance = float(np.min(lidar_ranges)) if len(lidar_ranges) > 0 else float('inf')
    is_stuck_flag = is_robot_stuck(current_pos, DT)

    # Reward calculation
    reward = calculate_reward(min_distance, nav_mode, is_stuck_flag)
    cumulative_reward += reward

    landmark_update_str = "|".join([
        f"L{u['landmark_id']}:{'Accepted' if u['accepted'] else 'Rejected'}"
        for u in slam.current_timestep_updates
    ]) if slam.current_timestep_updates else "none"

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        P00 = float(slam.P[0, 0]) if slam.P.shape[0] >= 2 else 0.0
        P01 = float(slam.P[0, 1]) if slam.P.shape[0] >= 2 else 0.0
        P11 = float(slam.P[1, 1]) if slam.P.shape[0] >= 2 else 0.0
        
        writer.writerow([
            f"{current_time:.3f}",
            est_x, est_y, est_yaw,
            true_x, true_y, true_yaw,
            pos_error, math.degrees(yaw_error),
            pos_uncertainty, math.degrees(yaw_uncertainty),
            P00, P01, P11,
            norm_pos_err, math.degrees(norm_yaw_err),
            landmarks_seen, landmarks_confident,
            v_cmd, omega_cmd,
            POLICY_TYPE, SIGMA_POLICY,
            nav_mode, min_distance, is_stuck_flag, recovery_mode,
            avg_nis, nis_consistent,
            reward, cumulative_reward,
            landmark_update_str
        ])

    # Console output for every 5 seconds
    if int(current_time * 1000) % 5000 < timestep:
        print(f"\n{'='*70}")
        print(f"Time: {current_time:.1f}s | Mode: {nav_mode} | Policy: {POLICY_TYPE} (σ={SIGMA_POLICY})")
        print(f"Position - Est: ({est_x:.2f}, {est_y:.2f}, {math.degrees(est_yaw):.1f}°)")
        print(f" - True: ({true_x:.2f}, {true_y:.2f}, {math.degrees(true_yaw):.1f}°)")
        print(f"Error - Pos: {pos_error:.3f}m | Heading: {math.degrees(yaw_error):.1f}°")
        print(f"Uncertainty- Pos: {pos_uncertainty:.3f}m | Heading: {math.degrees(yaw_uncertainty):.1f}°")
        print(f"Landmarks- Seen: {landmarks_seen} | Confident: {landmarks_confident}")
        print(f"EKF - Avg NIS: {avg_nis:.2f} | Consistent: {nis_consistent}")
        print(f"Reward- Current: {reward:.2f} | Cumulative: {cumulative_reward:.2f}")
        print(f"Safety- Min dist: {min_distance:.2f}m | Stuck: {is_stuck_flag}")
        
        if pos_uncertainty < 0.05 and pos_error > 0.2:
            print("Overconfident estimate (low uncertainty, high error)")
        if not nis_consistent:
            print(" NIS inconsistency detected. The filter could be over or under-confident)")

print(f"Simulation complete!")
print(f"Final cumulative reward: {cumulative_reward:.2f}")
print(f"Data logged to: {LOG_FILE}")
