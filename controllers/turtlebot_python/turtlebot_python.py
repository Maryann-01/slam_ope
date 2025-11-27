from controller import Supervisor
import math
import numpy as np
import csv
import os
from collections import defaultdict, deque
import random

SIGMA_POLICY = 0.4 
POLICY_TYPE = "evaluation" if SIGMA_POLICY < 0.6 else "logging"
MY_RANDOM_SEED = 42
np.random.seed(MY_RANDOM_SEED)
random.seed(MY_RANDOM_SEED)

# Robot parameters
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160
MAX_SPEED = 6.28
STATE_SIZE = 3
LM_SIZE = 2
Q_process_scale = [0.25, 0.25, 1.2, 1.8] 
R_measurement = np.diag([0.12**2, (np.deg2rad(4.0))**2]) 
MAHALANOBIS_THRESHOLD = 12.59  
LANDMARK_CONFIDENCE_THRESHOLD = 1  
MAX_LANDMARK_RANGE = 3.0
MIN_LANDMARK_RANGE = 0.15
FORWARD_SPEED = 0.20
TURN_SPEED = 0.45
EMERGENCY_DISTANCE = 0.30
SAFE_FRONT = 0.70
SAFE_SIDE = 0.45
MAX_LINEAR_ACCEL = 0.5
MAX_ANGULAR_ACCEL = 1.5
STUCK_VELOCITY_THRESHOLD = 0.10
STUCK_TIME_THRESHOLD = 20.0
POSITION_HISTORY_SIZE = 60
RECOVERY_BACKUP_TIME = 1.2
RECOVERY_TURN_ANGLE = 60
REWARD_WEIGHTS = {
    'new_landmark': 20.0,
    'landmark_reobservation': 2.0,
    'uncertainty_reduction': 8.0,
    'good_localization': 4.0,
    'exploration': 0.5,
    'stuck': -15.0,
    'collision': -30.0,
}

LOG_FILE = f'ekf_slam_{POLICY_TYPE}_sigma{SIGMA_POLICY}.csv'
CONSOLE_LOG_INTERVAL = 5.0
print(f"EKF-SLAM Implementation")
print(f"Policy: {POLICY_TYPE.upper()} (σ = {SIGMA_POLICY})")
print(f"Log file: {LOG_FILE}")

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
DT = timestep / 1000.0

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
if camera:
    camera.enable(timestep)
    if camera.hasRecognition():
        camera.recognitionEnable(timestep)
        print("Camera recognition enabled")
    else:
        camera = None
else:
    camera = None

try:
    robot_node = robot.getSelf()
    if robot_node:
        test_pos = robot_node.getPosition()
        USE_GROUND_TRUTH = test_pos is not None
        if USE_GROUND_TRUTH:
            print("Ground truth available")
except:
    robot_node = None
    USE_GROUND_TRUTH = False

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def get_true_pose():
    if not USE_GROUND_TRUTH or robot_node is None:
        return None, None, None
    
    try:
        pos = robot_node.getPosition()
        orient = robot_node.getOrientation()
        
        if pos is None or orient is None:
            return None, None, None
        
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

        return float(true_x), float(true_y), float(yaw)
        
    except Exception as e:
        print(f"Error getting true pose: {e}")
        return None, None, None

def get_wheel_odometry(left_enc, right_enc, prev_left_enc, prev_right_enc):
    if prev_left_enc is None or prev_right_enc is None:
        return 0.0, 0.0
    
    d_left = (left_enc - prev_left_enc) * WHEEL_RADIUS
    d_right = (right_enc - prev_right_enc) * WHEEL_RADIUS
    
    if DT == 0:
        return 0.0, 0.0
    
    v = (d_right + d_left) / (2.0 * DT)
    omega = (d_right - d_left) / (WHEEL_BASE * DT)
    
    # global step_count
    # if 'step_count' not in globals():
        # step_count = 0
    # step_count += 1
    # if step_count < 10:
        # print(f"Odom: v={v:.3f}, omega={omega:.3f}, d_left={d_left:.4f}, d_right={d_right:.4f}")
    
    v = np.clip(v, -2.0, 2.0)
    omega = np.clip(omega, -4.0, 4.0)
    
    return v, omega

class EKFSLAM:
    def __init__(self):
        self.x = np.zeros((STATE_SIZE, 1))
        self.P = np.diag([0.01, 0.01, np.deg2rad(10.0)**2])
        
        self.landmarks = {}
        self.landmark_positions = {}
        self.observation_counts = defaultdict(int)
        self.landmark_last_seen = {}
        
        self.total_updates = 0
        self.rejected_updates = 0
        self.new_landmarks_this_step = 0
        self.reobservations_this_step = 0
        self.nis_history = deque(maxlen=50)
        
        self.uncertainty_history = deque(maxlen=100)
        self.prev_uncertainty = 1.0
        self.uncertainty_reduced_this_step = False
    
    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))
    
    def predict(self, v, omega, dt):
        if dt <= 0 or np.any(np.isnan([v, omega])):
            return
        
        theta = float(self.x[2, 0])
        if abs(omega) < 1e-6:
            dx = v * dt * math.cos(theta)
            dy = v * dt * math.sin(theta)
            dtheta = 0.0
        else:
            dx = (v/omega) * (math.sin(theta + omega*dt) - math.sin(theta))
            dy = (v/omega) * (-math.cos(theta + omega*dt) + math.cos(theta))
            dtheta = omega * dt
        
        self.x[0, 0] += dx
        self.x[1, 0] += dy
        self.x[2, 0] = self.normalize_angle(self.x[2, 0] + dtheta)
        n = self.x.shape[0]
        F = np.eye(n)
        
        if abs(omega) < 1e-6:
            F[0, 2] = -v * dt * math.sin(theta)
            F[1, 2] = v * dt * math.cos(theta)
        else:
            F[0, 2] = (v/omega) * (math.cos(theta + omega*dt) - math.cos(theta))
            F[1, 2] = (v/omega) * (math.sin(theta + omega*dt) - math.sin(theta))
        
        G_u = np.array([
            [dt * math.cos(theta), 0.0],
            [dt * math.sin(theta), 0.0],
            [0.0, dt]
        ])
        a1, a2, a3, a4 = Q_process_scale
        Q_v_omega = np.diag([
            (a1 * abs(v) + 0.01)**2,
            (a3 * abs(v) + a4 * abs(omega) + 0.01)**2
        ])
        
        Q_state = G_u @ Q_v_omega @ G_u.T
        Q = np.zeros((n, n))
        Q[:STATE_SIZE, :STATE_SIZE] = Q_state
        
#Predict covariance
        self.P = F @ self.P @ F.T + Q
#minimum covariance
        MIN_COV = 1e-8
        for i in range(min(STATE_SIZE, n)):
            self.P[i, i] = max(self.P[i, i], MIN_COV)
        
        self._ensure_positive_definite()
    
    def _ensure_positive_definite(self):
        self.P = (self.P + self.P.T) / 2.0
        
        try:
            eigvals = np.linalg.eigvalsh(self.P)
            min_eig = np.min(eigvals)
            if min_eig < 1e-10:
                shift = -min_eig + 1e-8
                self.P += np.eye(self.P.shape[0]) * shift
        except:
            pass
    
    def add_landmark(self, landmark_id, measurement, current_time):
        r, bearing = float(measurement[0]), float(measurement[1])
        if r < MIN_LANDMARK_RANGE or r > MAX_LANDMARK_RANGE:
            return
        
        x_r, y_r, theta_r = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        
        lm_x = x_r + r * math.cos(theta_r + bearing)
        lm_y = y_r + r * math.sin(theta_r + bearing)
        
        idx = len(self.landmarks)
        self.landmarks[landmark_id] = idx
        self.landmark_positions[landmark_id] = np.array([lm_x, lm_y])
        self.landmark_last_seen[landmark_id] = current_time
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
        
# Augment covariance
        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[:n_old, :n_old] = self.P
        P_new[n_old:n_old + 2, n_old:n_old + 2] = P_ll
        
        P_cross = P_rr @ J_x.T
        P_new[:STATE_SIZE, n_old:n_old + 2] = P_cross
        P_new[n_old:n_old + 2, :STATE_SIZE] = P_cross.T
        
        self.P = P_new
        self._ensure_positive_definite()
        
        self.new_landmarks_this_step += 1
    
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
        
        if q < 1e-8:
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
        self.new_landmarks_this_step = 0
        self.reobservations_this_step = 0
        self.uncertainty_reduced_this_step = False
    
    def update_with_landmark(self, landmark_id, measurement, current_time):
        self.observation_counts[landmark_id] += 1
        
# New landmark addition
        if landmark_id not in self.landmarks:
            if measurement[0] < MAX_LANDMARK_RANGE:
                self.add_landmark(landmark_id, measurement, current_time)
            return None
        
# Update last seen time
        self.landmark_last_seen[landmark_id] = current_time
        
        if self.observation_counts[landmark_id] < LANDMARK_CONFIDENCE_THRESHOLD:
            return None
        
        self.reobservations_this_step += 1
        
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
                return nis
            K = self.P @ H.T @ np.linalg.inv(S)
            
           
            self.x += K @ innovation.reshape(-1, 1)
            self.x[2, 0] = self.normalize_angle(self.x[2, 0])
            
# Joseph form of updating covariance for numerical stability
            I_KH = np.eye(self.P.shape[0]) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R_measurement @ K.T
            
            self._ensure_positive_definite()
            
            self.total_updates += 1
            self.nis_history.append(nis)
            
            return nis
            
        except (np.linalg.LinAlgError, ValueError):
            self.rejected_updates += 1
            return None
    
    def get_pose(self):
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])
    
    def get_uncertainty(self):
        pos_cov = self.P[:2, :2]
        eigvals, _ = np.linalg.eigh(pos_cov)
        pos_uncertainty = math.sqrt(max(eigvals))
        yaw_uncertainty = math.sqrt(max(self.P[2, 2], 0.0))
        
        self.uncertainty_history.append(pos_uncertainty)
        
       
        if len(self.uncertainty_history) > 10:
            recent_avg = np.mean(list(self.uncertainty_history)[-10:])
            if recent_avg < self.prev_uncertainty * 0.92:
                self.uncertainty_reduced_this_step = True
            self.prev_uncertainty = recent_avg
        
        return pos_uncertainty, yaw_uncertainty
    
    def get_landmark_counts(self):
        total_seen = len(self.observation_counts)
        confident = sum(1 for c in self.observation_counts.values() 
                       if c >= LANDMARK_CONFIDENCE_THRESHOLD)
        return total_seen, confident
    
    def get_consistency_metrics(self):
        if len(self.nis_history) >= 5:
            avg_nis = float(np.mean(self.nis_history))
            nis_consistent = 0.5 <= avg_nis <= 5.0
        else:
            avg_nis = 0.0
            nis_consistent = True
        return avg_nis, nis_consistent

def detect_landmarks():
    observations = []
    
    if camera is None or not camera.hasRecognition():
        return observations
    
    n_objects = camera.getRecognitionNumberOfObjects()
    
    for i in range(n_objects):
        try:
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
            
            if MIN_LANDMARK_RANGE < r < MAX_LANDMARK_RANGE:
                r_noisy = r + np.random.normal(0, 0.08)
                bearing_noisy = bearing + np.random.normal(0, np.deg2rad(2.0))
                observations.append((landmark_id, np.array([r_noisy, bearing_noisy])))
        except:
            continue
    
    return observations

def calculate_reward(min_distance, nav_mode, is_stuck, v_smooth,
                    slam, pos_error, true_available):
    reward = 0.0
    
    if slam.new_landmarks_this_step > 0:
        reward += slam.new_landmarks_this_step * REWARD_WEIGHTS['new_landmark']
    
    if slam.reobservations_this_step > 0:
        reward += slam.reobservations_this_step * REWARD_WEIGHTS['landmark_reobservation']
    
    if slam.uncertainty_reduced_this_step:
        reward += REWARD_WEIGHTS['uncertainty_reduction']
    
    if true_available and pos_error is not None and pos_error < 0.25:
        reward += REWARD_WEIGHTS['good_localization']
    
    if min_distance > EMERGENCY_DISTANCE and abs(v_smooth) > 0.05:
        reward += REWARD_WEIGHTS['exploration']
    
    if is_stuck or nav_mode in ["recovery_backup", "recovery_turn"]:
        reward += REWARD_WEIGHTS['stuck']
    
    if min_distance < EMERGENCY_DISTANCE:
        reward += REWARD_WEIGHTS['collision']
    
    return reward

class VelocitySmoother:
    def __init__(self):
        self.last_linear = 0.0
        self.last_angular = 0.0
    
    def smooth(self, target_linear, target_angular, dt):
        max_accel_linear = MAX_LINEAR_ACCEL * dt
        max_accel_angular = MAX_ANGULAR_ACCEL * dt
        
        linear_diff = target_linear - self.last_linear
        if abs(linear_diff) > max_accel_linear:
            target_linear = self.last_linear + math.copysign(max_accel_linear, linear_diff)
        
        angular_diff = target_angular - self.last_angular
        if abs(angular_diff) > max_accel_angular:
            target_angular = self.last_angular + math.copysign(max_accel_angular, angular_diff)
        
        self.last_linear = target_linear
        self.last_angular = target_angular
        return target_linear, target_angular

velocity_smoother = VelocitySmoother()
position_history = deque(maxlen=POSITION_HISTORY_SIZE)
stuck_timer = 0.0

def is_robot_stuck(current_pos, dt):
    global position_history, stuck_timer
    
    position_history.append(current_pos)
    
    if len(position_history) < 30:
        return False
    
    recent = list(position_history)[-40:]
    max_distance = 0.0
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            d = math.hypot(recent[i][0] - recent[j][0],
                          recent[i][1] - recent[j][1])
            max_distance = max(max_distance, d)
    
    if max_distance < STUCK_VELOCITY_THRESHOLD:
        stuck_timer += dt
    else:
        stuck_timer = max(0.0, stuck_timer - dt)
    
    return stuck_timer > STUCK_TIME_THRESHOLD

recovery_mode = False
recovery_start_time = 0.0
recovery_stage = 0

def recovery_behavior(current_time):
    global recovery_mode, recovery_start_time, recovery_stage, stuck_timer
    
    if not recovery_mode:
        recovery_mode = True
        recovery_start_time = current_time
        recovery_stage = 0
        print("Stuck detected...starting recovery")
        return -0.15, 0.0, "recovery_backup"
    
    elapsed = current_time - recovery_start_time
    
    if recovery_stage == 0:
        if elapsed < RECOVERY_BACKUP_TIME:
            return -0.15, 0.0, "recovery_backup"
        else:
            recovery_stage = 1
            turn_dir = random.choice([-1, 1])
            return 0.0, turn_dir * TURN_SPEED * 0.8, "recovery_turn"
    
    elif recovery_stage == 1:
        turn_time = math.radians(RECOVERY_TURN_ANGLE) / (TURN_SPEED * 0.8)
        if elapsed < RECOVERY_BACKUP_TIME + turn_time:
            turn_dir = 1 if recovery_stage == 1 else -1
            return 0.0, turn_dir * TURN_SPEED * 0.8, "recovery_turn"
        else:
            recovery_stage = 2
            print("Recovery complete")
            recovery_mode = False
            stuck_timer = 0.0
            position_history.clear()
            return 0.10, 0.0, "recovery_resume"
    
    return 0.0, 0.0, "recovery_complete"

TURN_DIRECTION = 1

def collision_safe_navigation(lidar_ranges, current_pos, current_time):
    global TURN_DIRECTION, recovery_mode
    
    is_stuck_flag = is_robot_stuck(current_pos, DT)
    
    if is_stuck_flag or recovery_mode:
        return recovery_behavior(current_time)
    
    if len(lidar_ranges) == 0:
        return 0.0, 0.0, "no_lidar"
    
    n = len(lidar_ranges)
    
    front_narrow = lidar_ranges[max(n // 2 - 15, 0):min(n // 2 + 15, n)]
    front_wide = lidar_ranges[max(n // 2 - 45, 0):min(n // 2 + 45, n)]
    left_sector = lidar_ranges[min(n // 2 + 45, n - 1):min(n // 2 + 90, n)]
    right_sector = lidar_ranges[max(n // 2 - 90, 0):max(n // 2 - 45, 1)]
    
    front_narrow_dist = min(front_narrow) if len(front_narrow) > 0 else float('inf')
    front_wide_dist = min(front_wide) if len(front_wide) > 0 else float('inf')
    left_dist = min(left_sector) if len(left_sector) > 0 else float('inf')
    right_dist = min(right_sector) if len(right_sector) > 0 else float('inf')
    
    if front_narrow_dist < EMERGENCY_DISTANCE:
        return 0.0, 0.0, "emergency_stop"
    if front_wide_dist < SAFE_FRONT:
        v = 0.05
        TURN_DIRECTION = 1 if right_dist > left_dist else -1
        omega = TURN_DIRECTION * TURN_SPEED
        mode = "turning"
    else:
        v = FORWARD_SPEED
        omega = 0.0
       
        if left_dist < SAFE_SIDE:
            avoidance_gain = (SAFE_SIDE - left_dist) / SAFE_SIDE
            omega += -TURN_SPEED * 0.4 * avoidance_gain
            mode = "avoiding_left"
        
        if right_dist < SAFE_SIDE:
            avoidance_gain = (SAFE_SIDE - right_dist) / SAFE_SIDE
            omega += TURN_SPEED * 0.4 * avoidance_gain
            mode = "avoiding_right"
        
        if left_dist >= SAFE_SIDE and right_dist >= SAFE_SIDE:
            mode = "exploring"
    
    return v, omega, mode

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'time', 'est_x', 'est_y', 'est_yaw',
            'true_x', 'true_y', 'true_yaw',
            'pos_error', 'yaw_error_deg',
            'pos_uncertainty', 'yaw_uncertainty_deg',
            'normalized_pos_error', 'normalized_yaw_error',
            'landmarks_seen', 'landmarks_confident',
            'new_landmarks_step', 'reobservations_step',
            'v_smooth', 'omega_smooth', 'v_cmd', 'omega_cmd',
            'policy_type', 'sigma_policy',
            'nav_mode', 'min_distance', 'is_stuck',
            'avg_nis', 'nis_consistent',
            'reward', 'cumulative_reward'
        ]
        writer.writerow(header)
    print(f"Log file created: {LOG_FILE}\n")

slam = EKFSLAM()
prev_left_enc = None
prev_right_enc = None
start_time = robot.getTime()
cumulative_reward = 0.0
last_console_log = 0.0

print("Starting EKF-SLAM\n")
while robot.step(timestep) != -1:
    current_time = robot.getTime() - start_time
    slam.reset_timestep_updates()
    
    left_enc = left_encoder.getValue()
    right_enc = right_encoder.getValue()
    lidar_ranges = np.array(lidar.getRangeImage())
    
    v_odom, omega_odom = get_wheel_odometry(left_enc, right_enc,
                                            prev_left_enc, prev_right_enc)
    prev_left_enc, prev_right_enc = left_enc, right_enc
    
    slam.predict(v_odom, omega_odom, DT)
    
    landmark_observations = detect_landmarks()
    for lid, meas in landmark_observations:
        slam.update_with_landmark(lid, meas, current_time)
    
    est_x, est_y, est_yaw = slam.get_pose()
    current_pos = (est_x, est_y)
    
    v_base, omega_base, nav_mode = collision_safe_navigation(
        lidar_ranges, current_pos, current_time
    )
    
    v_smooth, omega_smooth = velocity_smoother.smooth(v_base, omega_base, DT)
    
    v_cmd = v_smooth + np.random.normal(0, SIGMA_POLICY * 0.08)
    omega_cmd = omega_smooth + np.random.normal(0, SIGMA_POLICY * 0.12)
    v_cmd = np.clip(v_cmd, -0.30, 0.30)
    omega_cmd = np.clip(omega_cmd, -0.8, 0.8)
    
    v_left = (2 * v_cmd - omega_cmd * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    v_right = (2 * v_cmd + omega_cmd * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    v_left = max(min(v_left, MAX_SPEED), -MAX_SPEED)
    v_right = max(min(v_right, MAX_SPEED), -MAX_SPEED)
    
    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)
    
    true_x, true_y, true_yaw = get_true_pose()
    
    if true_x is not None:
        pos_error = math.hypot(est_x - true_x, est_y - true_y)
        yaw_diff = normalize_angle(est_yaw - true_yaw)
        yaw_error = abs(yaw_diff)
        true_available = True
    else:
        pos_error = 0.0
        yaw_error = 0.0
        true_available = False
    
    pos_uncertainty, yaw_uncertainty = slam.get_uncertainty()
    landmarks_seen, landmarks_confident = slam.get_landmark_counts()
    avg_nis, nis_consistent = slam.get_consistency_metrics()
    
    norm_pos_err = pos_error / max(pos_uncertainty, 1e-6) if pos_uncertainty > 0 else 0.0
    norm_yaw_err = yaw_error / max(yaw_uncertainty, 1e-6) if yaw_uncertainty > 0 else 0.0
    
#reward calculation
    min_distance = float(np.min(lidar_ranges)) if len(lidar_ranges) > 0 else float('inf')
    is_stuck_flag = is_robot_stuck(current_pos, DT)
    
    reward = calculate_reward(min_distance, nav_mode, is_stuck_flag, v_smooth,
                             slam, pos_error, true_available)
    cumulative_reward += reward
    
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{current_time:.3f}",
            est_x, est_y, est_yaw,
            true_x if true_x is not None else 0.0,
            true_y if true_y is not None else 0.0,
            true_yaw if true_yaw is not None else 0.0,
            pos_error, math.degrees(yaw_error),
            pos_uncertainty, math.degrees(yaw_uncertainty),
            norm_pos_err, math.degrees(norm_yaw_err),
            landmarks_seen, landmarks_confident,
            slam.new_landmarks_this_step, slam.reobservations_this_step,
            v_smooth, omega_smooth, v_cmd, omega_cmd,
            POLICY_TYPE, SIGMA_POLICY,
            nav_mode, min_distance, is_stuck_flag,
            avg_nis, nis_consistent,
            reward, cumulative_reward
        ])
    
    if current_time - last_console_log >= CONSOLE_LOG_INTERVAL:
        last_console_log = current_time
        print(f"\n")
        print(f"Time: {current_time:.1f}s | Mode: {nav_mode} | Policy: {POLICY_TYPE}")
        print(f"Position:")
        print(f"Est:({est_x:.2f}, {est_y:.2f}, {math.degrees(est_yaw):.1f}°)")
        if true_x is not None:
            print(f"True: ({true_x:.2f}, {true_y:.2f}, {math.degrees(true_yaw):.1f}°)")
            print(f"Error: {pos_error:.3f}m pos, {math.degrees(yaw_error):.1f}° yaw")
        print(f"Uncertainty: {pos_uncertainty:.3f}m pos, {math.degrees(yaw_uncertainty):.1f}° yaw")
        print(f"Landmarks: {landmarks_seen} seen, {landmarks_confident} confident")
        print(f"Reward: {reward:.2f} | Cumulative: {cumulative_reward:.1f}")
        print(f"Safety: min dist = {min_distance:.2f}m | Stuck: {is_stuck_flag}")
        
        if pos_uncertainty > 0.5:
            print("High uncertainty")
        if not nis_consistent:
            print("NIS inconsistency")
        if slam.new_landmarks_this_step > 0:
            print(f"Found {slam.new_landmarks_this_step} new landmarks!")
        
        print()
print(f"Simulation complete!")
print(f"Final cumulative reward: {cumulative_reward:.2f}")
print(f"Data logged to: {LOG_FILE}")
print(f"Total landmarks: {landmarks_seen}")
print(f"Updates: {slam.total_updates} accepted, {slam.rejected_updates} rejected")
if true_x is not None:
    print(f"Final position error: {pos_error:.3f}m")