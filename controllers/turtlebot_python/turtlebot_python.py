from controller import Supervisor
import math
import numpy as np
import csv
import os

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Wheel encoders
left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Lidar
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

# Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
if camera.hasRecognition():
    camera.recognitionEnable(timestep)
else:
    print("Camera recognition is not available on this device.")

# Constants
WHEEL_RADIUS = 0.033    # meters
WHEEL_BASE = 0.160      # meters
MAX_SPEED = 6.28        # rad/s

# Control parameters
FORWARD_SPEED = 3.0
TURN_SPEED = 2.0

# Logging setup
LOG_FILE = 'basic_slam_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['time', 'x', 'y', 'theta', 'v', 'omega',
                  'left_enc', 'right_enc', 'true_x', 'true_y', 'true_theta']
        writer.writerow(header)

x_est = np.array([0.0, 0.0, 0.0])
P = np.eye(3) * 1e-3

Q = np.array([[0.01 ** 2, 0],
              [0, (math.radians(2)) ** 2]])

R_landmark = np.diag([0.1 ** 2, math.radians(5) ** 2])

landmarks = {}
landmark_initialized = set()

prev_left_enc = None
prev_right_enc = None

start_time = robot.getTime()


robot_node = robot.getSelf()

# Resetting robot position and orientation
robot_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])
robot_node.getField('rotation').setSFRotation([0, 1, 0, 0])
robot.step(timestep)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def odometry_update(x_prev, v, omega, dt):
    x, y, theta = x_prev
    if abs(omega) < 1e-5:
        dx = v * dt * math.cos(theta)
        dy = v * dt * math.sin(theta)
        dtheta = 0.0
    else:
        dx = (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
        dy = (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
        dtheta = omega * dt
    return np.array([x + dx, y + dy, wrap_angle(theta + dtheta)])


def jacobian_F(x, v, omega, dt):
    _, _, theta = x
    if abs(omega) < 1e-5:
        return np.array([[1, 0, -v * dt * math.sin(theta)],
                         [0, 1, v * dt * math.cos(theta)],
                         [0, 0, 1]])
    else:
        r = v / omega
        omega_dt = omega * dt
        return np.array([[1, 0, r * (math.cos(theta + omega_dt) - math.cos(theta))],
                         [0, 1, r * (math.sin(theta + omega_dt) - math.sin(theta))],
                         [0, 0, 1]])


def jacobian_V(x, v, omega, dt):
    _, _, theta = x
    if abs(omega) < 1e-5:
        return np.array([[dt * math.cos(theta), 0],
                         [dt * math.sin(theta), 0],
                         [0, dt]])
    else:
        r = v / omega
        omega_dt = omega * dt
        return np.array([
            [(math.sin(theta + omega_dt) - math.sin(theta)) / omega,
             v * (math.sin(theta) - math.sin(theta + omega_dt)) / (omega ** 2) + v * dt * math.cos(theta + omega_dt) / omega],
            [(-math.cos(theta + omega_dt) + math.cos(theta)) / omega,
             v * (math.cos(theta + omega_dt) - math.cos(theta)) / (omega ** 2) + v * dt * math.sin(theta + omega_dt) / omega],
            [0, dt]])


def get_wheel_odometry(left_enc, right_enc, prev_left_enc, prev_right_enc):
    if prev_left_enc is None or prev_right_enc is None:
        return 0.0, 0.0
    d_left = (left_enc - prev_left_enc) * WHEEL_RADIUS
    d_right = (right_enc - prev_right_enc) * WHEEL_RADIUS
    dt = timestep / 1000.0
    if dt == 0:
        return 0.0, 0.0
    v = (d_right + d_left) / (2.0 * dt)
    omega = (d_right - d_left) / (WHEEL_BASE * dt)
    return v, omega


def ekf_predict(x, P, v, omega, Q, dt):
    F = jacobian_F(x, v, omega, dt)
    V = jacobian_V(x, v, omega, dt)
    x_pred = odometry_update(x, v, omega, dt)
    P_pred = F @ P @ F.T + V @ Q @ V.T
    return x_pred, P_pred


def landmark_measurement_model(x, lm_pos):
    dx = lm_pos[0] - x[0]
    dy = lm_pos[1] - x[1]
    r = math.sqrt(dx * dx + dy * dy)
    bearing = wrap_angle(math.atan2(dy, dx) - x[2])
    return np.array([r, bearing])


def jacobian_H(x, lm_pos):
    dx = lm_pos[0] - x[0]
    dy = lm_pos[1] - x[1]
    q = dx * dx + dy * dy
    sqrt_q = math.sqrt(q)
    H = np.zeros((2, 3))
    H[0, 0] = -dx / sqrt_q
    H[0, 1] = -dy / sqrt_q
    H[0, 2] = 0
    H[1, 0] = dy / q
    H[1, 1] = -dx / q
    H[1, 2] = -1
    return H


def ekf_update(x, P, z, R, lm_pos):
    z_pred = landmark_measurement_model(x, lm_pos)
    H = jacobian_H(x, lm_pos)
    y = z - z_pred
    y[1] = wrap_angle(y[1])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_upd = x + K @ y
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = (np.eye(len(x)) - K @ H) @ P
    return x_upd, P_upd


TURN_DIRECTION = 1


def control_strategy(ranges):
    if len(ranges) == 0:
        return 0.0, 0.0
    n = len(ranges)
    front_sector = ranges[max(n // 2 - 10, 0):min(n // 2 + 10, n)]
    left_sector = ranges[min(n // 2 + 10, n - 1):min(n // 2 + 30, n)]
    right_sector = ranges[max(n // 2 - 30, 0):max(n // 2 - 10, 0)]
    front_dist = min(front_sector) if len(front_sector) > 0 else float('inf')
    left_dist = min(left_sector) if len(left_sector) > 0 else float('inf')
    right_dist = min(right_sector) if len(right_sector) > 0 else float('inf')
    SAFE_FRONT = 0.6
    SAFE_SIDE = 0.4
    if front_dist < SAFE_FRONT:
        v = 0.1
        omega = TURN_DIRECTION * TURN_SPEED
        return v, omega
    else:
        v = FORWARD_SPEED
        omega = 0.0
        if left_dist < SAFE_SIDE:
            omega = -TURN_SPEED * 0.4
        elif right_dist < SAFE_SIDE:
            omega = TURN_SPEED * 0.4
        return v, omega


def detect_landmarks(camera):
    observations = []
    if camera is None or not camera.hasRecognition():
        return observations
    n_obj = camera.getRecognitionNumberOfObjects()
    robot_node_for_obj = robot.getSelf()  
    for i in range(n_obj):
        obj = camera.getRecognitionObjects()[i]
        lm_id = obj.get_id()
        lm_node = robot.getFromId(lm_id)
        if lm_node is None:
            continue
        lm_pose = lm_node.getPose(robot_node_for_obj)
        lm_pos = np.array([lm_pose[3], lm_pose[7]])
        r = np.linalg.norm(lm_pos)
        bearing = math.atan2(lm_pos[1], lm_pos[0])
        observations.append((lm_id, np.array([r, bearing])))
    return observations


while robot.step(timestep) != -1:
    time_now = robot.getTime() - start_time
    left_enc = left_encoder.getValue()
    right_enc = right_encoder.getValue()
    lidar_ranges = np.array(lidar.getRangeImage()) if lidar else np.array([])

    # Get wheel odometry
    v, omega = get_wheel_odometry(left_enc, right_enc, prev_left_enc, prev_right_enc)
    prev_left_enc, prev_right_enc = left_enc, right_enc

    # EKF predict step
    x_est, P = ekf_predict(x_est, P, v, omega, Q, timestep / 1000.0)

    # Handle NaN states
    if np.any(np.isnan(x_est)):
        print("Warning: SLAM state diverged to NaN. Resetting pose.")
        x_est = np.array([0.0, 0.0, 0.0])
        P = np.eye(3) * 1e-3
        landmarks.clear()
        landmark_initialized.clear()

    # Landmark detection
    observations = []
    if camera.hasRecognition():
        observations = detect_landmarks(camera)

    # Process landmarks
    for lm_id, z in observations:
        if lm_id not in landmarks:
            # Initialize new landmark
            lm_x = x_est[0] + z[0] * math.cos(z[1] + x_est[2])
            lm_y = x_est[1] + z[0] * math.sin(z[1] + x_est[2])
            landmarks[lm_id] = np.array([lm_x, lm_y])
            landmark_initialized.add(lm_id)

        # EKF update with landmark
        lm_pos = landmarks[lm_id]
        x_est, P = ekf_update(x_est, P, z, R_landmark, lm_pos)

    # Control strategy
    v_cmd, omega_cmd = control_strategy(lidar_ranges)

    # Converting to wheel velocities
    v_left = (2 * v_cmd - omega_cmd * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    v_right = (2 * v_cmd + omega_cmd * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    v_left = max(min(v_left, MAX_SPEED), -MAX_SPEED)
    v_right = max(min(v_right, MAX_SPEED), -MAX_SPEED)

    # Set motor velocities
    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)

    # Get TRUE position and orientation
    true_pos = robot_node.getPosition()
    true_rot = robot_node.getOrientation()  # Rotation matrix (9 elements)
    

    if int(time_now * 10) % 30 == 0:  # Every 3 seconds
        print(f"Debug - Raw Webots: pos=({true_pos[0]:.3f}, {true_pos[1]:.3f}, {true_pos[2]:.3f})")
        print(f"Debug - Raw Webots: rot=({true_rot[0]:.3f}, {true_rot[1]:.3f}, {true_rot[2]:.3f}, {true_rot[3]:.3f}, {true_rot[4]:.3f}, {true_rot[5]:.3f}, {true_rot[6]:.3f}, {true_rot[7]:.3f}, {true_rot[8]:.3f})")
    
   
    true_x = true_pos[0]           
    true_y = true_pos[1]          

    # Trying different yaw calculations to find the correct one
    yaw1 = math.atan2(true_rot[6], true_rot[0])  
    yaw2 = math.atan2(true_rot[3], true_rot[0])  
    yaw3 = math.atan2(true_rot[1], true_rot[4])  
    yaw4 = math.atan2(-true_rot[2], true_rot[8]) 
    
    true_yaw = yaw1  # math.atan2(true_rot[6], true_rot[0])
    
    # Debugging calculations
    if int(time_now * 10) % 30 == 0:  # Every 3 seconds
        print(f"Debug - Yaw options: {math.degrees(yaw1):.1f}°, {math.degrees(yaw2):.1f}°, {math.degrees(yaw3):.1f}°, {math.degrees(yaw4):.1f}°")
        print(f"Debug - Control: v_cmd={v_cmd:.2f}, omega_cmd={omega_cmd:.2f}")

    # Log data
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            time_now,
            x_est[0], x_est[1], x_est[2],  # EKF estimate
            v_cmd, omega_cmd,
            left_enc, right_enc,
            true_x, true_y, true_yaw  # True position (x, y, theta)
        ])


    if int(time_now * 1000) % 1000 < timestep:
        print(f"Time {time_now:.2f}s | "
              f"TRUE: x={true_x:.2f} y={true_y:.2f} θ={math.degrees(true_yaw):.1f}° | "
              f"EKF: x={x_est[0]:.2f} y={x_est[1]:.2f} θ={math.degrees(x_est[2]):.1f}°")