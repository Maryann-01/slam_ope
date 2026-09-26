# Offline Policy Evaluation for Mobile Robot Navigation under SLAM Uncertainty


## Research Question

How does SLAM-induced localization uncertainty affect the reliability of offline policy evaluation (OPE) methods when applied to mobile robot navigation? Specifically, can we accurately predict how a target navigation policy will perform from logged data of a behaviour policy, when the robot's state estimates carry uncertainty from SLAM?

## Approach

The project studies how SLAM uncertainty (represented by pose covariance matrices) affects both the accuracy of OPE predictions and the calibration of confidence in those predictions. We compare importance sampling and model-based OPE methods across different uncertainty levels.

## System Setup

- **Simulator**: Webots R2025a  
- **Robot**: TurtleBot3 Burger with 360° LiDAR  
- **SLAM**: slam_toolbox (ROS2 Jazzy)  
- **Middleware**: ROS2 Jazzy  
- **OS**: Ubuntu 24.04 (WSL2)

## Repository Structure

~~~
WebotsProject/
├── ros2_implementation/        Current ROS2 + slam_toolbox version
│   └── slam_webots_pkg/
│       ├── launch/             ROS2 launch files
│       ├── config/             slam_toolbox parameters
│       └── slam_webots_pkg/
│           └── behaviour_policy.py
├── worlds/                     Webots simulation worlds
├── libraries/                  Webots libraries
├── plugins/                    Webots plugins
├── protos/                     Webots protos
└── archive/                    Pre-ROS2 work (EKF-SLAM implementation)
~~~

## Behaviour Policy

A reactive navigation policy that selects directions based on LiDAR distances, with Gaussian noise (σ = 0.3) added to angular velocity to make the policy stochastic. This stochasticity is required for importance sampling to be applicable.

## Data Logging

Each timestep logs:
- Episode and timestep indices
- SLAM-estimated pose (x, y, yaw)
- Pose covariance (cov_xx, cov_yy, cov_yaw)
- LiDAR distances (front, left, right sectors)
- Actions (linear and angular velocity)
- Reward

Rewards combine forward progress, safety penalties near obstacles, and a smoothness term.

## Status

- [x] Behaviour policy implemented in ROS2
- [x] 50 episodes of trajectory data collected with SLAM pose estimates and covariances
- [x] Importance sampling pipeline (in progress)
- [ ] Model-based OPE pipeline
- [ ] Target policy validation in simulation
- [ ] Comparative analysis across SLAM uncertainty levels

## How to Run

1. Open Webots and load `worlds/turtlebot_slam.wbt`
2. Press play in Webots
3. In a separate terminal:
```bash
   cd ~/ros2_ws
   source install/setup.bash
   ros2 launch slam_webots_pkg slam_webots.launch.py
```

Data logs to `behaviour_a_sigma0.3.csv` in the workspace directory.

## Documentation

The `archive/` folder contains the earlier EKF-SLAM implementation that was superseded by this ROS2 + slam_toolbox version.
