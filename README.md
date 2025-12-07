#Offline Policy Evaluation for Mobile Robots with SLAM
##Project Overview
This project implements and compares offline policy evaluation (OPE) methods for mobile robot navigation under localization uncertainty. Using data logged from a TurtleBot3 Burger robot running EKF-SLAM in the Webots simulator, we predict how well a target control policy would perform without requiring real-world deployment.

###Question:
Can we predict the performance of a low-noise policy (policy=0.4) using only logged data from a high-noise exploratory policy (policy=0.8)?

###Key findings:
-Model-Based OPE (Neural Network): 7.1% average error (range: 4.4% to 62% across seeds)
-Importance Sampling: 9.4% error (consistent across runs)


#System Architecture
1. Simulation Environment
   - Platform: Webots R2023a
   - Robot: TurtleBot3 Burger
   - Sensors:
     - LiDAR (360° laser range finder)
     - Wheel encoders (odometry)
    
2.  ###**SLAM Implementation**
   - **Algorithm**: Extended Kalman Filter (EKF) SLAM
   - **State**: Robot pose (x, y, theta) + landmark positions
   - **Uncertainty Tracking**: Covariance matrices for state estimates
   - **Features:**
     - Real-time localization
     - Landmark detection and tracking 
     - Uncertainty quantification
       
3. ###**Control Policies**
   Two Gaussian noise policies tested:
   | Policy               | Noise Level     | Purpose                        | Timesteps | Cummulative Reward |
   |----------------------|-----------------|--------------------------------|-----------|--------------------|
   |Behaviour (policy=0.8)|High exploration |Data collection, Model Training |4885       | 14,424             |
   |Target (policy=0.4)   |Low exploration  |Evalauation Target, Validation  |4896       | 17,936             |

4. ##OPE Methods Implemented
   ###Model-Based OPE
   - Dynamics Model: f(state, action) → next_state
   - Reward Model: g(state) → reward
   - Implementation: Random Forest and Neural Networks
   - Process: Simulate target policy performance using learned models
  
##Data Format
###CSV Columns
###State Features:
`est_x`, `est_y`, `est_yaw` :EKF-SLAM estimated robot pose
`pos_uncertainty`: Position covariance magnitude
`yaw_uncertainty_deg`: Orientation uncertainty (degrees)
`pos_error` :Ground truth localization error
`landmarks_seen` : Number of detected landmarks
`landmarks_confident` : High-confidence landmark count
`min_distance` :Distance to nearest obstacle

    
