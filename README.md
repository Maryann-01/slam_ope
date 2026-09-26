# Offline Policy Evaluation for Mobile Robot Navigation under SLAM Uncertainty

 
## Research Question
 
How does SLAM-induced localization uncertainty affect the reliability of offline policy evaluation (OPE) methods when applied to mobile robot navigation? Specifically, can we accurately predict how a target navigation policy will perform from logged data of a behaviour policy, when the robot's state estimates carry uncertainty from SLAM?
 
## Approach

The robot records a SLAM position estimate, typically off by about a metre and by over three metres in the worst episodes. Reward and termination depend on where the robot actually was, which never appears in the logged data.

The design keeps those two separate on purpose:

- **The policy conditions on the SLAM estimate**, so a biased estimate produces suboptimal actions.
- **Reward and termination are computed from ground truth**, so a mislocalised robot cannot be rewarded for reaching a goal it never reached.

Without that split, localisation error is either irrelevant, because a policy steering on obstacle sensors alone never consults the estimate, or invisible, because a robot scored on its own estimate collects reward for a goal it never reached.
 
## System Setup
 
- **Simulator**: Webots R2025a  
- **Robot**: TurtleBot3 Burger with LDS-01 LiDAR (5 Hz, 3.5 m range)  
- **SLAM**: slam_toolbox, synchronous mode  
- **Middleware**: ROS2 Jazzy  
- **OS**: Ubuntu 24.04 (WSL2)  
- **Control rate**: 10 Hz
## Repository Structure
 
~~~
WebotsProject/
├── ros2_implementation/
│   ├── run_episodes.sh              Per-episode runner with automatic retry
│   ├── validate_dataset.py          Seven-check dataset validator
│   └── slam_webots_pkg/
│       ├── launch/                  One-episode launch, shuts down on exit
│       ├── config/                  slam_toolbox parameters
│       ├── resource/                Robot descriptions
│       └── slam_webots_pkg/
│           ├── behaviour_policy.py  Policy, reward and episode logging
│           └── reset_plugin.py      Webots supervisor: teleport + ground truth
├── worlds/                          Webots simulation worlds
├── libraries/                       Webots libraries
├── plugins/                         Webots plugins
├── protos/                          Webots protos
└── archive/                         Pre-ROS2 work (EKF-SLAM implementation)
~~~
 
## Episode Independence
 
OPE estimators assume episodes are independent and identically distributed. slam_toolbox's own reset service left the node unresponsive, so independence is achieved by relaunching the entire stack once per episode:
 
- Each episode starts a fresh slam_toolbox process with an empty map
- The robot is teleported to a fixed nominal start pose plus Gaussian noise (0.10 m in x and y, 0.10 rad in yaw), seeded by episode number
- Ground-truth pose is published every simulation step by the supervisor plugin
- Failed episodes are retried automatically
## Policies
 
Both policies share the same form and differ only in steering gain.
 
~~~
heading_err = angle to goal, computed from the ESTIMATED pose
mean        = clip(gain * heading_err, -1, +1)
              - 0.5 if left_dist  < 0.55
              + 0.5 if right_dist < 0.55
action      ~ Normal(mean, 0.30), clipped to +/-0.9
~~~
 
| | Gain | Sigma |
|---|---|---|
| Behaviour policy | 1.5 | 0.30 |
| Target policy | 3.0 | 0.30 |
 
Sigma is identical in both so their action distributions share the same support, which importance sampling requires.
 
## Task and Reward
 
The robot drives to a goal sampled per episode from five validated free-space points. The horizon is 500 steps (50 s).
 
~~~
r = 10 * (reduction in TRUE distance to goal)
    - 0.01                             time cost
    - 0.05   if min lidar < 0.30 m     proximity penalty
    + 10     if true distance < 0.75 m -> terminated, "goal"
    - 10     if min lidar < 0.15 m     -> terminated, "collision"
~~~
 
Collision is defined by a LiDAR proximity threshold rather than physical contact. At 0.15 m with a 0.138 m wide robot this corresponds to roughly 0.08 m clearance from the chassis.
 
## Data Logging
 
Each timestep logs 65 columns:
 
- Episode, timestep, wall clock time, goal position, sampled start pose
- SLAM pose in the map frame and its covariance (cov_xx, cov_yy, cov_xy, cov_yaw)
- SLAM pose transformed to world coordinates, and its source (TF or topic)
- **Ground-truth pose** from the Webots supervisor
- **Odometry pose**
- LiDAR distances (front, left, right sectors)
- Distance and heading error to goal, from both the estimate and the truth
- Action taken, the policy's intended mean, sigma, and the action's log-density
- Reward, `terminated`, `truncated`, and termination reason
- Full next-state for every field above
Two logging decisions are load-bearing. The behaviour policy's action log-density must be recorded at collection time because it cannot be reconstructed afterwards, and no importance-sampling estimator exists without it. `terminated` and `truncated` are kept separate because a value estimator must treat a genuine termination as zero future value and a time-limit cutoff as non-zero.
 
## Dataset
 
A **transition** is one timestep, i.e. one row of a CSV. The three outcome rows count how each episode ended.
 
| | Behaviour policy | Target policy |
|---|---|---|
| Episodes retained | 199 | 50 |
| Transitions logged | 58,728 | used only as a reference value |
| Ended at goal | 39 (19.6%) | 9 (18.0%) |
| Ended on proximity | 88 (44.2%) | 27 (54.0%) |
| Ended at the step limit | 72 (36.2%) | 14 (28.0%) |
 
One episode was excluded after the host machine suspended mid-run, producing a 1385 s wall-clock gap inside a 50 s episode. Detected automatically by the validator.
 
The 50 target episodes are never shown to any estimator. They exist only to measure the value the estimators are trying to predict.
 
## Discount Factor
 
Gamma is the discount factor, controlling how much a reward earned later counts relative to one earned now. Selecting it turned out to be a methodological question rather than a parameter choice.
 
| Gamma | Mean return, goal | Collision | Step limit |
|---|---|---|---|
| 0.99 | +9.49 | **+1.62** | +4.64 |
| 0.995 | +16.82 | **+0.37** | +4.77 |
| **0.999** | +28.05 | -2.55 | -0.28 |
| 1.0 | +32.23 | -3.80 | -3.92 |
 
At gamma of 0.995 or below, collisions score **positive**, because the -10 penalty arrives around step 150 and is discounted away. At gamma of 1.0 the Bellman update is not a contraction and fitted Q evaluation does not converge. Gamma of 0.999 was the only value tested that satisfies both constraints.
 
## Results
 
Ground truth at gamma = 0.999, measured by running each policy:
 
~~~
V(behaviour) = +4.272 +/- 1.261   (199 episodes)
V(target)    = +6.928 +/- 2.189   ( 50 episodes)
~~~
 
Estimator accuracy against the withheld target value:
 
| Estimator | Estimate | Error |
|---|---|---|
| Importance sampling | 0.018 | -6.910 |
| Per-decision IS | 1.685 | -5.243 |
| Weighted IS | 8.423 | **+1.495** |
| Fitted Q evaluation | 5.414 | **-1.514** |
 
Weighted importance sampling and fitted Q evaluation both recover the reference to within its own standard error. Plain and per-decision importance sampling collapse toward zero at an effective sample size of 5.56 of 199 episodes.
 
Fitted Q evaluation was calibrated on the on-policy task, where the behaviour policy's true value is available from data the estimator already sees. It produced +4.659 against a true +4.272, an error of +0.387. Hyperparameters were selected using that task only; the target policy's value was never used for tuning.
 
### Does SLAM's reported covariance track its actual error?
 
Measured over all 58,728 timesteps:
 
| Quantity | Correlation with actual localisation error |
|---|---|
| cov_xx, Pearson | -0.147 |
| cov_xx, Spearman | -0.224 |
| cov_xx, controlling for elapsed timestep | **-0.024** |
| Elapsed timestep alone | **+0.779** |
 
Mean localisation error by covariance decile, lowest to highest:
 
~~~
1.620, 1.224, 1.020, 0.985, 1.010, 0.889, 0.956, 1.122, 1.151, 0.779 m
~~~
 
The raw correlation is almost entirely a shared dependence on elapsed time. Controlling for that, the published covariance carries essentially no information about accumulated localisation error, while elapsed time predicts it well.
 
slam_toolbox publishes the covariance calculated from the **scan match**, a local measure of how well the latest scan aligned with the map. The error being measured is accumulated drift, which in a one-way traverse with few loop closures is never corrected. Across 8 episodes only 3 revisit events were detected.
 
## Status
 
- [x] Behaviour policy implemented in ROS2
- [x] Data collection pipeline with independent episodes
- [x] 199 behaviour episodes and 50 target episodes collected and validated
- [x] Importance sampling, weighted IS and per-decision IS
- [x] Fitted Q evaluation
- [x] Covariance versus localisation error analysis
- [ ] Model-based OPE pipeline
- [ ] Doubly robust estimator
- [ ] Bootstrap confidence intervals
- [ ] Additional seeds for the state-representation comparison
- [ ] Second environment
## How to Run
 
1. Open Webots and load `worlds/turtlebot_slam.wbt`
2. Press play in Webots
3. In a separate terminal:
~~~bash
   cd ~/ros2_ws
   source install/setup.bash
   ./run_episodes.sh 1 200 500 behaviour_a 0.3 ~/ope_data
~~~
 
Arguments are start episode, end episode, max timesteps, policy name, sigma and output directory. One CSV per episode is written to the output directory.
 
4. Validate the collected data:
~~~bash
   python3 validate_dataset.py ~/ope_data
~~~
 
## Known Limitations
 
1. One environment and one robot, so generalisation is untested.
2. The model-based and doubly robust estimators are not yet implemented, so only model-free methods have been compared.
3. Only scan-match covariance is available. Pose-graph marginal covariance is not published by slam_toolbox and was not computed.
4. The task affords few loop closures, so localisation uncertainty accumulates monotonically and is rarely corrected.
5. Collision is defined by a LiDAR threshold, so objects below the LiDAR plane can be contacted without triggering termination.
6. Commanded velocity is tracked at a median of about 0.87 of nominal due to actuator dynamics. The logged action is the commanded value.
7. Q is evaluated at the target policy's mean action rather than averaged over its noise distribution.
8. The state-representation comparison uses three seeds per arm, which is insufficient to separate them.
9. No confidence intervals yet.
## Documentation
 
The `archive/` folder contains the earlier EKF-SLAM implementation that was superseded by this ROS2 + slam_toolbox version.
