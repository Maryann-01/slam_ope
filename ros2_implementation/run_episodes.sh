#!/usr/bin/env bash
# Usage: ./run_episodes.sh START END [MAX_TIMESTEPS] [POLICY_NAME] [SIGMA]
START=${1:-1}
END=${2:-50}
MAXT=${3:-300}
POLICY=${4:-behaviour_a}
SIGMA=${5:-0.3}
OUT=${6:-~/ope_data}
GAIN=${7:-1.5}

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
mkdir -p "$OUT"
rm -f "$OUT"/${POLICY}_ep*.csv.aborted

for ((i=START; i<=END; i++)); do
  echo "=================== EPISODE $i / $END  ($POLICY, sigma=$SIGMA) ==================="
  export OPE_EPISODE_SEED=$i
  for attempt in 1 2 3; do
    timeout 240 ros2 launch slam_webots_pkg slam_webots.launch.py \
        episode:=$i seed:=$i output_dir:=$OUT max_timesteps:=$MAXT \
        policy_name:=$POLICY sigma:=$SIGMA gain:=$GAIN
    if [ -f "$OUT/${POLICY}_ep$(printf %04d $i).csv" ]; then break; fi
    echo "--- episode $i attempt $attempt failed, retrying ---"
    pkill -9 -f webots_ros2_driver 2>/dev/null
    pkill -9 -f slam_toolbox 2>/dev/null
    sleep 8
  done
  pkill -9 -f webots_ros2_driver 2>/dev/null
  sleep 6
done

echo "Done. Files in $OUT:"
ls -1 "$OUT" | tail -n 5
