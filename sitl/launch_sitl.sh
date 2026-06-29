#!/usr/bin/env bash
# sitl/launch_sitl.sh
#
# Launches the full simulation stack used for this project:
#   1. Gazebo with an Iris quadcopter model + a disaster-scenario world
#      (rubble, debris piles, scattered "victim" actors/props, parked vehicles)
#   2. ArduPilot SITL (ArduCopter) connected to that Gazebo model via the
#      ardupilot_gazebo plugin
#   3. The perception pipeline (pipeline/realtime_pipeline.py), which
#      subscribes to the Gazebo camera feed and sends MAVLink commands back
#      to SITL based on detections.
#
# One-time setup (not run by this script — see README for full steps):
#   git clone https://github.com/ArduPilot/ardupilot.git
#   git clone https://github.com/ArduPilot/ardupilot_gazebo.git
#   (build ardupilot_gazebo plugin per its README, source ROS + Gazebo env)
#
# Usage:
#   chmod +x sitl/launch_sitl.sh
#   ./sitl/launch_sitl.sh disaster_world.world

set -e

WORLD_FILE="${1:-worlds/disaster_response.world}"
CONNECTION_STRING="udp:127.0.0.1:14550"

echo "[1/3] Starting Gazebo with world: ${WORLD_FILE}"
gzserver --verbose "${WORLD_FILE}" &
GAZEBO_PID=$!
sleep 5

echo "[2/3] Starting ArduPilot SITL (ArduCopter, gazebo frame)"
( cd "${ARDUPILOT_HOME:-../ardupilot}/ArduCopter" && \
  ../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map ) &
SITL_PID=$!
sleep 10

echo "[3/3] Starting perception pipeline"
python3 ../pipeline/realtime_pipeline.py \
  --weights ../runs/train/disaster_uav_full/weights/best.pt \
  --mavlink "${CONNECTION_STRING}" \
  --camera-topic /iris/camera/image_raw

trap "kill ${GAZEBO_PID} ${SITL_PID} 2>/dev/null" EXIT
