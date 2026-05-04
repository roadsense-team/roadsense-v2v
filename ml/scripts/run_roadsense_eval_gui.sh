#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $(basename "$0") <scenario_name>" >&2
  echo "Example: $(basename "$0") eval_n5_000" >&2
  exit 1
fi

SCENARIO_NAME="$1"
SCENARIO_DIR="/home/amirkhalifa/RoadSense2/roadsense-v2v/ml/scenarios/datasets/dataset_v6_formation_fix/eval/${SCENARIO_NAME}"

if [ ! -d "$SCENARIO_DIR" ]; then
  echo "Scenario directory not found: $SCENARIO_DIR" >&2
  exit 1
fi

XAUTH_FILE="${XAUTHORITY:-$(find /run/user/$(id -u) -maxdepth 1 -name '.mutter-Xwaylandauth.*' 2>/dev/null | head -n1)}"
if [ -z "$XAUTH_FILE" ]; then
  echo "No Xwayland auth file found" >&2
  exit 1
fi

[ -z "${DISPLAY:-}" ] && export DISPLAY=:0

rm -f "$SCENARIO_DIR/fcd_output.csv" "$SCENARIO_DIR/ssm_output.xml"

cleanup() {
  xhost -local: >/dev/null 2>&1 || true
}
trap cleanup EXIT

xhost +local: >/dev/null

docker run --rm -it \
  --security-opt label=disable \
  --user "$(id -u):$(id -g)" \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$XAUTH_FILE:/tmp/.Xauthority:ro" \
  -v "$SCENARIO_DIR:/data:Z" \
  -w /data \
  ghcr.io/eclipse-sumo/sumo:main \
  sumo-gui -c scenario.sumocfg
