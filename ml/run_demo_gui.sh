#!/bin/bash
# RoadSense PoC Demo — SUMO GUI with Trained Model
# ==================================================
# Fedora/Wayland-compatible launcher for the ConvoyEnv GUI demo.
#
# Usage:
#   ./ml/run_demo_gui.sh                          # default: Run 025 model
#   ./ml/run_demo_gui.sh --scenario base --delay 0.2
#   ./ml/run_demo_gui.sh --model_path ml/path/to/other_model.zip
#   ./ml/run_demo_gui.sh --no-model               # random actions (testing)

set -e

IMAGE_NAME="roadsense-ml:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Default model for PoC demo (Run 025, 500k checkpoint)
DEFAULT_MODEL="ml/results/run_025_replay_v1/checkpoints/replay_ft_500000_steps.zip"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ---- Build image if needed ----
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    log_info "Building Docker image (first time, ~4 min)..."
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

# ---- Wayland/X11 display setup ----
GUI_ARGS=""

if [[ -n "$WAYLAND_DISPLAY" ]]; then
    log_info "Platform: Fedora Wayland (XWayland)"
    [[ -z "$DISPLAY" ]] && export DISPLAY=:0

    # Find Xwayland auth file (Mutter on Fedora)
    XAUTH="${XAUTHORITY:-}"
    if [[ -z "$XAUTH" ]]; then
        XAUTH="$(find /run/user/$(id -u) -maxdepth 1 -name '.mutter-Xwaylandauth.*' 2>/dev/null | head -n1)"
    fi

    xhost +local: >/dev/null 2>&1 || log_warn "xhost failed — GUI may not work"

    GUI_ARGS="--security-opt label=disable"
    GUI_ARGS="$GUI_ARGS --user $(id -u):$(id -g)"
    GUI_ARGS="$GUI_ARGS -e DISPLAY=$DISPLAY"
    GUI_ARGS="$GUI_ARGS -e QT_X11_NO_MITSHM=1"
    GUI_ARGS="$GUI_ARGS -v /tmp/.X11-unix:/tmp/.X11-unix:rw"

    if [[ -n "$XAUTH" && -f "$XAUTH" ]]; then
        log_info "Xwayland auth: $XAUTH"
        GUI_ARGS="$GUI_ARGS -e XAUTHORITY=/tmp/.Xauthority"
        GUI_ARGS="$GUI_ARGS -v $XAUTH:/tmp/.Xauthority:ro"
    else
        log_warn "No Xwayland auth file found — GUI may fail"
    fi
else
    log_info "Platform: X11"
    xhost +local:docker >/dev/null 2>&1 || true
    GUI_ARGS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
fi

# ---- Build demo command ----
# Check if --no-model was passed (skip default model)
DEMO_ARGS=()
USE_DEFAULT_MODEL=true

for arg in "$@"; do
    if [[ "$arg" == "--no-model" ]]; then
        USE_DEFAULT_MODEL=false
    else
        DEMO_ARGS+=("$arg")
    fi
done

# Add default model if no --model_path was explicitly passed and --no-model wasn't set
HAS_MODEL_ARG=false
for arg in "$@"; do
    if [[ "$arg" == "--model_path" || "$arg" == "-m" ]]; then
        HAS_MODEL_ARG=true
        break
    fi
done

if [[ "$USE_DEFAULT_MODEL" == true && "$HAS_MODEL_ARG" == false ]]; then
    DEMO_ARGS+=(--model_path "$DEFAULT_MODEL")
fi

log_info "Starting ConvoyEnv GUI Demo..."
log_info "  Model: ${HAS_MODEL_ARG:+custom}${USE_DEFAULT_MODEL:+$DEFAULT_MODEL}${USE_DEFAULT_MODEL:-RANDOM}"
log_info "  Extra args: ${DEMO_ARGS[*]:-none}"

# ---- Run ----
# shellcheck disable=SC2086
docker run --rm -it \
    $GUI_ARGS \
    -e SUMO_AVAILABLE=true \
    -v "$REPO_ROOT:/work:Z" \
    -w /work \
    "$IMAGE_NAME" \
    python3 ml/demo_convoy_gui.py "${DEMO_ARGS[@]}"

# Clean up xhost
if [[ -n "$WAYLAND_DISPLAY" ]]; then
    xhost -local: >/dev/null 2>&1 || true
else
    xhost -local:docker >/dev/null 2>&1 || true
fi
