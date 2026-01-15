#!/bin/bash
# RoadSense ML Docker Runner
# ==========================
# Cross-platform script for running the ML training container.
# Handles GUI setup for Linux (X11/Wayland), Windows (WSL2), and macOS.
#
# Usage:
#   ./run_docker.sh              # Run tests (headless)
#   ./run_docker.sh gui          # Interactive shell with GUI support
#   ./run_docker.sh demo         # Run visualization demo
#   ./run_docker.sh train        # Run RL training (headless)
#   ./run_docker.sh bash         # Interactive shell (headless)

set -e

IMAGE_NAME="roadsense-ml:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Detect platform
detect_platform() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if grep -q Microsoft /proc/version 2>/dev/null; then
            echo "wsl"
        elif [[ -n "$WAYLAND_DISPLAY" ]]; then
            echo "linux-wayland"
        else
            echo "linux-x11"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows-native"
    else
        echo "unknown"
    fi
}

# Build the image if needed
build_if_needed() {
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        log_info "Building Docker image..."
        docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
    fi
}

# Setup GUI environment variables and mounts
setup_gui() {
    local platform=$(detect_platform)
    GUI_ARGS=""

    case "$platform" in
        linux-wayland)
            log_info "Platform: Linux (Wayland via XWayland)"
            # XWayland provides X11 socket on Wayland
            if [[ -z "$DISPLAY" ]]; then
                export DISPLAY=:0
            fi
            # Allow local Docker connections (redirect all output)
            xhost +local:docker >/dev/null 2>&1 || log_warn "xhost not available, GUI may not work"
            GUI_ARGS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
            ;;
        linux-x11)
            log_info "Platform: Linux (X11)"
            xhost +local:docker >/dev/null 2>&1 || log_warn "xhost not available"
            GUI_ARGS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
            ;;
        wsl)
            log_info "Platform: Windows WSL2"
            # WSLg on Windows 11 sets DISPLAY automatically
            if [[ -z "$DISPLAY" ]]; then
                # Try WSLg default
                export DISPLAY=:0
            fi
            if [[ -S "/tmp/.X11-unix/X0" ]]; then
                log_info "WSLg detected (Windows 11)"
                GUI_ARGS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
            else
                log_warn "WSLg not detected. For Windows 10, install VcXsrv:"
                log_warn "  1. Install VcXsrv from https://sourceforge.net/projects/vcxsrv/"
                log_warn "  2. Run XLaunch with 'Disable access control' checked"
                log_warn "  3. Set DISPLAY: export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0"
                # Try to get Windows host IP for VcXsrv
                WIN_HOST=$(cat /etc/resolv.conf 2>/dev/null | grep nameserver | awk '{print $2}')
                if [[ -n "$WIN_HOST" ]]; then
                    export DISPLAY="${WIN_HOST}:0"
                    GUI_ARGS="-e DISPLAY=$DISPLAY"
                fi
            fi
            ;;
        macos)
            log_info "Platform: macOS"
            log_warn "macOS requires XQuartz for GUI:"
            log_warn "  1. Install: brew install --cask xquartz"
            log_warn "  2. Open XQuartz, go to Preferences > Security > Allow network clients"
            log_warn "  3. Restart and run: xhost +localhost"
            # macOS uses IP instead of socket
            export DISPLAY="host.docker.internal:0"
            GUI_ARGS="-e DISPLAY=$DISPLAY"
            ;;
        *)
            log_error "Unknown platform. GUI may not work."
            log_info "Running in headless mode..."
            GUI_ARGS=""
            ;;
    esac

    echo "$GUI_ARGS"
}

# Main run function
run_container() {
    local mode="${1:-test}"
    local gui_args=""
    local cmd=""
    local interactive=""

    case "$mode" in
        test|tests)
            log_info "Running unit tests (headless)..."
            cmd="pytest ml/tests/unit/ -v --tb=short"
            ;;
        integration)
            log_info "Running integration tests (requires SUMO)..."
            cmd="pytest ml/tests/integration/ -v --tb=short"
            ;;
        gui|shell|bash)
            log_info "Starting interactive shell with GUI support..."
            gui_args=$(setup_gui)
            cmd="bash"
            interactive="-it"
            ;;
        demo)
            log_info "Running GUI demo..."
            gui_args=$(setup_gui)
            cmd="python3 ml/demo_convoy_gui.py"
            interactive="-it"
            ;;
        train)
            log_info "Running RL training (headless)..."
            cmd="python3 -m ml.training.train_convoy ${*:2}"
            ;;
        generate)
            log_info "Generating scenarios..."
            cmd="python -m ml.scripts.gen_scenarios ${*:2}"
            ;;
        *)
            # Pass through custom command
            cmd="$*"
            ;;
    esac

    build_if_needed

    # SELinux-aware volume mount (:Z relabels for container access)
    # shellcheck disable=SC2086
    docker run --rm $interactive \
        $gui_args \
        -e SUMO_AVAILABLE=true \
        -e GIT_COMMIT \
        -e CONTAINER_IMAGE="$IMAGE_NAME" \
        -v "$REPO_ROOT:/work:Z" \
        -w /work \
        "$IMAGE_NAME" \
        $cmd
}

# Show help
show_help() {
    echo "RoadSense ML Docker Runner"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  test, tests     Run unit tests (default, headless)"
    echo "  integration     Run integration tests with SUMO"
    echo "  gui, shell      Interactive shell with GUI support"
    echo "  demo            Run ConvoyEnv visualization demo"
    echo "  train           Run RL training (headless)"
    echo "  generate        Generate augmented scenarios"
    echo "  bash            Interactive shell (headless)"
    echo "  help            Show this help"
    echo ""
    echo "Platform support:"
    echo "  Linux (X11):      Full GUI support"
    echo "  Linux (Wayland):  GUI via XWayland"
    echo "  Windows 11/WSL2:  GUI via WSLg"
    echo "  Windows 10/WSL2:  GUI via VcXsrv (manual setup)"
    echo "  macOS:            GUI via XQuartz (manual setup)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run unit tests"
    echo "  $0 demo               # Watch simulation in SUMO GUI"
    echo "  $0 gui                # Interactive shell, then run commands"
    echo ""
}

# Entry point
case "${1:-test}" in
    help|-h|--help)
        show_help
        ;;
    *)
        run_container "$@"
        ;;
esac
