# RoadSense V2V System

Vehicle-to-Vehicle hazard detection using ESP32, machine learning, and mesh networking.

## Quick Start

### Simulation Environment (Recommended)

Get started with the pre-built simulation environment in minutes:

#### For Windows 11 Users (RECOMMENDED - Easiest Setup!)

**Step 1: Open WSL2 Ubuntu Terminal**

Open "Ubuntu" from Start Menu or run `wsl` in PowerShell.

**Step 2: Pull and Run from WSL2**

```bash
# Pull the pre-built Docker image (one-time download, ~10-15 min)
docker pull amirkhalifa285/roadsense-sim:v1.0.0

# Run the simulation environment with GUI support (WSLg handles display automatically!)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  amirkhalifa285/roadsense-sim:v1.0.0 bash

# Inside the container, run a Veins simulation:
cd /root/v2v-workspace/veins-veins-5.3.1
./bin/veins_launchd -vv -p 9999 &
cd examples/veins
./run
```

**The OMNeT++ GUI will appear on your Windows desktop automatically!** ✅

---

#### For Linux/Fedora Users

```bash
# Pull the image
docker pull amirkhalifa285/roadsense-sim:v1.0.0

# Run with X11 forwarding
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  amirkhalifa285/roadsense-sim:v1.0.0 bash

# Inside the container, run simulation:
cd /root/v2v-workspace/veins-veins-5.3.1
./bin/veins_launchd -vv -p 9999 &
cd examples/veins
./run
```

**Docker Hub:** https://hub.docker.com/r/amirkhalifa285/roadsense-sim

**Available Tags:**
- `v1.0.0` - Initial release (OMNeT++ 6.2.0, SUMO 1.22.0, Veins 5.3.1) - **Recommended for reproducible research**
- `latest` - Always points to most recent version

See [`simulation/README.md`](simulation/README.md) for detailed simulation setup and building from source.

### Hardware Setup

For ESP32 firmware development, see [`hardware/README.md`](hardware/README.md).

## Project Structure

This repository contains the source code for the RoadSense project. The project is divided into the following subdirectories:

- `hardware/`: ESP32 firmware
- `simulation/`: Veins/SUMO scenarios (Dockerfile available)
- `bridge/`: Hardware-in-the-loop bridge
- `ml/`: Machine learning pipeline
- `mcp/`: MCP server configuration
- `docs/`: Project documentation
- `scripts/`: Utility scripts
- `tests/`: Integration tests
- `ui/`: Web dashboard (future)
- `.github/`: GitHub specific files
- `config/`: Global configurations
- `.vscode/`: VSCode workspace settings
