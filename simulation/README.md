# RoadSense Simulation Environment

This directory contains the Dockerfile to build a containerized simulation environment for the RoadSense V2V project, including OMNeT++ 6.2.0, SUMO 1.22.0, and Veins 5.3.1.

## Quick Start (Recommended - Pre-Built Image)

For most users, downloading the pre-built image from Docker Hub is faster and easier than building from source.

### Prerequisites

- **Docker Desktop** installed on your system
- For GUI support:
  - **Windows 11**: WSL2 with Ubuntu distribution
  - **Linux/Fedora**: X11 server (usually pre-installed)
  - **macOS**: XQuartz

### For Windows 11 Users (RECOMMENDED!)

**✨ This is the easiest and most reliable method for Windows!**

**Step 0: Configure Docker Desktop (One-Time Setup)**

1. Open **Docker Desktop** → Settings (gear icon)
2. Go to **Resources** → **WSL Integration**
3. Enable:
   - ✅ **"Enable integration with my default WSL distro"**
   - ✅ Toggle on your **Ubuntu** distribution
4. Click **"Apply & Restart"**

**Step 1: Open WSL2 Ubuntu Terminal**

Open "Ubuntu" from Start Menu, or run in PowerShell:
```powershell
wsl
```

**Step 2: Pull and Run from WSL2**

```bash
# Pull the pre-built image (one-time download, ~10-15 min)
docker pull amirkhalifa285/roadsense-sim:v1.0.0

# Run the container (WSLg handles GUI automatically!)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  amirkhalifa285/roadsense-sim:v1.0.0 bash
```

**Why this approach?**
- ✅ **GUI works automatically** (no manual IP configuration!)
- ✅ Native Linux environment for development
- ✅ No need for VcXsrv or third-party X11 servers
- ✅ WSLg is built into Windows 11

---

### For Linux/Fedora Users

```bash
# Pull the image
docker pull amirkhalifa285/roadsense-sim:v1.0.0

# Run with X11 forwarding
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  amirkhalifa285/roadsense-sim:v1.0.0 bash
```

**Docker Hub Repository:** https://hub.docker.com/r/amirkhalifa285/roadsense-sim

**Available Tags:**
- `v1.0.0` - Initial release (OMNeT++ 6.2.0, SUMO 1.22.0, Veins 5.3.1) - **Recommended for reproducible research**
- `latest` - Always points to most recent version (use `v1.0.0` for reproducible results)

---

## Running Simulations

Once inside the container, you can run Veins simulations:

```bash
# Start the Veins launch daemon
cd /root/v2v-workspace/veins-veins-5.3.1
./bin/veins_launchd -vv -p 9999 &

# Run the example simulation (GUI will appear)
cd examples/veins
./run

# For command-line (no GUI) mode:
./run -u Cmdenv
```

### Installed Software Locations

Inside the container, the simulation stack is pre-installed:

- **OMNeT++**: `/root/omnetpp-6.2.0`
- **SUMO**: `/root/v2v-workspace/sumo-1.22.0`
- **Veins**: `/root/v2v-workspace/veins-veins-5.3.1`

Environment variables are pre-configured, so you can use `omnetpp`, `sumo`, and related commands directly.

---

## Advanced: Building from Source

If you need to modify the Dockerfile or simulation environment, you can build the image locally.

### Build the Docker Image

```bash
# From the roadsense-v2v directory:
cd /path/to/roadsense-v2v
docker build -t roadsense-sim -f simulation/Dockerfile .
```

**Note:** Building from source takes **15-20 minutes** and downloads ~2-3 GB of packages. The pre-built image is recommended unless you need customization.

### Run Your Custom Build

```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  roadsense-sim bash
```

---

## Version Management

When updating the Dockerfile, follow semantic versioning:

| Change Type | Example | New Version |
|-------------|---------|-------------|
| **Patch** (bug fixes) | Fix configuration typo | v1.0.1 |
| **Minor** (new features) | Add custom scenarios | v1.1.0 |
| **Major** (breaking changes) | Upgrade to OMNeT++ 7.0 | v2.0.0 |

**To publish a new version:**

```bash
# Build with new version tag
docker build -t amirkhalifa285/roadsense-sim:v1.1.0 -f simulation/Dockerfile .

# Also tag as latest
docker tag amirkhalifa285/roadsense-sim:v1.1.0 amirkhalifa285/roadsense-sim:latest

# Push both tags to Docker Hub
docker push amirkhalifa285/roadsense-sim:v1.1.0
docker push amirkhalifa285/roadsense-sim:latest
```

---

## Troubleshooting

### GUI Windows Don't Appear (Windows 11 WSL2)

**Solution 1: Update WSL**

Exit the container, then in PowerShell:

```powershell
wsl --update
wsl --shutdown
```

Wait 10 seconds, restart Docker Desktop, then try again from WSL2.

**Solution 2: Check Docker Desktop WSL Integration**

1. Docker Desktop → Settings → Resources → WSL Integration
2. Ensure "Enable integration with my default WSL distro" is ✅ ON
3. Toggle on your Ubuntu distribution
4. Apply & Restart

**Solution 3: Verify DISPLAY Variable**

In WSL2 Ubuntu:

```bash
echo $DISPLAY
```

Should show `:0` or `:0.0`. If empty, WSLg might not be configured properly.

---

### GUI Windows Don't Appear (Linux)

```bash
# Allow Docker to access X11 display
xhost +local:docker

# Then run the container
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  amirkhalifa285/roadsense-sim:v1.0.0 bash
```

---

### Docker Command Not Found in WSL2

**Solution:**

1. Open Docker Desktop → Settings → Resources → WSL Integration
2. Enable integration with your Ubuntu distribution
3. Click "Apply & Restart"
4. Close and reopen your WSL2 terminal

---

### Container Runs But Simulation Commands Fail

Verify the simulation tools are installed:

```bash
# Inside the container:
opp_run --version   # Should show: OMNeT++ 6.2.0
sumo --version      # Should show: SUMO 1.22.0
```

Note: `omnetpp -v` tries to launch the IDE (GUI app), use `opp_run --version` instead for command-line verification.

---

## Academic Use

For reproducible research, always specify the version tag in your papers:

```
The simulations were conducted using the RoadSense simulation environment
(Docker image: amirkhalifa285/roadsense-sim:v1.0.0) which includes OMNeT++ 6.2.0,
SUMO 1.22.0, and Veins 5.3.1.
```

This ensures other researchers can replicate your exact simulation environment.
