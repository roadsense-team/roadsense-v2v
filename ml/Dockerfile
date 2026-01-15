# RoadSense V2V ML Training Environment
# =====================================
#
# Single container: SUMO + Python ML stack (zero-latency TraCI).
# Supports headless training (all platforms) and GUI visualization.
#
# QUICK START
# -----------
# Build:
#   docker build -t roadsense-ml:latest .
#
# Run tests (headless):
#   docker run --rm -v $(pwd):/work:Z roadsense-ml:latest
#
# See run_docker.sh for GUI and platform-specific commands.

FROM ghcr.io/eclipse-sumo/sumo:main

LABEL maintainer="Amir Khalifa"
LABEL description="RoadSense V2V ML Training Environment"
LABEL version="1.0.0"

# The SUMO image is Debian-based and has Python 3 pre-installed
# Install venv for isolated Python environment
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to avoid conflicts with system packages
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --no-cache-dir --upgrade pip

# Install ML dependencies in the isolated venv
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set Python path for imports (project code mounted at /work)
ENV PYTHONPATH="/work"

# Working directory (overridden by volume mount)
WORKDIR /work

# Verify installation
RUN sumo --version && python -c "import gymnasium; import stable_baselines3; print('ML stack: OK')"

# Default: run unit tests (headless, works everywhere)
CMD ["pytest", "tests/unit/", "-v", "--tb=short"]
