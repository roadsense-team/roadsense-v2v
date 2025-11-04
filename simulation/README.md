# RoadSense Simulation Environment

This directory contains the Dockerfile to build a containerized simulation environment for the RoadSense project.

## Prerequisites

- Docker installed on your system.

## Build the Docker Image

To build the Docker image, run the following command from the `roadsense-v2v` directory:

```bash
docker build -t roadsense-sim -f simulation/Dockerfile .
```

## Run the Docker Container

To run the Docker container, use the following command:

```bash
docker run -it --rm roadsense-sim
```

This will start the container and drop you into a bash shell.

## Using the Simulation Environment

Inside the container, the simulation environment is pre-installed and configured.

- **OMNeT++** is installed in `/root/omnetpp-6.2.0`.
- **SUMO** is installed in `/root/v2v-workspace/sumo-1.22.0`.
- **Veins** is installed in `/root/v2v-workspace/veins-veins-5.3.1`.

The environment variables are set up, so you can directly use the `omnetpp`, `sumo`, and other commands.
