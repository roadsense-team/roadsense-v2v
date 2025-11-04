# Dockerfile for RoadSense V2V Simulation Environment

# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Install core development tools for OMNeT++
RUN apt-get install -y make diffutils pkg-config ccache clang lld gdb lldb \
    bison flex perl sed gawk python3 python3-pip python3-venv python3-dev \
    libxml2-dev zlib1g-dev doxygen graphviz xdg-utils libdw-dev wget

# Install GUI support for OMNeT++ (Qtenv and IDE)
RUN apt-get install -y qt6-base-dev qt6-base-dev-tools qmake6 libqt6svg6 \
    qt6-wayland libwebkit2gtk-4.1-0

# Install 3D visualization support (optional but recommended)
RUN apt-get install -y libopenscenegraph-dev

# Clean package cache
RUN apt-get clean

# Set up working directory
WORKDIR /root

# Download and extract OMNeT++
RUN wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-6.2.0/omnetpp-6.2.0-linux-x86_64.tgz
RUN tar xvfz omnetpp-6.2.0-linux-x86_64.tgz

# Set up environment variables for OMNeT++
WORKDIR /root/omnetpp-6.2.0
ENV PATH="/root/omnetpp-6.2.0/bin:$PATH"

# Configure and build OMNeT++
RUN bash -c "chmod +x setenv && . ./setenv && python3 -m venv .venv && . .venv/bin/activate && python3 -m pip install -r python/requirements.txt && ./configure"
RUN bash -c ". ./setenv && make"

# Verify OMNeT++ installation
RUN omnetpp -v

# Create workspace for SUMO and Veins
WORKDIR /root
RUN mkdir v2v-workspace
WORKDIR /root/v2v-workspace

# Download and extract SUMO
RUN wget https://sumo.dlr.de/releases/1.22.0/sumo-src-1.22.0.tar.gz
RUN tar xzf sumo-src-1.22.0.tar.gz

# Install SUMO dependencies
RUN apt-get install -y cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev python3-dev swig default-jdk maven libeigen3-dev

# Build and install SUMO
WORKDIR /root/v2v-workspace/sumo-1.22.0
RUN mkdir build
WORKDIR /root/v2v-workspace/sumo-1.22.0/build
RUN cmake ..
RUN make -j$(nproc)
RUN make install

# Verify SUMO installation
RUN sumo --version

# Download and extract Veins
WORKDIR /root/v2v-workspace
RUN wget https://github.com/sommer/veins/archive/refs/tags/veins-5.3.1.tar.gz
RUN tar xzf veins-5.3.1.tar.gz
WORKDIR /root/v2v-workspace/veins-veins-5.3.1

# Configure and build Veins
RUN ./configure
RUN make

# Verify Veins build
RUN find / -name "libveins_dbg.so" | grep .

# Final CMD
CMD ["/bin/bash"]
