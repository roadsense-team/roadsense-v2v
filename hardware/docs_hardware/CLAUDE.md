# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

-- always update the docs/hw_firmware_migration_plan.md docs/HARDWARE_PROGRESS.txt with the most up to date information that is inside the repo CLAUDE.md file"
## Project Overview

**RoadSense V2V System** is a Vehicle-to-Vehicle (V2V) hazard detection and communication system using 3 ESP32 hardware units representing vehicles. The system combines real-time sensor data from MPU9250 9-axis IMUs, NEO-6M GPS modules, and ESP-NOW mesh networking to enable infrastructure-free collision avoidance and traffic optimization.

This is a continuation project that enhances an existing RoadSense system by:
- Adding advanced sensors (gyroscope, magnetometer) to existing accelerometer-based detection
- Implementing quantized machine learning models suitable for embedded systems
- Setting up comprehensive Veins/SUMO/OMNeT++ simulation environment
- Improving mesh network communication protocols

---

## 📁 IMPORTANT: Directory Structure (For Amir)

**⚠️ NOTE:** This CLAUDE.md file and associated planning documents are **NOT in the Git repository**. They are personal workspace files.

### Amir's Personal Workspace Structure:

**Fedora Laptop:**
```
/home/amirkhalifa/RoadSense2/              ← PERSONAL WORKSPACE (NOT in Git)
├── CLAUDE.md                               ← Personal AI context (NOT pushed)
├── docs/                                   ← Personal planning docs (NOT pushed)
│   ├── HARDWARE_PROGRESS.md
│   ├── hw_firmware_migration_plan.md
│   └── legacy/
└── roadsense-v2v/                         ← GIT REPOSITORY (pushed to GitHub)
    ├── .git/                               ← Git tracks THIS folder only
    ├── hardware/                           ← ESP32 firmware
    ├── simulation/                         ← Veins/SUMO scenarios
    ├── bridge/                             ← HIL bridge
    ├── ml/                                 ← ML training
    └── README.md                           ← Project README (in Git)
```

**Windows Laptop:**
```
C:\Projects\RoadSense2\                    ← PERSONAL WORKSPACE (NOT in Git)
├── CLAUDE.md                               ← Personal AI context (NOT pushed)
├── docs\                                   ← Personal planning docs (NOT pushed)
│   ├── HARDWARE_PROGRESS.md
│   ├── hw_firmware_migration_plan.md
│   └── legacy\
└── roadsense-v2v\                         ← GIT REPOSITORY (pushed to GitHub)
    ├── .git\                               ← Git tracks THIS folder only
    ├── hardware\                           ← ESP32 firmware
    ├── simulation\                         ← Veins/SUMO scenarios
    ├── bridge\                             ← HIL bridge
    ├── ml\                                 ← ML training
    └── README.md                           ← Project README (in Git)
```

### Key Points for AI Agents:

1. **Working Directory for Git Operations:**
   - Fedora: `cd /home/amirkhalifa/RoadSense2/roadsense-v2v`
   - Windows: `cd C:\Projects\RoadSense2\roadsense-v2v`

2. **When Executing Git Commands:**
   - Always `cd` into `roadsense-v2v/` first
   - Example: `cd /home/amirkhalifa/RoadSense2/roadsense-v2v && git status`

3. **Files NOT in Git (Personal Only):**
   - `/home/amirkhalifa/RoadSense2/CLAUDE.md` ← This file (Amir's notes)
   - `/home/amirkhalifa/RoadSense2/docs/` ← Planning documents
   - Any files in parent `RoadSense2/` folder

4. **Files IN Git (Team Shared):**
   - Everything inside `roadsense-v2v/` folder
   - Teammates clone: `git clone git@github.com:roadsense-team/roadsense-v2v.git`
   - Teammates don't have Amir's personal CLAUDE.md or planning docs

5. **Path Conventions in This File:**
   - When we say "project root", we mean `roadsense-v2v/` (the Git repo)
   - When we say "workspace root", we mean `RoadSense2/` (Amir's personal folder)

### For Teammates (Not Amir):

Your directory structure will be different. You'll only have:
```
~/roadsense-v2v/          ← Git repository (Linux/Mac)
C:\roadsense-v2v\         ← Git repository (Windows)
```

You won't have Amir's personal CLAUDE.md or planning documents. This is normal!

---

## Key Architecture Decisions

### Hardware Configuration
- **3 ESP32 DevKit units** representing 3 vehicles with unique IDs: "V001", "V002", "V003"
- **MPU9250**: 9-axis IMU (integrates MPU6500 accelerometer+gyroscope + AK8963 magnetometer)
- **NEO-6M GPS**: UART-based position tracking at 1Hz
- **ESP-NOW mesh**: Primary communication protocol (2.4GHz, <250ms latency target)
- **I²C address**: 0x68/0x69 for MPU9250
- **Pin assignments per unit**:
  - I2C_SDA: 21, I2C_SCL: 22 (MPU9250)
  - GPS_TX: 16, GPS_RX: 17
  - LED_STATUS: 2
  - BUTTON_CALIB: 0

---

## 🚨 CRITICAL: Protocol Abstraction Strategy

**⚠️ ATTENTION ALL AI AGENTS: Read this section FIRST before making any architectural decisions!**

### The Problem (Identified November 6, 2025)
The previous RoadSense system used a **custom V2V protocol** that only worked between RoadSense devices - not interoperable with real-world V2V systems. This is academically unacceptable and defeats the purpose of V2V safety research.

### Our Solution: Protocol Abstraction Layer
We implement **standard V2V messages at the application layer** while using **ESP-NOW as a transport prototype**:

```
┌──────────────────────────────────────────────────────────┐
│  APPLICATION LAYER (Standards-Compliant)                 │
│  SAE J2735 BSM / ETSI CAM Message Format                │ ← INTEROPERABLE
│  - Same structure in hardware AND simulation             │
│  - Can be decoded by any V2V system                      │
└────────────────┬─────────────────────────────────────────┘
                 │
     ┌───────────┴────────────┐
     │                        │
┌────▼──────────┐   ┌────────▼─────────┐
│  HARDWARE     │   │  SIMULATION      │
│  (ESP32)      │   │  (Veins)         │
│  Transport:   │   │  Transport:      │
│  ESP-NOW      │◄──┤  Generic         │
│  - 250B max   │Bridge│  Wireless        │
│  - 10-50ms    │   │  - Emulates      │
│  - 2% loss    │   │    ESP-NOW       │
│  - Broadcast  │   │    constraints   │
└───────────────┘   └──────────────────┘
         │                   │
         └─────────┬─────────┘
                   │
          ┌────────▼────────┐
          │  HIL BRIDGE     │
          │  (Python)       │
          │  Domain         │
          │  Randomization: │
          │  - Latency      │
          │  - Packet loss  │
          │  - Size limits  │
          │  - Jitter       │
          └─────────────────┘
```

### Why This Works for ML Research

**Key Insight**: Machine learning models train on **message content and timing**, NOT on RF modulation details.

**What ML sees:**
- ✅ Vehicle position (lat/lon from GPS)
- ✅ Vehicle dynamics (speed, acceleration from IMU)
- ✅ Message arrival time (timestamp)
- ✅ Message freshness (age, staleness)
- ✅ Message reliability (did it arrive? how late?)

**What ML does NOT see:**
- ❌ RF signal strength (RSSI)
- ❌ Channel fading characteristics
- ❌ MAC layer contention
- ❌ Carrier frequency (2.4 GHz vs 5.9 GHz)

### Academic Justification

1. **Application layer is standards-compliant** - Messages follow SAE J2735 BSM or ETSI CAM format
2. **Transport layer is prototype-only** - ESP-NOW used due to budget constraints (~$2000 for real 802.11p radios)
3. **Simulation validates standard behavior** - Veins/OMNeT++ can model IEEE 802.11p or generic wireless
4. **HIL bridge emulates realistic constraints** - Latency, loss, and range match real V2V networks
5. **Architecture is modular** - Swapping ESP-NOW for 802.11p requires NO application code changes

### Implementation Requirements

**ALL CODE MUST:**
- ✅ Use shared message schema (C struct) between hardware and simulation
- ✅ Enforce 250-byte payload limit
- ✅ Include timestamps in all messages
- ✅ Handle stale messages (drop if >500ms old)
- ✅ Apply domain randomization in HIL bridge
- ✅ Support both ESP-NOW (hardware) and UDP (simulation) transports

**NEVER:**
- ❌ Assume IEEE 802.11p-specific features (ACKs, CSMA/CA details)
- ❌ Hard-code transport layer in application logic
- ❌ Use custom message formats that deviate from BSM/CAM standards
- ❌ Ignore network constraints (latency, loss, size) in simulation

---

## 🔧 IMPLEMENTATION REALITY CHECK

**⚠️ READ THIS BEFORE MAKING ANY TECHNICAL DECISIONS OR TIMELINE ESTIMATES**

### Critical Professor Requirements (NON-NEGOTIABLE)

1. **✅ On-device ML inference** - Model MUST run on ESP32, not laptop (professor explicitly requested this)
   - Primary: TensorFlow Lite quantization (<100KB model)
   - Fallback: Dual-ESP32 configuration (one for sensors, one for ML)
   - **This is core to the project value proposition**

2. **✅ Real-world data collection** - Must physically deploy in vehicles
   - Minimum 3 scenarios: convoy, intersection, lane-change
   - Record GPS + IMU + V2V messages with proper timestamps

3. **✅ Standards-compliant V2V messages** - SAE J2735 BSM format
   - Not custom protocol like previous RoadSense

4. **✅ Demonstrable improvement** - Show metrics better than previous system
   - More scenarios, better accuracy, lower false positives

### HIL Status: OPTIONAL (Can Be Stretch Goal)

**Question**: Is Hardware-in-the-Loop (HIL) bridge critical?

**Answer**: **NO** - Here's why:

**What HIL provides**:
- Test 3 physical + N virtual vehicles (scalability testing)
- Validate cross-platform message compatibility
- Test dangerous scenarios virtually

**What you actually need to graduate**:
- ✅ Good ML collision detection model
- ✅ 3 ESP32s communicating successfully
- ✅ On-device ML inference working
- ✅ Real-world validation in 3-4 scenarios

**Recommendation**: Make HIL a **stretch goal** for Month 8-9, not a core dependency.

### Complexity Assessment by Component

#### 🟢 **LOW COMPLEXITY** (1-2 weeks each)
- ESP-NOW mesh: ~5 function calls, Arduino examples work
- GPS parsing: TinyGPS++ library, basic parsing 30 minutes
- Python UDP: Simple send/receive 20 lines

#### 🟡 **MEDIUM COMPLEXITY** (3-4 weeks each)
- Message format: Use Protocol Buffers or MessagePack (saves 2 weeks vs raw C structs)
- SUMO GPS import: 70% of time is data cleaning (dropouts, multipath, jumps)
- Basic simulation scenarios: Creating realistic, repeatable scenarios

#### 🔴 **HIGH COMPLEXITY** (4-8 weeks - BE CAREFUL)

**TFLite Quantization & ESP32 Deployment** ⚠️ **PROFESSOR-REQUIRED**:
- Reality: TFLite Micro ≠ TensorFlow Lite (limited subset)
- Quantization often drops accuracy 15-20%
- May need quantization-aware training
- ESP32 has only 520KB SRAM total
- **Estimated time**: 2-3 weeks if lucky, **6-8 weeks if accuracy loss is bad**

**CRITICAL FALLBACK - Dual ESP32 Setup**:
```
ESP32 #1: Sensor Hub → SPI/UART → ESP32 #2: ML Processor
```
- ESP32 #2 has ALL memory for ML (no sensor overhead)
- Can use 400KB+ for tensor arena
- Still "on-device" inference (just distributed)
- Cost: ~$15 extra per vehicle

**Recommendation**: **Plan for dual-ESP32 from the start** as fallback. Don't discover you need it in Month 9.

**OMNeT++/Veins Deep Customization** ⚠️ **TIME TRAP**:
- Learning curve: 2-3 weeks minimum
- Making Veins emulate ESP-NOW perfectly: 4-6 weeks if you know OMNeT++, **6-10 weeks from scratch**
- **BETTER APPROACH**: Use Veins 802.11p as-is, document differences, train on 802.11p, fine-tune on real ESP-NOW data
- This is **academically valid** and saves 2 months

**HIL Bridge** (if you decide to do it):
- Timing synchronization: 2 weeks alone
- Message translation: ESP32 binary ↔ Veins objects
- State conflicts, race conditions
- **Estimated time**: 4-6 weeks basic, **8-12 weeks production quality**
- **Recommendation**: **Skip HIL unless you have 2+ months for it**

### Realistic 10-Month Timeline

**Months 1-2: Hardware Setup**
- ESP32 + GPS: 2 weeks (EASY)
- MPU9250 + calibration: 2-3 weeks (MEDIUM)
- ESP-NOW mesh: 1 week code + 1-2 weeks debugging
- Initial data collection: 1 week
- **Risk**: LOW

**Months 3-4: Core Software & Data**
- Message format (use ProtoBuf): 1 week
- Data cleaning pipeline: 3-4 weeks (MEDIUM but critical)
- SUMO augmentation setup: 2-3 weeks
- **Risk**: MEDIUM-HIGH

**Months 4-5: Simulation & Dataset**
- SUMO scenarios (3 core types): 2-3 weeks
- Run augmentation (1 real → 20+ synthetic): 2 weeks
- Veins simulation runs (use 802.11p as-is): 2 weeks
- Generate training dataset: 1 week
- **Risk**: MEDIUM

**Months 6-7: ML Development** ⚠️ **CRITICAL PHASE**
- Model architecture: 1 week
- Training on mixed dataset: 2-3 weeks
- **TFLite quantization**: **4-6 weeks** (HIGH RISK)
- **Risk**: HIGH - **This is your biggest risk**

**Months 8-9: Deployment & Validation**
- ESP32 deployment: 2-3 weeks
- Real-world testing (3 scenarios): 3-4 weeks
- Model tuning: 2-4 weeks
- Optional: HIL bridge: 4-6 weeks
- **Risk**: VERY HIGH

**Month 10: Finalization**
- Demo preparation: 2 weeks
- Documentation: 2 weeks
- **Risk**: Depends on earlier phases

### Recommended Scope (Realistic for 10 Months)

**✅ CORE DELIVERABLES (Must Have)**:

1. **Hardware**: 3 ESP32 units (potentially dual-ESP32 per vehicle), MPU9250 + GPS working, ESP-NOW mesh with logging

2. **Data Collection**: Minimum 3 real-world scenarios (convoy, intersection, lane-change), properly timestamped GPS + IMU + V2V logs

3. **Simulation & Dataset**: SUMO augmentation pipeline (1 real → 10-20 synthetic), **Use Veins 802.11p as-is**, training dataset: real + augmented (100+ scenarios)

4. **Machine Learning** ⚠️ **PROFESSOR-REQUIRED**: Collision detection model, **TFLite model deployed ON ESP32** (<100KB quantized), **Fallback: Dual-ESP32 if quantization fails**, inference <100ms

5. **Validation**: Test in 3-4 real scenarios, measure accuracy/latency/false positives, **show improvement over previous RoadSense**

**⚠️ STRETCH GOALS (If Time Permits)**:
- HIL bridge (3 physical + N virtual vehicles)
- 5+ different scenario types
- Deep OMNeT++ customization

**❌ LIKELY SACRIFICES (If Schedule Slips)**:
- HIL remains "future work"
- Use Veins 802.11p without deep customization
- Dual-ESP32 setup instead of single-ESP32 ML
- 3 scenarios instead of 5+

### Critical Engineering Decisions

**Decision 1: On-Device ML** ⚠️ **NON-NEGOTIABLE**
- ✅ **MUST**: Get model running on ESP32 (professor requirement)
- ✅ **PLAN**: Dual-ESP32 as fallback from Month 1
- ❌ **DON'T**: Assume single ESP32 will work - plan for dual

**Decision 2: Veins Customization**
- ❌ **DON'T**: Spend 6-10 weeks making Veins perfectly match ESP-NOW
- ✅ **DO**: Use 802.11p as-is, document differences, fine-tune on real data

**Decision 3: HIL Priority**
- ❌ **DON'T**: Make HIL a core requirement (saves 2 months)
- ✅ **DO**: Make it a Month 8-9 stretch goal

**Decision 4: Scenario Count**
- ✅ **DO**: Commit to 3 scenarios (convoy, intersection, lane-change)
- ⚠️ **STRETCH**: Aim for 4-5 if data collection goes smoothly

**Decision 5: Message Format**
- ✅ **DO**: Use Protocol Buffers or MessagePack
- ❌ **DON'T**: Use raw C structs (2 weeks debugging padding/endianness)

**Decision 6: Dual-ESP32 Planning**
- ✅ **DO**: Design system architecture for dual-ESP32 from Day 1
- ✅ **DO**: Order 6 ESP32s (2 per vehicle) immediately

### Success Criteria (Realistic)

System demonstrates:
1. ✅ ESP-NOW V2V communication (3 vehicles)
2. ✅ **ML model running on ESP32** (single or dual-ESP32)
3. ✅ Collision detection in 3 scenarios with >85% accuracy
4. ✅ <100ms end-to-end latency (sensor → inference → warning)
5. ✅ Measurable improvement over previous RoadSense
6. ✅ Standards-compliant BSM message format
7. ✅ Real-world validation with data

This is **excellent** for B.Sc final project and publishable.

---

## Communication Architecture

**Transport Layer (Hardware):**
1. **ESP-NOW mesh** (primary): 250-byte payload, 10-50ms latency, ~2% packet loss, 100m range
2. **WiFi UDP** (secondary): Bridge to Veins/SUMO (Port 4211: ESP32, Port 4210: Veins)

**Application Layer:**
- SAE J2735 Basic Safety Message (BSM) format
- Includes: vehicle ID, position, speed, heading, timestamp
- ≤250 bytes (ESP-NOW limit)

---

## Machine Learning Components

### Model Architecture
- **Input**: 30 features (3 vehicles × 10 features each)
- **Architecture**: Dense layers (64→32→16→4 neurons) with dropout
- **Output**: 4-class softmax (None/Low/Medium/High collision risk)
- **Framework**: TensorFlow → TensorFlow Lite for ESP32

### Model Optimization
- **Target size**: <100KB quantized model
- **Inference time**: <50ms on ESP32
- **Quantization**: INT8 for input/output tensors
- **Memory**: 10KB tensor arena on ESP32
- **Fallback**: Dual-ESP32 if model too large

---

## Development Environment

### PlatformIO Configuration
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps =
  bolderflight/Bolder Flight Systems MPU9250@^1.0.2
  mikalhart/TinyGPSPlus@^1.0.3
  bblanchon/ArduinoJson@^6.21.0
monitor_speed = 115200
```

### Simulation (Docker)
```bash
cd simulation/
docker build -t roadsense-sim .
./run-veins-simulation.sh  # Linux/Fedora
# Windows 11: WSLg provides native GUI support
```

---

## Simulation Environment

### Veins/SUMO/OMNeT++ Setup
This project uses a **fresh installation** (no previous SMARTS framework). Key configuration:

**Software Stack:**
- **Veins 5.3.1**: Vehicular network simulation framework
- **SUMO 1.22.0**: Traffic simulation (vehicles, routes, traffic lights)
- **OMNeT++ 6.2.0**: Network simulation engine
- **Python bridge**: Connects ESP32 hardware to simulation via UDP

**Installation Method:**
- Dockerized complete stack (see `simulation/Dockerfile`)
- Base image: Ubuntu 24.04
- Automated build following official installation guides
- Tagged as `roadsense-sim` Docker image

**Docker Environment Setup (All Teammates):**

**Step 1: Build the Docker Image (ONE TIME per machine)**
```bash
cd simulation/
docker build -t roadsense-sim .
# This takes ~15-20 minutes. Only needs to be done once!
```

**Step 2: Run the Simulation (Platform-Specific)**

**On Fedora/Linux (Amir's setup):**
```bash
# Use the provided script for automatic X11 setup
./run-veins-simulation.sh

# OR for interactive development:
./run-docker-with-x11.sh
```

**On Windows 11 (with WSLg - recommended for team):**
```bash
# WSLg provides automatic GUI support, no extra setup needed
docker run -it --rm \
  --name roadsense-container \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  roadsense-sim bash

# Inside container:
cd /root/v2v-workspace/veins-veins-5.3.1
./bin/veins_launchd -vv -p 9999 &
cd examples/veins
./run
```

**IMPORTANT NOTE FOR WINDOWS TEAMMATES:**
These X11 scripts are **Linux-specific** and will NOT work on Windows. Windows users should:
- **Windows 11**: WSLg provides native GUI support (no extra config needed)
- **Windows 10**: Install VcXsrv or Xming for X11 forwarding
- Simply run: `docker run -it --rm -e DISPLAY=host.docker.internal:0 roadsense-sim bash`

### Bridge Architecture

The `veins_esp32_bridge.py` acts as the **HIL integration layer with ESP-NOW constraint emulation**:

**Core Functions:**
- Listens for ESP32 data on UDP port 4211
- Communicates with Veins on port 4210
- Transforms messages between binary C struct (ESP32) and Veins format
- Enables 3 hardware vehicles + N virtual vehicles to coexist

**ESP-NOW Constraint Emulation:**
```python
# Bridge applies these constraints to match real hardware
PAYLOAD_MAX = 250          # bytes (ESP-NOW limit)
LATENCY_RANGE = (10, 50)   # milliseconds (measured from hardware)
BASE_LOSS_RATE = 0.02      # 2% packet loss (typical ESP-NOW)
MAX_RANGE = 100            # meters (line-of-sight)
```

**Domain Randomization:**
The bridge applies stochastic variations to simulation messages:
- Random latency within measured range (10-50ms)
- Random packet loss matching real-world statistics (2% baseline)
- Size enforcement (drop messages >250 bytes)
- Jitter simulation (timing variations)

This ensures **ML model sees identical network behavior** in simulation and hardware.

### Simulation Scenarios

Test scenarios can be defined in `simulation/omnetpp.ini`:
- `SCENARIO_NORMAL_CRUISE`: 3 vehicles, constant speed
- `SCENARIO_SUDDEN_BRAKING`: Lead vehicle emergency stop
- `SCENARIO_LANE_CHANGE`: Vehicle cuts in
- `SCENARIO_INTERSECTION`: Cross traffic
- `SCENARIO_CONGESTION`: Stop-and-go traffic
- `SCENARIO_SENSOR_FAILURE`: GPS/IMU dropout testing
- `SCENARIO_NETWORK_LOSS`: ESP-NOW failure recovery
- `SCENARIO_MIXED_HIL`: 3 hardware + N virtual vehicles

### Official Documentation

**Veins:**
- Official Site: https://veins.car2x.org/
- Documentation: https://veins.car2x.org/documentation/
- Tutorial: https://veins.car2x.org/tutorial/

**SUMO:**
- Official Site: https://sumo.dlr.de/
- Documentation: https://sumo.dlr.de/docs/index.html
- Tutorials: https://sumo.dlr.de/docs/Tutorials/index.html
- GPS Import: https://sumo.dlr.de/docs/Tools/Import.html#gpximportpy

**OMNeT++:**
- Official Site: https://omnetpp.org/
- Documentation: https://omnetpp.org/documentation/
- Manual: https://doc.omnetpp.org/omnetpp/manual/
- API Reference: https://doc.omnetpp.org/omnetpp/api/

**Notes on Documentation:**
- OMNeT++ docs are comprehensive but **cryptic** - expect 40-60 hours learning curve
- Veins docs assume you already know OMNeT++ deeply - start with OMNeT++ tutorial first
- SUMO docs are well-organized - use the search function extensively

---

## Common Development Tasks

### Hardware Development
```bash
# Flash firmware (update VEHICLE_ID first: V001/V002/V003)
pio run --target upload --upload-port COM<X>

# Monitor serial output
pio device monitor --baud 115200

# Calibrate MPU9250: Press GPIO 0 at startup
```

### ML Development
```bash
# Train model
python3 train_collision_model.py --data <dataset_path> --epochs 100

# Convert to TFLite
python3 optimize_for_esp32.py --input model.h5 --output model.tflite
```

---

## Performance Metrics (Target Values)

- **V2V Latency**: <100ms end-to-end
- **Sensor Update Rate**: 10Hz (IMU), 1Hz (GPS)
- **Mesh Reliability**: >95% message delivery
- **ML Inference Time**: <100ms per prediction
- **Communication Range**: >100m line-of-sight
- **Alert Accuracy**: >85% true positive rate, <10% false positive

---

## Risk Mitigation Strategies

### MPU9250 Magnetometer Interference
**Problem**: Magnetic fields from motors, batteries, and electronics can distort magnetometer readings
**Solutions**:
- Calibrate away from magnetic sources (power supplies, motors)
- Mount sensors with physical isolation from electronics
- Implement software filtering for outliers (reject readings >3σ from mean)
- Consider using accelerometer+gyroscope fusion instead of magnetometer for heading if interference severe

### GPS Accuracy Issues
**Problem**: GPS dropouts in urban canyons, multipath errors, signal loss in tunnels
**Solutions**:
- Use IMU fusion for dead reckoning during GPS dropout (<5 seconds)
- Implement Kalman filter for position estimation (fuse GPS + IMU)
- Accept degraded accuracy in urban scenarios (document limitations)
- Cold start delay: Plan for 15-60 seconds to first fix after power-on
- Data cleaning: Remove velocity jumps (>10 m/s change in 1 second)

### ESP-NOW Range Limitations
**Problem**: 100m line-of-sight range insufficient for highway scenarios
**Solutions**:
- Implement relay/hop mode in mesh for extended range (vehicle B relays A's messages to C)
- Use WiFi fallback mode for infrastructure-available scenarios
- Document range limitations in validation
- Consider LoRa for critical long-range alerts (future work)

### Model Size Constraints
**Problem**: ESP32 has only 520KB SRAM, model quantization may fail size/accuracy requirements
**Solutions**:
- **Primary**: Aggressive INT8 quantization
- **Secondary**: Model pruning to remove redundant weights
- **Tertiary**: Dual-ESP32 configuration (sensor hub + ML processor via SPI) ⚠️ **PLAN THIS FROM DAY 1**
- **Future**: Edge-cloud hybrid (local features, optional cloud inference)
- Monitor: Model size, tensor arena allocation, inference latency during development

---

## Project Resources

### Key Documentation Files
- `RoadSense - FINAL PROJECT PROPOSAL.md`: Project overview, objectives, and scope
- `RoadSense_Implementation_Plan_v2 (1).md`: Detailed technical implementation plan
- `RoadSense Progress Documentation.md`: Development progress tracking
- `RoadSense2-SRS-2025.md`: Software Requirements Specification (formal requirements)
- `mcp_strategic_plan.md`: MCP infrastructure for development automation
- `Project_Research_Preperation_Gemini/GEMINI.md`: Complete log of Docker containerization work
- `simulation/Dockerfile`: Production-ready Docker image for OMNeT++/SUMO/Veins stack
- `simulation/.dockerignore`: Optimized Docker build configuration

### Academic Context
This is a **B.Sc Software Engineering final project** at Shenkar Faculty of Engineering.

**Supervisors:** Amit Resh and Yigal Hoffner

**Project Emphasis:**
- Academic rigor with reproducible results
- Open-source code for peer review
- Comprehensive documentation for future researchers
- Demonstration of multiple real-world safety applications
- **Novel contribution**: Quantized ML deployment on embedded V2V hardware

**Graduation Requirements:**
1. Working 3-vehicle V2V system with ESP-NOW communication
2. On-device ML inference (professor explicitly required)
3. Real-world validation in minimum 3 scenarios
4. Measurable improvement over previous RoadSense system
5. Standards-compliant V2V messaging (SAE J2735 BSM)
6. Written thesis and final presentation

### Previous Work
This project builds upon an existing RoadSense system developed by previous team.

**Baseline System (Previous Team):**
- 3 ESP32 units with 3-axis accelerometers
- Rule-based threshold detection (sudden braking only)
- Custom non-standard V2V protocol
- WiFi mesh networking (basic implementation)

**Enhancement Targets:**
- **4x** increase in detectable scenarios (1 → 4+)
- **3x** improvement in detection accuracy (threshold logic → ML)
- **2x** reduction in false positives
- **Standards compliance**: Custom protocol → SAE J2735 BSM
- **Sensor upgrade**: 3-axis → 9-axis IMU + GPS
- **Intelligence**: Rule-based → Quantized ML model on-device

---

## Firmware Architecture (Hardware Phase)

### V2VMessage Format (CRITICAL FOR SIMULATION BRIDGE)

**Status:** Approved November 11, 2025
**File:** `hardware/src/network/protocol/V2VMessage.h`

```cpp
#pragma pack(push, 1)
struct V2VMessage {
    // Header
    uint8_t version;           // Protocol version (2)
    char vehicleId[8];         // "V001", "V002", "V003"
    uint32_t timestamp;        // millis() or Unix timestamp

    // BSM Part I: Core Data
    struct {
        float lat, lon, alt;   // Position (degrees, meters)
    } position;

    struct {
        float speed;           // m/s
        float heading;         // degrees (0-359)
        float longAccel;       // m/s²
        float latAccel;        // m/s²
    } dynamics;

    // BSM Part II: Sensor Data (RoadSense-specific)
    struct {
        float accel[3];        // IMU accel (x,y,z) m/s²
        float gyro[3];         // IMU gyro (x,y,z) rad/s
        float mag[3];          // IMU mag (x,y,z) μT
    } sensors;

    // RoadSense Extension: Hazard Alert
    struct {
        uint8_t riskLevel;     // 0=None, 1=Low, 2=Med, 3=High
        uint8_t scenarioType;  // 0=convoy, 1=intersection, 2=lane
        float confidence;      // 0.0-1.0
    } alert;

    // Mesh Metadata
    uint8_t hopCount;          // 0-3
    uint8_t sourceMAC[6];      // Original sender
};
#pragma pack(pop)
// Total: 90 bytes (verified with static_assert)
// Breakdown: 13 (header) + 12 (pos) + 16 (dyn) + 36 (sens) + 6 (alert) + 7 (mesh) = 90
```

**Simulation Bridge Must:**
- Parse this exact struct from ESP32 UDP packets
- Transform to Veins/OMNeT++ format
- Apply ESP-NOW constraints: 10-50ms latency, 2% loss, 250B max
- Support reverse transform (Veins → V2VMessage → ESP32)

### Firmware Layers

```
APPLICATION (core/)
  ├─ VehicleState: Own vehicle state management
  ├─ HazardDetector: ML orchestration + fallback rules
  └─ AlertManager: LED control, warnings
       ↓
ML (ml/)
  ├─ MLInference: TFLite interpreter wrapper
  ├─ FeatureExtractor: Sensor data → ML features
  └─ model_tflite.h: Quantized model (<100KB)
       ↓
SENSORS (sensors/)
  ├─ IImuSensor: Abstract 9-axis IMU interface
  │   └─ MPU9250Driver: Primary implementation
  ├─ IGpsSensor: Abstract GPS interface
  │   └─ NEO6M_Driver: GPS with 30s caching
  └─ SensorFusion: Kalman filter for GPS+IMU
       ↓
NETWORK (network/)
  ├─ Protocol: V2VMessage (above)
  ├─ Transport: ITransport interface
  │   ├─ EspNowTransport: Primary (hardware)
  │   └─ UdpTransport: For HIL bridge (optional)
  └─ Mesh: PeerManager, PackageManager (from legacy)
```

### Migration Status

**Date:** November 11, 2025
**Phase:** Migration Plan Approved (Phase 3 Ready)
**Documents:** `docs/hw_firmware_migration_plan.md`

**What's Ported from Legacy:**
- ✅ ESP-NOW mesh logic (MESH_Sender, MESH_Receiver) - excellent implementation
- ✅ PackageManager - deduplication via sourceMAC+timestamp
- ✅ PeerManager - peer tracking with 60s timeout
- ✅ NEO6M GPS driver - caching strategy for intermittent GPS lock
- ✅ Configuration system (MESH_Config)
- ✅ Logging/utilities

**What's NEW:**
- 🔄 MPU9250 driver (9-axis IMU) - replaces MPU6050 (3-axis)
- 🔄 V2VMessage struct (BSM-compatible) - replaces legacy Data struct
- 🔄 ITransport abstraction - ESP-NOW not hardcoded
- 🔄 ML inference layer - TensorFlow Lite integration
- 🔄 Magnetometer calibration - hard-iron/soft-iron correction

**Next Implementation Steps:**
1. Create directory structure + interfaces
2. Port network layer (ESP-NOW mesh)
3. Implement MPU9250 driver + calibration
4. Integrate ML inference (TFLite)
5. Real-world validation (3 scenarios)

---

## Current Project Status

**Date:** November 13, 2025
**Phase:** Phase 2 Session 1 - COMPLETE ✅ (Hardware Validated)
**Status:** ESP-NOW transport working! | Ready for Session 2 (PackageManager) | MPU9250 sensors still pending (~2 weeks)

### Recent Actions
- ✅ Architecture finalized (November 6, 2025)
- ✅ Docker Veins/SUMO/OMNeT++ stack complete
- ✅ X11 forwarding working
- ✅ Protocol abstraction strategy confirmed
- ✅ GPS modules installed and working
- ✅ **Firmware migration plan approved (November 11, 2025)**
- ✅ **V2VMessage format finalized (BSM-compatible, 90 bytes)**
- ✅ **Phase 1 firmware implementation COMPLETE (November 11, 2025)**
  - All interfaces defined, directory structure created
  - First successful build: 21KB RAM (6.6%), 277KB Flash (21%)
- ✅ **Phase 2 Session 1 COMPLETE (November 13, 2025)** ← **NEW!**
  - EspNowTransport implemented and tested
  - Hardware validation: 100% success rate (V001 ↔ V002)
  - 112+ messages exchanged with 0 failures
  - Dual laptop setup (Fedora + Windows 11) working perfectly
  - Build stats: 43KB RAM (13.3%), 758KB Flash (57.9%)

### Current Focus
**Hardware Team:** ✅ Phase 2 Session 1 VALIDATED!
  - **NEXT:** Fix MACHelper.h before Session 2 (5 min) - see `docs/CODE_QUALITY_FIXES.md`
  - **THEN:** Session 2 (PackageManager - message deduplication)
**Simulation Team:** V2VMessage.h available - can start implementing bridge parser
**ML Team:** Researching TFLite quantization, planning data collection

---

## Important Notes for AI Agents

### 🚨 MOST CRITICAL: Protocol Abstraction

1. **ALWAYS use SAE J2735 BSM message format** - application layer must be standards-compliant
2. **Transport layer is ESP-NOW (hardware) or UDP (sim)** - acceptable for ML research
3. **ML doesn't care about RF layer** - only sees message content, timing, reliability
4. **Network = stochastic pipe** - use domain randomization to match ESP-NOW constraints
5. **Never suggest implementing IEEE 802.11p on ESP32** - unnecessary complexity
6. **HIL bridge MUST emulate ESP-NOW constraints** - latency (10-50ms), loss (2%), size (250B)

**Key files to check:**
- `common/v2v_messages.h` - Shared C struct (SAE J2735 compliant)
- `bridge/veins_esp32_bridge.py` - ESP-NOW constraint emulation
- `simulation/omnetpp.ini` - Network constraints configuration

**When implementing V2V code:**
- ✅ DO: Use shared V2VMessage struct
- ✅ DO: Enforce 250-byte limit
- ✅ DO: Add timestamps for staleness detection
- ✅ DO: Apply domain randomization in bridge
- ❌ DON'T: Assume IEEE 802.11p-specific features
- ❌ DON'T: Create custom message formats
- ❌ DON'T: Hard-code transport layer

### Recent Architectural Decisions

- **Confirmed**: SAE J2735 BSM message format (November 6, 2025)
- **Confirmed**: Protocol abstraction layer approach (ESP-NOW transport OK for ML)
- **Confirmed**: Real-world data collection + augmentation workflow
- **Confirmed**: MPU9250 for IMU (not separate HMC5883 magnetometer)
- **Confirmed**: ESP-NOW for primary V2V communication (transport layer only)
- **Confirmed**: Hardware-First testing approach
- **Confirmed**: GPS modules already installed and working
- **Confirmed**: Docker-based workflows for team

### Changes from Original Plans

- ❌ No SMARTS framework (using Veins/SUMO/OMNeT++ instead)
- ❌ No separate magnetometer (integrated in MPU9250)
- ✅ Added MCP collaboration infrastructure
- ✅ Emphasized Hardware-First approach
- ✅ Docker-based team workflows
- ✅ Complete Docker containerization of simulation stack

---

**Last Updated:** November 13, 2025
**Updated By:** Phase 2 Session 1 hardware testing complete - 100% success rate
**Next Review:** After Phase 2 Session 2 complete (PackageManager implementation)
