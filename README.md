# SafeRide V2V System (formerly RoadSense)

**Vehicle-to-Vehicle (V2V) hazard detection using ESP32, Deep Reinforcement Learning, and ESP-NOW mesh networking.**

**SafeRide** (formerly known as RoadSense) enables infrastructure-free collision avoidance by allowing vehicles to share kinematic data (position, velocity, heading) directly with one another. The system uses a **Deep Sets** Reinforcement Learning agent to evaluate collision risk in dynamic environments with varying numbers of peers.

---

## 🚀 Key Features

*   **Infrastructure-Free:** Relies entirely on peer-to-peer ESP-NOW communication (no 4G/5G/WiFi AP required).
*   **Deep Sets Architecture:** Handles dynamic numbers of neighboring vehicles (n-element problem) without manual sorting or padding.
*   **Sim2Real Gap Bridging:** Custom **ESP-NOW Emulator** replicates hardware-measured packet loss, latency, and jitter within the simulation.
*   **Simulation-First:** Trained in SUMO (Traffic) + Gymnasium (RL) before hardware deployment.

---

## ⚡ Quick Start (Simulation & ML)

The project uses a containerized development environment for training and simulation.

### Prerequisites
*   **Docker** (Desktop or Engine)
*   **Python 3.10+** (for helper scripts)

### Running the Environment

1.  **Navigate to the ML directory:**
    ```bash
    cd roadsense-v2v/ml
    ```

2.  **Run Unit Tests (Builds Docker image automatically):**
    ```bash
    ./run_docker.sh test
    ```

3.  **Run Visualization Demo (SUMO GUI):**
    *   *Windows/Linux:* 
        ```bash
        ./run_docker.sh demo
        ```
    *   *Note:* Requires X11/WSLg setup (see [ML Docker Guide](../docs/20_KNOWLEDGE_BASE/ML_DOCKER_ENVIRONMENT.md)).

4.  **Run Training:**
    ```bash
    ./run_docker.sh train
    ```

---

## 🛠️ Hardware Stack

*   **MCU:** ESP32 DevKit V1
*   **IMU:** MPU6500 (Accel/Gyro)
*   **Compass:** QMC5883L (Magnetometer)
*   **GPS:** NEO-6M
*   **Storage:** MicroSD Module (Data Logging)

Firmware code is located in `hardware/` and uses **PlatformIO**.

---

## 📂 Project Structure

```
roadsense-v2v/
├── hardware/           # ESP32 Firmware (PlatformIO)
├── ml/                 # Machine Learning & Simulation Core
│   ├── envs/           # Gymnasium Environments (ConvoyEnv)
│   ├── espnow_emulator/# Python-based Network Emulator
│   ├── models/         # PPO Policies & Deep Sets Extractors
│   ├── scenarios/      # SUMO Traffic Scenarios (XML)
│   ├── training/       # Training Scripts
│   └── tests/          # Pytest Suite
├── common/             # Shared C/C++ Headers (V2VMessage.h)
├── docs/               # Documentation (Architecture, Plans)
└── scripts/            # Utility scripts
```

## 📖 Documentation

*   **[Deep Sets Architecture](../docs/00_ARCHITECTURE/DEEP_SETS_N_ELEMENT_ARCHITECTURE.md):** How we solve the n-peer problem.
*   **[ESP-NOW Emulator](../docs/00_ARCHITECTURE/ESPNOW_EMULATOR_DESIGN.md):** Design of the Sim2Real network layer.
*   **[Hardware Progress](../docs/20_KNOWLEDGE_BASE/HARDWARE_PROGRESS.md):** Current status of physical units.

---

## License

Private Repository. All rights reserved.