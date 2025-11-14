# Hardware Firmware Migration Plan
**RoadSense V2V Enhanced System - Legacy to Monorepo Migration**

**Status**: Approved (November 11, 2025)
**Phase**: Ready for Implementation (Phase 3)

---

## Executive Summary

This document provides the **complete migration strategy** for porting legacy RoadSense firmware (MPU6050-based) to the new monorepo architecture with MPU9250 9-axis IMU support, standards-compliant V2V messaging, and ML inference capabilities.

**Key Goals**:
- ✅ Preserve working ESP-NOW mesh networking
- ✅ Upgrade from MPU6050 (3-axis) to MPU9250 (9-axis IMU)
- ✅ Implement SAE J2735 BSM-compatible message format
- ✅ Add on-device ML inference (TensorFlow Lite)
- ✅ Maintain backward compatibility during transition

---

## Architecture Overview

### Layered Firmware Architecture

```
┌────────────────────────────────────────────────┐
│  APPLICATION LAYER (core/)                     │
│  VehicleState, HazardDetector, AlertManager   │
└────────────────┬───────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────┐
│  ML LAYER (ml/)                                │
│  MLInference, FeatureExtractor, TFLite model  │
└────────────────┬───────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────┐
│  SENSOR LAYER (sensors/)                       │
│  IImuSensor (MPU9250), IGpsSensor (NEO-6M)    │
└────────────────┬───────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────┐
│  NETWORK LAYER (network/)                      │
│  ├─ Protocol: V2VMessage (BSM format)         │
│  ├─ Transport: ESP-NOW (ITransport interface) │
│  └─ Mesh: PeerManager, PackageManager         │
└────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Interface-Based Design**: Abstract interfaces (IImuSensor, ITransport) for swappable implementations
2. **Transport Abstraction**: ESP-NOW is ONE transport option, not hardcoded
3. **Protocol First**: SAE J2735 BSM message format is sacred, transport is flexible
4. **Layered Dependencies**: Upper layers depend on lower layer interfaces, not implementations

---

## Target Directory Structure

```
hardware/
├── platformio.ini
├── src/
│   ├── main.cpp                      ← Entry point
│   ├── config.h                      ← Vehicle ID, pins, constants
│   │
│   ├── sensors/                      ← SENSOR LAYER
│   │   ├── imu/
│   │   │   ├── IImuSensor.h          ← Abstract 9-axis IMU interface
│   │   │   ├── MPU9250Driver.cpp/.h  ← Primary driver (NEW)
│   │   │   ├── MPU6050Driver.cpp/.h  ← Legacy compat (OPTIONAL)
│   │   │   └── Calibration.cpp/.h    ← Magnetometer calibration
│   │   ├── gps/
│   │   │   ├── IGpsSensor.h          ← Abstract GPS interface
│   │   │   └── NEO6M_Driver.cpp/.h   ← GPS driver (PORTED)
│   │   └── fusion/
│   │       └── SensorFusion.cpp/.h   ← Kalman filter
│   │
│   ├── network/                      ← NETWORK LAYER
│   │   ├── transport/
│   │   │   ├── ITransport.h          ← Abstract transport
│   │   │   ├── EspNowTransport.cpp/.h← ESP-NOW impl
│   │   │   └── UdpTransport.cpp/.h   ← UDP (optional)
│   │   ├── protocol/
│   │   │   ├── V2VMessage.h          ← BSM struct
│   │   │   ├── MessageBuilder.cpp/.h
│   │   │   └── MessageParser.cpp/.h
│   │   └── mesh/
│   │       ├── MeshManager.cpp/.h    ← Orchestrator
│   │       ├── PeerManager.cpp/.h    ← PORTED
│   │       ├── PackageManager.cpp/.h ← PORTED
│   │       └── legacy/               ← Reference only
│   │
│   ├── ml/                           ← ML LAYER
│   │   ├── MLInference.cpp/.h
│   │   ├── FeatureExtractor.cpp/.h
│   │   └── model_tflite.h
│   │
│   ├── core/                         ← APPLICATION LAYER
│   │   ├── VehicleState.cpp/.h
│   │   ├── HazardDetector.cpp/.h
│   │   └── AlertManager.cpp/.h
│   │
│   └── utils/                        ← UTILITIES
│       ├── Logger.cpp/.h             ← PORTED
│       ├── Storage.cpp/.h
│       └── Helpers.h                 ← PORTED
│
├── include/
└── test/
```

---

## V2VMessage Format (BSM-Compatible)

**Critical for simulation bridge integration**

```cpp
#pragma pack(push, 1)  // No padding

struct V2VMessage {
    // Header (8 bytes)
    uint8_t version;           // Protocol version (2)
    char vehicleId[8];         // "V001", "V002", "V003"
    uint32_t timestamp;        // millis() or Unix timestamp

    // BSM Part I: Core Data Set (28 bytes)
    struct Position {
        float lat;             // degrees
        float lon;             // degrees
        float alt;             // meters
    } position;

    struct Dynamics {
        float speed;           // m/s
        float heading;         // degrees (0-359)
        float longAccel;       // m/s²
        float latAccel;        // m/s²
    } dynamics;

    // BSM Part II: RoadSense Sensor Data (36 bytes)
    struct Sensors {
        float accel[3];        // m/s² (x, y, z)
        float gyro[3];         // rad/s (x, y, z)
        float mag[3];          // μT (x, y, z)
    } sensors;

    // RoadSense Extension: Hazard Alert (6 bytes)
    struct Alert {
        uint8_t riskLevel;     // 0=None, 1=Low, 2=Med, 3=High
        uint8_t scenarioType;  // 0=convoy, 1=intersection, 2=lane
        float confidence;      // 0.0-1.0
    } alert;

    // Mesh Metadata (7 bytes)
    uint8_t hopCount;          // 0-3
    uint8_t sourceMAC[6];      // Original sender MAC
};

#pragma pack(pop)

// Total: 90 bytes (well under 250-byte ESP-NOW limit)
```

---

## Key Interfaces

### IImuSensor (9-Axis IMU)

```cpp
class IImuSensor {
public:
    struct ImuData {
        float accel[3];      // m/s² (x, y, z)
        float gyro[3];       // rad/s (x, y, z)
        float mag[3];        // μT (x, y, z)
        uint32_t timestamp;
    };

    virtual bool begin() = 0;
    virtual ImuData read() = 0;
    virtual bool calibrate() = 0;          // Magnetometer calibration
    virtual bool loadCalibration() = 0;    // From NVS
    virtual bool saveCalibration() = 0;
    virtual bool isCalibrated() const = 0;
};
```

### ITransport (Network Abstraction)

```cpp
class ITransport {
public:
    using ReceiveCallback = std::function<void(const uint8_t*, size_t)>;

    virtual bool begin() = 0;
    virtual bool send(const uint8_t* data, size_t len) = 0;
    virtual void onReceive(ReceiveCallback callback) = 0;
    virtual size_t getMaxPayload() const = 0;
    virtual const char* getName() const = 0;
};
```

---

## Migration Checklist

### Phase 1: Foundation & Structure (Week 1-2)

#### Step 1.1: Create Directory Structure
- [ ] Create `hardware/src/sensors/imu/`
- [ ] Create `hardware/src/sensors/gps/`
- [ ] Create `hardware/src/network/transport/`
- [ ] Create `hardware/src/network/protocol/`
- [ ] Create `hardware/src/network/mesh/`
- [ ] Create `hardware/src/ml/`
- [ ] Create `hardware/src/core/`
- [ ] Create `hardware/src/utils/`

#### Step 1.2: Define Abstract Interfaces
- [ ] Write `IImuSensor.h` interface
- [ ] Write `IGpsSensor.h` interface
- [ ] Write `ITransport.h` interface
- [ ] Write `VehicleState.h` struct
- [ ] Write `V2VMessage.h` struct (BSM-compatible)
- [ ] Add size check: `static_assert(sizeof(V2VMessage) <= 250)`

#### Step 1.3: Copy Legacy Utilities
- [ ] Copy `Helpers.h` to `hardware/src/utils/`
- [ ] Copy `MACHelper.h` to `hardware/src/utils/`
- [ ] Copy `SerialConsoleMiddleware` → `Logger.cpp`
- [ ] Verify utilities compile independently

---

### Phase 2: Network Layer Migration (Week 2-3)

#### Step 2.1: Port ESP-NOW Mesh Logic
- [ ] Copy `MESH_Sender/Receiver` to `legacy/` (reference)
- [ ] Copy `PackageManager` to `mesh/`
- [ ] Copy `PeerManager` to `mesh/`
- [ ] Copy `CleanupManager` to `mesh/`

#### Step 2.2: Create Transport Abstraction
- [ ] Implement `EspNowTransport` (ITransport interface)
- [ ] Preserve exact ESP-NOW init sequence:
  ```cpp
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();  // ← CRITICAL
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_now_init();
  ```
- [ ] Test broadcast with simple payload

#### Step 2.3: Integrate V2VMessage Format
- [ ] Implement `MessageBuilder`
- [ ] Implement `MessageParser`
- [ ] Update `PackageManager` to use `V2VMessage`
- [ ] Update `PeerManager` to use `V2VMessage`
- [ ] Verify `sizeof(V2VMessage) <= 250`

#### Step 2.4: Validate Mesh Networking
- [ ] Flash 2 ESP32 units
- [ ] Verify bidirectional exchange
- [ ] Test hop-count relay (3 units in a line)
- [ ] Verify deduplication
- [ ] Measure latency (<50ms single hop)

---

### Phase 3: Sensor Layer Migration (Week 3-4)

#### Step 3.1: Port GPS Driver
- [ ] Copy `NEO6M_GPS_Service` → `NEO6M_Driver`
- [ ] Implement `IGpsSensor` interface
- [ ] Test standalone (print lat/lon)
- [ ] Verify GPS caching (30s timeout)

#### Step 3.2: Implement MPU9250 Driver (CRITICAL)
- [ ] Create `MPU9250Driver` (IImuSensor)
- [ ] Initialize Bolder Flight Systems library
- [ ] Implement `begin()` - sensor init
- [ ] Implement `read()` - accel + gyro + mag
- [ ] Implement `calibrate()` - figure-8 routine
- [ ] Implement NVS storage (Preferences library)
- [ ] Test on bench with serial output
- [ ] Validate calibration

#### Step 3.3: (Optional) Legacy MPU6050 Compatibility
- [ ] Create `MPU6050Driver` wrapping legacy code
- [ ] Add compile flag: `#define USE_MPU6050_COMPAT`

#### Step 3.4: Integrate Sensors
- [ ] Update `main.cpp` to use `IImuSensor*`
- [ ] Implement basic sensor fusion
- [ ] Populate `VehicleState`
- [ ] Test: sensors → VehicleState → V2VMessage → ESP-NOW

---

### Phase 4: ML Integration (Week 5-6)

#### Step 4.1: Prepare ML Stub
- [ ] Create `MLInference` (stub)
- [ ] Create `FeatureExtractor`
- [ ] Implement feature extraction (30 floats)
- [ ] Test with mock data

#### Step 4.2: Integrate TensorFlow Lite
- [ ] Add TFLite to `platformio.ini`
- [ ] Create placeholder `model_tflite.h`
- [ ] Implement TFLite interpreter
- [ ] Allocate tensor arena (10KB)

#### Step 4.3: Deploy Real Model
- [ ] Replace with trained model
- [ ] Test latency (<100ms)
- [ ] Test accuracy
- [ ] Implement dual-ESP32 fallback if needed

---

### Phase 5: Core Application Logic (Week 6-7)

#### Step 5.1: Implement HazardDetector
- [ ] Create `HazardDetector`
- [ ] Integrate FeatureExtractor → MLInference
- [ ] Implement fallback logic
- [ ] Test with 3 vehicles

#### Step 5.2: Implement AlertManager
- [ ] Create `AlertManager`
- [ ] Implement LED control (GPIO 2)
- [ ] Map risk levels to LED patterns
- [ ] Test thresholds

#### Step 5.3: Orchestrate Main Loop
- [ ] 10Hz sensor sampling
- [ ] ML inference every 100ms
- [ ] V2V broadcast every 100ms
- [ ] Hazard broadcast (20Hz if HIGH risk)

---

### Phase 6: Validation & Testing (Week 7-8)

#### Step 6.1: Bench Testing
- [ ] Test 3 units on desk
- [ ] Verify mesh range (100m)
- [ ] Measure latency (sensor → broadcast)
- [ ] Check for memory leaks (1 hour run)

#### Step 6.2: Real-World Testing
- [ ] Deploy in 3 test vehicles
- [ ] Test convoy scenario
- [ ] Test intersection scenario
- [ ] Test lane-change scenario
- [ ] Record GPS + IMU traces

#### Step 6.3: Performance Validation
- [ ] Measure accuracy (>85% requirement)
- [ ] Measure false positives (<10%)
- [ ] Measure latency (<100ms)
- [ ] Compare to baseline
- [ ] Document results

---

## Success Criteria

- [ ] All 3 ESP32 units communicate via ESP-NOW
- [ ] MPU9250 + GPS working on all units
- [ ] Magnetometer calibrated
- [ ] ML inference on-device (<100ms)
- [ ] ≥3 scenarios detected
- [ ] V2VMessage BSM-compatible
- [ ] End-to-end latency <100ms
- [ ] Detection accuracy >85%
- [ ] False positive rate <10%
- [ ] Improvement over baseline documented

---

## Critical Legacy Constraints

### ESP-NOW Initialization Sequence (DO NOT CHANGE)

```cpp
// CRITICAL: This order is fragile
WiFi.mode(WIFI_STA);
WiFi.disconnect();  // Removes STA auto-connect interference
esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
esp_now_init();
esp_now_register_send_cb(sentCallback);
esp_now_add_peer(&broadcastPeerInfo);  // FF:FF:FF:FF:FF:FF
```

### Deduplication Logic
- Uses `sourceMAC + timestamp` (millis())
- PackageManager tracks seen messages
- 60-second timeout for cleanup

### Network Limits
- Hop count: MAX 3
- Payload: MAX 250 bytes
- Cleanup interval: 1000ms
- Package timeout: 60 seconds

---

## MPU6050 → MPU9250 Migration Notes

**NOT a drop-in replacement:**
- MPU9250 gyroscope requires explicit init (MPU6050 doesn't)
- AK8963 magnetometer via I2C passthrough (complex)
- Different scaling factors
- Requires magnetometer calibration (hard-iron/soft-iron)

**Strategy:**
1. Use Bolder Flight Systems MPU9250 library (already in platformio.ini)
2. Implement IImuSensor interface
3. Add NVS calibration storage
4. Test side-by-side with MPU6050 (compile-time switch)
5. Deprecate MPU6050 after validation

---

## Legacy Code Reuse Summary

### ✅ KEEP AS-IS (Direct Port)
- ESP-NOW mesh (MESH_Sender, MESH_Receiver)
- PackageManager (deduplication)
- PeerManager (peer tracking)
- CleanupManager (garbage collection)
- NEO6M_GPS_Service (GPS caching)
- MESH_Config (configuration singleton)
- Middleware (logging, delays)
- Helpers (MAC comparison, utilities)

### 🔄 REFACTOR (Adapt)
- Data struct → V2VMessage (BSM format)
- IAccelerometerSensor → IImuSensor (9-axis)
- Mock generators (extend to 9-axis)
- main.cpp (MPU9250 init)
- MESH_Constants.h (MPU9250 pins)

### ❌ DROP (Rewrite)
- MPU6050_Service (tightly coupled to hardware)
- MPU6050 register maps

---

## Next Steps

**Once this plan is approved:**
1. Begin Phase 1 (foundation setup)
2. Use small, reviewable commits
3. Test each component independently
4. Document all decisions in progress tracker
5. Update CLAUDE.md after each major milestone

---

**Document Version:** 1.0
**Last Updated:** November 11, 2025
**Next Review:** After Phase 1 completion
