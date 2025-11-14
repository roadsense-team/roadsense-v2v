# Hardware Firmware Implementation Progress

**Project:** RoadSense V2V Enhanced System - Firmware Migration
**Started:** November 11, 2025
**Status:** ✅ Phase 2 Session 1 COMPLETE (Hardware Validated) | Ready for Session 2
**Last Updated:** November 13, 2025 - Session 5 (Hardware Testing Complete)

---

## Quick Status Overview

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| Phase 1: Foundation & Structure | ✅ **COMPLETE** | 13/13 | Directory structure, interfaces, utilities |
| Phase 2: Network Layer | 🔄 **IN PROGRESS** | 7/13 | Session 1 COMPLETE ✅ (hw validated!) |
| Phase 3: Sensor Layer | ⏳ Not Started | 0/14 | MPU9250 driver, GPS, calibration |
| Phase 4: ML Integration | ⏳ Not Started | 0/9 | TFLite, feature extraction, inference |
| Phase 5: Core Application | ⏳ Not Started | 0/9 | HazardDetector, AlertManager, main loop |
| Phase 6: Validation & Testing | ⏳ Not Started | 0/12 | Bench testing, real-world scenarios |

**Overall Progress:** 20/70 tasks (28.6%)

**🎯 NEXT MILESTONE:** Phase 2 Session 2 - PackageManager (message deduplication)

---

## Implementation Sessions Log

### Session 1: Migration Planning (November 11, 2025)

**Participants:** Hardware team + Claude
**Duration:** ~2 hours
**Accomplishments:**
- ✅ Analyzed legacy firmware (layer_1 repository)
- ✅ Designed new firmware architecture (layered, interface-based)
- ✅ Defined V2VMessage format (BSM-compatible, 85 bytes)
- ✅ Created migration checklist (70 tasks across 6 phases)
- ✅ Approved MPU6050 → MPU9250 migration strategy
- ✅ Documented everything in `docs/hw_firmware_migration_plan.md`
- ✅ Updated CLAUDE.md with critical info for simulation team

**Decisions Made:**
1. Use layered architecture: Sensors → Network → ML → Application
2. Abstract interfaces (IImuSensor, ITransport) for flexibility
3. Port excellent ESP-NOW mesh logic from legacy (minimal changes)
4. Use Bolder Flight Systems MPU9250 library (already in platformio.ini)
5. V2VMessage struct is 85 bytes (well under 250-byte limit)
6. Magnetometer calibration required (NVS storage)

**Key Artifacts Created:**
- `docs/hw_firmware_migration_plan.md` - Complete migration guide
- `docs/HARDWARE_PROGRESS.md` - This file (progress tracker)
- Updated `CLAUDE.md` with V2VMessage format + architecture

**Next Session Plan:**
- Start Phase 1: Create directory structure
- Define abstract interfaces (IImuSensor, ITransport, etc.)
- Copy legacy utilities (Helpers, Logger)

---

### Session 2: Phase 1 Implementation (November 11, 2025)

**Participants:** Hardware team + Claude
**Duration:** ~1.5 hours
**Status:** ✅ **PHASE 1 COMPLETE**

**Accomplishments:**
- ✅ Created complete directory structure (sensors/, network/, ml/, core/, utils/)
- ✅ Created platformio.ini with MPU9250 library and TFLite dependencies
- ✅ Created config.h with all vehicle constants, pins, performance targets
- ✅ Defined IImuSensor.h interface (9-axis IMU with calibration)
- ✅ Defined IGpsSensor.h interface (GPS with caching)
- ✅ Defined ITransport.h interface (network abstraction)
- ✅ Defined VehicleState.h struct (complete vehicle state)
- ✅ Defined V2VMessage.h (BSM-compatible, 85 bytes)
- ✅ Copied MACHelper.h from legacy
- ✅ Ported SerialConsoleMiddleware → Logger (simplified)
- ✅ Created skeleton main.cpp with LED blink
- ✅ Installed PlatformIO in VSCode (Fedora laptop)
- ✅ **FIRST SUCCESSFUL BUILD** - No errors!

**Build Statistics:**
- **RAM Usage:** 6.6% (21,528 / 327,680 bytes) - Excellent headroom
- **Flash Usage:** 21.1% (276,941 / 1,310,720 bytes) - Plenty of space
- **Firmware Size:** 276 KB
- **Status:** ✅ Clean compilation, zero errors

**Files Created (12 total):**
```
hardware/
├── platformio.ini
├── include/
│   ├── IImuSensor.h (9-axis interface)
│   ├── IGpsSensor.h (GPS interface)
│   ├── ITransport.h (network interface)
│   └── VehicleState.h (vehicle state struct)
├── src/
│   ├── config.h (vehicle constants, pins)
│   ├── main.cpp (skeleton firmware)
│   ├── network/protocol/
│   │   └── V2VMessage.h (BSM-compatible 85 bytes)
│   └── utils/
│       ├── Logger.h/cpp (colored serial logging)
│       └── MACHelper.h (MAC address utilities)
```

**Issues Fixed During Build:**
1. ❌ Legacy Helpers.h had dependencies on PackageManager (not yet ported)
   - **Fix:** Removed for now, will add properly in Phase 2
2. ⚠️ platformio.ini had deprecated `monitor_flags`
   - **Fix:** Removed and documented deprecation

**Key Decisions:**
1. Skip legacy Helpers.h/cpp until Phase 2 (needs PackageManager)
2. Simplified Logger (no file logging, just serial for now)
3. V2VMessage verified to fit in 250-byte ESP-NOW limit
4. All interfaces compile successfully

**Testing:**
- ✅ Code compiles without errors
- ✅ V2VMessage size fits ESP-NOW limit
- ✅ All interfaces recognized by compiler
- ⏸️ ESP32 hardware test pending (need to connect board)

**Next Session Plan:**
- Test firmware on actual ESP32 (LED blink, serial output)
- Start Phase 2: Port ESP-NOW mesh networking
- Implement V2VMessage send/receive
- Test with 2-3 ESP32 units

---

### Session 3: Phase 1 Code Quality Fixes (November 13, 2025)

**Participants:** Hardware team + Claude
**Duration:** ~30 minutes
**Status:** ✅ **PHASE 1 FINAL FIXES COMPLETE**

**Context:**
Code review identified 2 critical issues that needed fixing before Phase 2:
1. V2VMessage size ambiguity (claimed 85 bytes, actually 90 bytes)
2. Missing VEHICLE_ID warning during startup

**Accomplishments:**
- ✅ **Fixed V2VMessage size documentation** (was incorrectly documented as 85 bytes)
  - Calculated actual size: **90 bytes** with `#pragma pack(1)`
  - Added `static_assert(sizeof(V2VMessage) == 90, ...)` for compile-time verification
  - Updated all documentation comments with correct breakdown
  - Fixed header comment (was "8 bytes", now correctly "13 bytes")
- ✅ **Added VEHICLE_ID warning in setup()**
  - Prevents duplicate ID issues during multi-unit testing in Phase 5/6
  - Displays prominent warning on serial console at boot
  - Reminds user to edit config.h before flashing multiple units
- ✅ **Updated main.cpp size check** to verify 90 bytes (not 85)
- ✅ **Verified build still succeeds** - No compilation errors

**V2VMessage Size Breakdown (Verified):**
```
Header:   1 (version) + 8 (vehicleId) + 4 (timestamp) = 13 bytes
Position: 12 bytes (3 floats)
Dynamics: 16 bytes (4 floats)
Sensors:  36 bytes (9 floats)
Alert:    6 bytes (2 uint8_t + 1 float)
Mesh:     7 bytes (1 uint8_t + 6 uint8_t array)
TOTAL:    90 bytes
```

**Build Statistics (After Fixes):**
- **RAM Usage:** 6.6% (21,528 / 327,680 bytes)
- **Flash Usage:** 21.2% (277,661 / 1,310,720 bytes)
- **Status:** ✅ Clean compilation, zero errors, all static_asserts pass

**Files Modified:**
1. `hardware/src/network/protocol/V2VMessage.h`
   - Fixed size from 85→90 bytes
   - Added `static_assert(sizeof(V2VMessage) == 90, ...)`
   - Updated all size comments
   - Fixed header comment (8→13 bytes)
2. `hardware/src/main.cpp`
   - Added VEHICLE_ID warning banner in `setup()`
   - Updated size check (85→90 bytes)
   - Better error messages for size mismatches

**Deferred Issues (Non-Critical):**
- Logger string concatenation optimization (Phase 5)
- MACHelper.h Doxygen documentation (Phase 2)
- Magic number constants cleanup (Phase 2)
- Unit tests (Phase 6)

**Phase 1 Status:** ✅ **OFFICIALLY COMPLETE**
- All critical issues resolved
- Code compiles successfully
- Ready for Phase 2 (Network Layer Migration)

**Next Session Plan:**
- Begin Phase 2: Port ESP-NOW mesh networking
- Implement EspNowTransport (ITransport interface)
- Test V2VMessage send/receive between 2 ESP32 units

---

### Session 4: Phase 2 Session 1 - EspNowTransport Implementation (November 13, 2025)

**Participants:** Hardware team + Claude
**Duration:** ~2 hours
**Status:** ✅ **PHASE 2 SESSION 1 COMPLETE** (Software - Hardware testing pending)

**Context:**
Started Phase 2 (Network Layer Migration) following the incremental approach (Option A).
Goal: Implement basic ESP-NOW transport layer and test with 2 units.

**Accomplishments:**
- ✅ **Created Phase 2 directory structure**
  - `hardware/src/network/transport/` (EspNowTransport)
  - `hardware/src/network/mesh/legacy/` (reference files)
- ✅ **Copied legacy ESP-NOW code for reference**
  - MESH_Sender, MESH_Receiver, PackageManager, PeerManager
  - CleanupManager, Data, MESH_config
  - Files renamed to `.cpp.txt` to prevent compilation
- ✅ **Implemented EspNowTransport (ITransport interface)**
  - Full ESP-NOW initialization (preserves critical sequence)
  - Send/receive callbacks with statistics tracking
  - Singleton pattern for ESP-NOW callbacks
  - 192 lines (header) + 181 lines (implementation)
- ✅ **Created test firmware** (`main_phase2_session1.cpp.bak`)
  - 2-unit communication test
  - Sends V2VMessage every 2 seconds
  - Displays stats every 10 seconds
  - LED heartbeat indicator
- ✅ **Wrote comprehensive test plans**
  - 5 detailed hardware test procedures
  - Expected output, pass/fail criteria
  - Troubleshooting guide
- ✅ **Build verification successful**
  - RAM: 13.3% (43,624 bytes) - increased from 6.6%
  - Flash: 57.9% (758,313 bytes) - increased from 21.4%
  - ESP-NOW + WiFi stack adds ~36% flash (acceptable)

**Files Created:**
1. `hardware/src/network/transport/EspNowTransport.h` (192 lines)
2. `hardware/src/network/transport/EspNowTransport.cpp` (181 lines)
3. `hardware/src/main_phase2_session1.cpp.bak` (158 lines)
4. `hardware/src/network/mesh/legacy/README.md`
5. `docs/PHASE2_HARDWARE_TEST_PLANS.md` (450+ lines)
6. `docs/PHASE2_SESSION1_SUMMARY.md` (complete overview)

**Key Technical Decisions:**
1. **Preserved critical ESP-NOW init sequence**:
   ```cpp
   WiFi.mode(WIFI_STA);
   WiFi.disconnect();  // ← CRITICAL: prevents auto-connect interference
   esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
   esp_now_init();
   esp_now_register_send_cb();
   esp_now_register_recv_cb();
   esp_now_add_peer(broadcast);
   ```
2. **Used singleton pattern** for ESP-NOW callbacks (static C callbacks)
3. **Legacy files as reference only** - `.cpp.txt` prevents compilation
4. **Incremental testing approach** - verify transport before mesh complexity

**Testing Status:**
- ✅ Code compiles successfully
- ✅ Phase 1 firmware still works
- ⏳ **Hardware testing pending** (no ESP32 boards available)

**⚠️ CRITICAL: BEFORE CONTINUING PHASE 2 OR HARDWARE TESTING:**

**MUST READ THESE DOCUMENTS FIRST:**
1. **`docs/PHASE2_SESSION1_SUMMARY.md`** - Complete session overview
   - Quick start guide for hardware testing
   - Build statistics and analysis
   - Key implementation details
   - Next steps roadmap

2. **`docs/PHASE2_HARDWARE_TEST_PLANS.md`** - Comprehensive test procedures
   - Test Session 1: Basic ESP-NOW Communication (2 units)
   - Test Session 2: Three-Unit Relay (future)
   - Test Session 3: Latency Measurement
   - Test Session 4: Load Testing
   - Expected output at each step
   - Pass/fail criteria
   - Troubleshooting guide
   - Test results template

**DO NOT proceed to:**
- ❌ Phase 2 Session 2 (PackageManager port)
- ❌ Phase 3 (Sensor layer)
- ❌ Hardware testing

**WITHOUT first reviewing these documents!**

**Phase 2 Session 1 Status:** ✅ **SOFTWARE COMPLETE**
- All code implemented and compiling
- Documentation complete
- Ready for hardware validation

**Hardware Testing Checklist (When ESP32s available):**
- [ ] Review `docs/PHASE2_HARDWARE_TEST_PLANS.md`
- [ ] Review `docs/PHASE2_SESSION1_SUMMARY.md`
- [ ] Configure Unit 1 (VEHICLE_ID = "V001")
- [ ] Configure Unit 2 (VEHICLE_ID = "V002")
- [ ] Activate Phase 2 test firmware
- [ ] Flash both units
- [ ] Run Test Session 1 (Basic Communication)
- [ ] Verify initialization succeeds
- [ ] Verify send/receive works
- [ ] Check statistics (Send/Receive/Fail counts)
- [ ] Test range (>10m)
- [ ] Test interference (wood/metal/human)
- [ ] Document results in test template

**Next Session Plan (Phase 2 Session 2):**
- Port `PackageManager` from legacy (deduplication)
- Adapt `Data` struct → `V2VMessage`
- Implement sourceMAC+timestamp key
- Test duplicate message blocking
- **Estimated:** 2-3 hours
- **Prerequisite:** Session 1 hardware testing complete

**Legacy Reference Files Cleanup Plan:**
- **Location:** `hardware/src/network/mesh/legacy/*.cpp.txt`
- **Status:** Temporary reference files (renamed from `.cpp` to prevent compilation)
- **Keep until:** Phase 2 Session 4 complete + hardware validation passes
- **Then remove:** `rm -rf hardware/src/network/mesh/legacy/`
- **Or archive:** `mv legacy/ docs/legacy/reference_code/` (optional)
- **Validation checklist in:** `hardware/src/network/mesh/legacy/README.md`

**Blocker:** Awaiting hardware availability for testing

---

### Session 5: Phase 2 Session 1 - Hardware Testing (November 13, 2025)

**Participants:** Amir (Hardware team)
**Duration:** ~30 minutes
**Status:** ✅ **HARDWARE TESTING COMPLETE - ALL TESTS PASSED**

**Context:**
Hardware finally available! Testing Phase 2 Session 1 implementation with real ESP32 units.

**Setup:**
- **Unit 1 (V001):** Legacy board with GPS + MPU6050 (Fedora laptop)
- **Unit 2 (V002):** Bare ESP32 (Windows 11 laptop)
- **Configuration:** Dual laptop setup for side-by-side monitoring

**Accomplishments:**
- ✅ **Flashed both units successfully**
  - V001 on Fedora: `/dev/ttyUSB0`
  - V002 on Windows: COM port (auto-detected)
- ✅ **ESP-NOW initialization successful** on both units
- ✅ **Bidirectional communication working**
  - V001 receiving from V002 ✅
  - V002 receiving from V001 ✅
- ✅ **Message format verified**
  - V2VMessage = 90 bytes (confirmed)
  - Vehicle ID transmitted correctly
  - Timestamps incrementing
  - Hop count = 0 (direct communication)
- ✅ **100% success rate**
  - Send Count: 112+ messages (continuous)
  - Receive Count: 112+ messages (matching)
  - Fail Count: 0
  - Packet Loss: 0%
- ✅ **MAC address verified:** V002 = `30:C9:22:D1:84:2C`

**Test Results:**
```
Unit 1 (V001) - Fedora:
- Messages Sent:     112+
- Messages Received: 112+ (from V002)
- Fail Count:        0
- Success Rate:      100%

Unit 2 (V002) - Windows:
- Messages Sent:     112+
- Messages Received: 112+ (from V001)
- Fail Count:        0
- Success Rate:      100%
```

**Sample Output (V001):**
```
[DEBUG] [ESP-NOW] Received 90 bytes from 30:C9:22:D1:84:2C
[INFO ] [RECV] Received 90 bytes
[INFO ] [RECV] Vehicle: V002
[INFO ] [RECV] Timestamp: 8011
[INFO ] [RECV] Hop count: 0
[DEBUG] [ESP-NOW] Send queued (90 bytes)
[INFO ] [SEND] Message #112 sent (90 bytes)
[DEBUG] [ESP-NOW] ✓ Send confirmed (total: 112)
```

**Issues Resolved:**
1. ✅ **Fedora USB permissions** - Fixed with `chmod 666 /dev/ttyUSB0`
2. ✅ **PlatformIO compilation error** - Renamed backup file to `.cpp.bak`
3. ✅ **Windows IntelliSense** - Rebuilt after first Build

**Key Validations:**
1. ✅ EspNowTransport implementation works flawlessly
2. ✅ V2VMessage struct transmits correctly (90 bytes binary)
3. ✅ Critical ESP-NOW init sequence preserved from legacy - perfect
4. ✅ Legacy board works with new firmware (sensors not yet used)
5. ✅ Bare ESP32 works identically - hardware-agnostic code confirmed
6. ✅ Dual laptop setup superior for testing

**Detailed Documentation:**
- Created: `docs/PHASE2_SESSION1_TEST_RESULTS.md` (comprehensive test report)

**Phase 2 Session 1 Status:** ✅ **COMPLETE - HARDWARE VALIDATED**
- Software implementation: COMPLETE ✅
- Hardware testing: COMPLETE ✅
- All success criteria met ✅

**Next Session Plan (Phase 2 Session 2):**
- **CRITICAL PRE-SESSION:** Fix MACHelper.h broken include (5 min) ⚠️
  - See `docs/CODE_QUALITY_FIXES.md` for details
- Port `PackageManager` from legacy (deduplication)
- Adapt `Data` struct → `V2VMessage`
- Implement sourceMAC+timestamp key
- **Quick fixes during session:** Logger reserve() + magic numbers (7 min)
- Test duplicate message blocking with 2 units
- **Estimated:** 2.5-3 hours (including fixes)
- **No blockers:** Hardware available and tested

**Critical Learnings:**
1. ✅ Hardware testing revealed no issues - implementation is solid
2. ✅ Dual laptop setup is ideal - highly recommended for future sessions
3. ✅ Legacy ESP-NOW code quality confirmed - excellent foundation
4. ✅ Phase 2 architecture validated - ready to build upon

---

## Critical Information for Future Sessions

### V2VMessage Format (DO NOT CHANGE WITHOUT TEAM APPROVAL)

```cpp
#pragma pack(push, 1)
struct V2VMessage {
    uint8_t version;           // 2
    char vehicleId[8];         // "V001", "V002", "V003"
    uint32_t timestamp;        // millis()
    struct { float lat, lon, alt; } position;
    struct { float speed, heading, longAccel, latAccel; } dynamics;
    struct { float accel[3], gyro[3], mag[3]; } sensors;
    struct { uint8_t riskLevel, scenarioType; float confidence; } alert;
    uint8_t hopCount;
    uint8_t sourceMAC[6];
};
#pragma pack(pop)
// Total: 90 bytes (verified with static_assert)
// Breakdown: 13 (header) + 12 (pos) + 16 (dyn) + 36 (sens) + 6 (alert) + 7 (mesh) = 90
```

### Directory Structure Target

```
hardware/src/
├── sensors/
│   ├── imu/        (IImuSensor, MPU9250Driver, Calibration)
│   ├── gps/        (IGpsSensor, NEO6M_Driver)
│   └── fusion/     (SensorFusion, Kalman filter)
├── network/
│   ├── transport/  (ITransport, EspNowTransport)
│   ├── protocol/   (V2VMessage, MessageBuilder, MessageParser)
│   └── mesh/       (MeshManager, PeerManager, PackageManager)
├── ml/             (MLInference, FeatureExtractor, model_tflite.h)
├── core/           (VehicleState, HazardDetector, AlertManager)
└── utils/          (Logger, Storage, Helpers)
```

### Legacy Code Reuse Map

**✅ PORTED & HARDWARE VALIDATED (Phase 2 Session 1):**
- MESH_Sender → `EspNowTransport` (network/transport/)
  - Preserves critical initialization sequence
  - Implements ITransport interface
  - Status: ✅ Complete, ✅ Hardware tested (100% success rate)

**⏳ TO BE PORTED (Phase 2 Sessions 2-4):**
- MESH_Receiver → Will integrate into MeshManager
- PackageManager → network/mesh/ (Session 2)
- PeerManager → network/mesh/ (Session 3)
- CleanupManager → network/mesh/ (Session 3)
- NEO6M_GPS_Service → sensors/gps/NEO6M_Driver
- MESH_Config → config/
- Middleware (logging, delays) → utils/

**Refactor/Adapt:**
- Data struct → V2VMessage (NEW format)
- IAccelerometerSensor → IImuSensor (9-axis)
- main.cpp → Use new interfaces

**Drop/Rewrite:**
- MPU6050_Service → Write new MPU9250Driver

### ESP-NOW Initialization Sequence (CRITICAL - DO NOT CHANGE ORDER)

```cpp
WiFi.mode(WIFI_STA);
WiFi.disconnect();  // ← CRITICAL: removes STA auto-connect interference
esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
esp_now_init();
esp_now_register_send_cb(sentCallback);
esp_now_add_peer(&broadcastPeerInfo);  // FF:FF:FF:FF:FF:FF
```

---

## Phase-by-Phase Progress

### Phase 1: Foundation & Structure (Week 1-2)

**Goal:** Set up directory structure, define interfaces, port utilities

#### Step 1.1: Create Directory Structure (0/8)
- [ ] Create `hardware/src/sensors/imu/`
- [ ] Create `hardware/src/sensors/gps/`
- [ ] Create `hardware/src/network/transport/`
- [ ] Create `hardware/src/network/protocol/`
- [ ] Create `hardware/src/network/mesh/`
- [ ] Create `hardware/src/ml/`
- [ ] Create `hardware/src/core/`
- [ ] Create `hardware/src/utils/`

#### Step 1.2: Define Abstract Interfaces (0/6)
- [ ] Write `IImuSensor.h` interface
- [ ] Write `IGpsSensor.h` interface
- [ ] Write `ITransport.h` interface
- [ ] Write `VehicleState.h` struct
- [ ] Write `V2VMessage.h` struct
- [ ] Add size check: `static_assert(sizeof(V2VMessage) <= 250)`

#### Step 1.3: Copy Legacy Utilities (0/4)
- [ ] Copy `Helpers.h` to `utils/`
- [ ] Copy `MACHelper.h` to `utils/`
- [ ] Copy `SerialConsoleMiddleware` → `Logger.cpp`
- [ ] Verify utilities compile

---

### Phase 2: Network Layer Migration (Week 2-3)

**Goal:** Port ESP-NOW mesh, implement V2VMessage, test networking

**Status:** 🔄 **IN PROGRESS** - Session 1 complete (54% done - 7/13 tasks)

#### Session 1: EspNowTransport (✅ COMPLETE - HARDWARE VALIDATED)
- [x] Copy legacy ESP-NOW files to `legacy/` (reference)
- [x] Implement `EspNowTransport` (ITransport interface)
- [x] Preserve critical ESP-NOW init sequence
- [x] Create test firmware (2-unit communication)
- [x] Write hardware test plans
- [x] Verify code compiles
- [x] **Hardware testing** (✅ COMPLETE - 100% success rate)

#### Session 2: PackageManager (⏳ NOT STARTED)
- [ ] Port `PackageManager` from legacy
- [ ] Adapt `Data` → `V2VMessage`
- [ ] Implement deduplication (sourceMAC + timestamp)
- [ ] Test duplicate message blocking
- [ ] Verify 60-second timeout cleanup

#### Session 3: PeerManager + CleanupManager (⏳ NOT STARTED)
- [ ] Port `PeerManager` from legacy
- [ ] Implement peer tracking (MAC, lastSeen, hops)
- [ ] Port `CleanupManager`
- [ ] Implement periodic garbage collection
- [ ] Test peer inactivity timeout (60s)

#### Session 4: MeshManager Integration (⏳ NOT STARTED)
- [ ] Create `MeshManager` orchestrator
- [ ] Integrate transport + package + peer managers
- [ ] Implement relay logic (hop-count)
- [ ] Test 3-unit relay (V001 ↔ V002 ↔ V003)
- [ ] Measure latency (<50ms single hop)

**Progress:** 4/13 tasks complete (30%)

---

### Phase 3: Sensor Layer Migration (Week 3-4)

**Goal:** Port GPS, implement MPU9250 driver, integrate sensors

*(14 tasks - see migration plan for details)*

---

### Phase 4: ML Integration (Week 5-6)

**Goal:** Integrate TensorFlow Lite, deploy model, test inference

*(9 tasks - see migration plan for details)*

---

### Phase 5: Core Application Logic (Week 6-7)

**Goal:** Implement hazard detection, alerts, orchestrate main loop

*(9 tasks - see migration plan for details)*

---

### Phase 6: Validation & Testing (Week 7-8)

**Goal:** Bench testing, real-world scenarios, performance validation

*(12 tasks - see migration plan for details)*

---

## Known Blockers & Dependencies

### Hardware Blockers
- ⏳ **MPU9250 sensors not yet arrived** (~2 weeks ETA)
  - **Impact:** Cannot test IMU driver, calibration, sensor fusion
  - **Mitigation:** Work on network layer, interfaces, GPS driver in parallel
  - **When unblocked:** Immediately start Phase 3 (sensor layer)

### Software Dependencies
- ✅ PlatformIO environment set up
- ✅ Legacy firmware available for reference
- ✅ Bolder Flight Systems MPU9250 library in platformio.ini
- ⏳ ML model not yet trained (ML team working on it)
  - **Mitigation:** Use stub/placeholder model for testing

---

## Testing Strategy

### Component Testing
Each phase should have standalone tests:
- **Phase 1:** Compile check for interfaces
- **Phase 2:** ESP-NOW message exchange (2 units)
- **Phase 3:** Sensor data serial output (IMU + GPS)
- **Phase 4:** ML inference latency test
- **Phase 5:** End-to-end system test (3 units)

### Integration Testing
- Test with 2 ESP32 units first (simpler)
- Add 3rd unit for mesh relay testing
- Verify hop-count logic works

### Real-World Testing
- Deploy in 3 test vehicles
- Test convoy, intersection, lane-change scenarios
- Record GPS traces + IMU data

---

## Communication Between Sessions

### For Next Session (Tomorrow)
**What to tell Claude:**
"Continue from Hardware Phase 3 implementation. We approved the migration plan in `docs/hw_firmware_migration_plan.md`. Start with Phase 1: create directory structure and define interfaces. Reference V2VMessage format from CLAUDE.md."

**Files to Reference:**
- `docs/hw_firmware_migration_plan.md` - Complete checklist
- `docs/HARDWARE_PROGRESS.md` - This file (progress tracker)
- `CLAUDE.md` - V2VMessage format + architecture overview
- `docs/legacy/legacy_firmware_keep_drop_plan.md` - What to port

### For Component-Specific Sessions
When working on specific components (e.g., MPU9250 driver), start a NEW chat with:
"Implement MPU9250 driver for RoadSense. See `docs/hw_firmware_migration_plan.md` Phase 3, Step 3.2. Reference IImuSensor interface and V2VMessage format from CLAUDE.md."

---

## Success Metrics

**Phase 1 Complete When:**
- [ ] All directories created
- [ ] All interfaces compile
- [ ] Utilities ported and working

**Phase 2 Complete When:**
- [ ] ESP-NOW mesh working (2-3 units)
- [ ] V2VMessage serialization/parsing works
- [ ] Latency <50ms for single hop

**Phase 3 Complete When:**
- [ ] MPU9250 driver reads all 9 axes
- [ ] Magnetometer calibration works
- [ ] GPS driver working with caching

**Phase 4 Complete When:**
- [ ] TFLite model loads
- [ ] Inference <100ms
- [ ] Feature extraction correct

**Phase 5 Complete When:**
- [ ] Hazard detection working
- [ ] Alerts displayed on LED
- [ ] 10Hz main loop stable

**Phase 6 Complete When:**
- [ ] 3 real-world scenarios tested
- [ ] Accuracy >85%
- [ ] False positives <10%
- [ ] Latency <100ms end-to-end

---

**Last Updated:** November 13, 2025 - Session 5 (Hardware Testing Complete)
**Next Session:** Phase 2 Session 2 - PackageManager (message deduplication)
**Code Quality:** See `docs/CODE_QUALITY_FIXES.md` for pending fixes before Session 2
**Estimated Completion:** End of December 2025 (on track - ahead of schedule!)
