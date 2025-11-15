# COMPREHENSIVE AI AGENT IMPLEMENTATION PROMPT
## SAE J2735 BSM Compatibility Implementation for RoadSense V2V System

---

## MISSION OBJECTIVE
Implement a complete, production-ready, SAE J2735-202409 compliant Basic Safety Message (BSM) system for the RoadSense V2V ESP32-based vehicle-to-vehicle hazard detection platform. The implementation MUST be fully standards-compliant, memory-efficient for embedded systems, and seamlessly integrate with the existing project architecture.

---

## PROJECT CONTEXT

### Current State
- **Repository**: roadsense-v2v (newly initialized)
- **Platform**: ESP32 microcontroller (240 MHz dual-core, 520 KB SRAM)
- **Primary Goal**: Vehicle-to-Vehicle hazard detection using mesh networking
- **Current Status**: Empty repository structure with placeholder directories
- **Network Stack**: Will use ESP-NOW or WiFi for V2V communication
- **Programming Language**: C/C++ for ESP32 firmware

### Directory Structure (Existing)
```
roadsense-v2v/
├── hardware/src/drivers/     # ESP32 firmware (TARGET LOCATION)
├── simulation/scenarios/      # Veins/SUMO scenarios
├── bridge/src/               # Hardware-in-the-loop bridge
├── ml/                       # Machine learning pipeline
├── docs/                     # Documentation
├── config/                   # Global configurations
└── tests/                    # Integration tests
```

---

## TECHNICAL REQUIREMENTS SPECIFICATION

### 1. SAE J2735-202409 BSM PART I (MANDATORY FIELDS)

You MUST implement ALL of the following fields with EXACT specifications:

#### 1.1 Message Header
```
Field: msgCnt
- Type: uint8_t
- Range: 0-127 (wraps around)
- Purpose: Sequence number for message tracking
- Increment: +1 for each new BSM transmission
- Implementation: Rolling counter with modulo 128
```

#### 1.2 Temporary Vehicle ID
```
Field: id (temporaryID)
- Type: uint8_t[4] (32-bit array)
- Range: Random 4-byte identifier
- Purpose: Privacy-preserving vehicle identification
- Persistence: Should change every 5 minutes per SAE J2735
- Implementation: Use ESP32 random number generator with seed from MAC address
- Security: MUST NOT be tied to VIN or permanent identifiers
```

#### 1.3 Time Reference
```
Field: secMark (DSecond)
- Type: uint16_t
- Range: 0-60000 milliseconds
- Resolution: 1 millisecond
- Purpose: Time within current minute (UTC)
- Implementation: Extract milliseconds from GPS time or RTC
- Rollover: At 60000ms, resets to 0
- Critical: Must be synchronized across vehicles (GPS time preferred)
```

#### 1.4 Position (3D Coordinates)
```
Field: lat (Latitude)
- Type: int32_t
- Units: 1/10 micro-degrees (0.0000001 degrees)
- Range: -900000000 to 900000001 (unavailable)
- Resolution: ~1.11 cm at equator
- Implementation: GPS latitude * 10000000
- Example: 42.3456789° = 423456789

Field: long (Longitude)
- Type: int32_t
- Units: 1/10 micro-degrees
- Range: -1799999999 to 1800000001 (unavailable)
- Resolution: ~1.11 cm at equator
- Implementation: GPS longitude * 10000000

Field: elev (Elevation)
- Type: int16_t
- Units: 0.1 meters
- Range: -4096 to 61439 (representing -409.6m to 6143.9m)
- Value: -4096 = unavailable
- Implementation: GPS altitude * 10
```

#### 1.5 Positional Accuracy
```
Field: accuracy (PositionalAccuracy)
- Structure: Contains 3 sub-fields

  1. semiMajor (SemiMajorAxisAccuracy)
     - Type: uint8_t
     - Units: 0.05 meters
     - Range: 0-254 (255 = unavailable)
     - Represents: 0 to 12.7 meters

  2. semiMinor (SemiMinorAxisAccuracy)
     - Type: uint8_t
     - Units: 0.05 meters
     - Range: 0-254 (255 = unavailable)

  3. orientation (SemiMajorAxisOrientation)
     - Type: uint16_t
     - Units: 0.0054932479 degrees
     - Range: 0-65535 (represents 0-359.9945078786 degrees)
     - Reference: From True North clockwise

- Source: GPS DOP (Dilution of Precision) values
- Implementation: Parse from GPS NMEA $GPGSA sentence
```

#### 1.6 Transmission State
```
Field: transmission (TransmissionState)
- Type: uint8_t (enum)
- Values:
  0 = neutral
  1 = park
  2 = forwardGears
  3 = reverseGears
  4 = reserved1
  5 = reserved2
  6 = reserved3
  7 = unavailable
- Source: CAN bus OBD-II PID or manual GPIO selection
```

#### 1.7 Speed
```
Field: speed (Velocity)
- Type: uint16_t
- Units: 0.02 m/s
- Range: 0-8190 (8191 = unavailable)
- Represents: 0 to 163.8 m/s (0 to 589.68 km/h)
- Implementation: GPS speed or wheel sensor
- Formula: speed_value = (speed_m_s / 0.02)
- Example: 100 km/h = 27.78 m/s = 1389 units
```

#### 1.8 Heading
```
Field: heading (Heading)
- Type: uint16_t
- Units: 0.0125 degrees
- Range: 0-28799 (28800 = unavailable)
- Represents: 0 to 359.9875 degrees
- Reference: True North = 0, clockwise
- Implementation: GPS course over ground
- Formula: heading_value = (degrees / 0.0125)
```

#### 1.9 Steering Wheel Angle
```
Field: angle (SteeringWheelAngle)
- Type: int8_t
- Units: 1.5 degrees
- Range: -126 to +126 (127 = unavailable)
- Represents: -189° to +189°
- Reference: 0 = straight, positive = left, negative = right
- Source: Steering angle sensor or IMU-derived
```

#### 1.10 Acceleration Set (4-Way)
```
Field: accelSet (AccelerationSet4Way)
- Structure: 4 acceleration values

  1. long (LongitudinalAcceleration)
     - Type: int16_t
     - Units: 0.01 m/s²
     - Range: -2000 to +2001 (2001 = unavailable)
     - Represents: -20 to +20 m/s²

  2. lat (LateralAcceleration)
     - Type: int16_t
     - Units: 0.01 m/s²
     - Range: -2000 to +2001

  3. vert (VerticalAcceleration)
     - Type: int8_t
     - Units: 0.02 G
     - Range: -127 to +127 (-127 = unavailable)

  4. yaw (YawRate)
     - Type: int16_t
     - Units: 0.01 degrees/second
     - Range: -32767 to +32767 (0 = straight)

- Source: IMU (MPU6050, MPU9250, or similar)
- Critical: Must calibrate IMU for vehicle orientation
```

#### 1.11 Brake System Status
```
Field: brakes (BrakeSystemStatus)
- Structure: Bitfield structure

  1. wheelBrakes (BrakeAppliedStatus)
     - Type: uint8_t (5 bits used)
     - Bit 4: unavailable
     - Bit 3: leftFront
     - Bit 2: leftRear
     - Bit 1: rightFront
     - Bit 0: rightRear

  2. traction (TractionControlStatus)
     - Type: uint8_t (2 bits)
     - Values: 0=unavailable, 1=off, 2=on, 3=engaged

  3. abs (AntiLockBrakeStatus)
     - Type: uint8_t (2 bits)
     - Values: 0=unavailable, 1=off, 2=on, 3=engaged

  4. scs (StabilityControlStatus)
     - Type: uint8_t (2 bits)
     - Values: 0=unavailable, 1=off, 2=on, 3=engaged

  5. brakeBoost (BrakeBoostApplied)
     - Type: uint8_t (2 bits)
     - Values: 0=unavailable, 1=off, 2=on

  6. auxBrakes (AuxiliaryBrakeStatus)
     - Type: uint8_t (2 bits)
     - Values: 0=unavailable, 1=off, 2=on, 3=reserved

- Total Size: 2 bytes
- Source: CAN bus or GPIO brake light sensor (simplified)
```

#### 1.12 Vehicle Size
```
Field: size (VehicleSize)
- Structure: 2 dimensions

  1. width (VehicleWidth)
     - Type: uint8_t
     - Units: 1 cm
     - Range: 0-1023 (1023 = unavailable)
     - Represents: 0 to 10.23 meters

  2. length (VehicleLength)
     - Type: uint16_t
     - Units: 1 cm
     - Range: 0-16383 (16383 = unavailable)
     - Represents: 0 to 163.83 meters

- Implementation: Static configuration values per vehicle type
```

### 2. BSM PART II (OPTIONAL - IMPLEMENT AS FUTURE EXTENSION)

Create placeholder structures for:
- Path History (GPS breadcrumb trail)
- Path Prediction (trajectory forecasting)
- Vehicle Safety Extensions
- Special Vehicle Extensions (emergency vehicles)

These should be DEFINED but DISABLED by default to conserve memory.

---

## IMPLEMENTATION ARCHITECTURE

### 3. FILE STRUCTURE TO CREATE

```
hardware/src/
├── bsm/
│   ├── bsm_core.h                    # Main BSM structure definitions
│   ├── bsm_core.c                    # BSM encoding/decoding logic
│   ├── bsm_encoder.h                 # Message serialization
│   ├── bsm_encoder.c                 # Binary packing functions
│   ├── bsm_decoder.h                 # Message deserialization
│   ├── bsm_decoder.c                 # Binary unpacking functions
│   ├── bsm_validator.h               # Compliance validation
│   ├── bsm_validator.c               # Range checking, sanity tests
│   └── bsm_config.h                  # Compile-time configuration
├── sensors/
│   ├── gps_interface.h               # GPS data acquisition
│   ├── gps_interface.c               # NMEA parser for NEO-6M/NEO-M8N
│   ├── imu_interface.h               # IMU data acquisition
│   ├── imu_interface.c               # MPU6050/MPU9250 I2C driver
│   ├── vehicle_interface.h           # Vehicle CAN/OBD-II interface
│   └── vehicle_interface.c           # OBD-II PID reader (optional)
├── network/
│   ├── v2v_network.h                 # V2V communication layer
│   ├── v2v_network.c                 # ESP-NOW or WiFi implementation
│   ├── v2v_protocol.h                # Message framing protocol
│   └── v2v_protocol.c                # Packet assembly/disassembly
└── main/
    ├── main.cpp                      # ESP32 main application
    ├── bsm_app.h                     # Application-level BSM logic
    └── bsm_app.c                     # BSM transmission scheduling

config/
└── bsm_compliance_spec.json          # SAE J2735 field specifications

docs/
├── BSM_IMPLEMENTATION.md             # Implementation documentation
├── BSM_COMPLIANCE_CHECKLIST.md       # Verification checklist
└── BSM_TESTING_GUIDE.md              # Testing procedures

tests/
├── test_bsm_encoding.c               # Unit tests for encoder
├── test_bsm_decoding.c               # Unit tests for decoder
├── test_bsm_validation.c             # Compliance tests
└── test_integration.c                # End-to-end tests
```

### 4. DATA STRUCTURE DESIGN

#### 4.1 Main BSM Structure (C/C++)

```c
// hardware/src/bsm/bsm_core.h

#ifndef BSM_CORE_H
#define BSM_CORE_H

#include <stdint.h>
#include <stdbool.h>

// SAE J2735 Type Definitions
typedef uint8_t MsgCount_t;           // 0-127
typedef uint8_t TemporaryID_t[4];     // 4 bytes
typedef uint16_t DSecond_t;           // 0-60000 ms
typedef int32_t Latitude_t;           // 1/10 micro-degree
typedef int32_t Longitude_t;          // 1/10 micro-degree
typedef int16_t Elevation_t;          // 0.1 meter
typedef uint8_t SemiAxisAccuracy_t;   // 0.05 meter
typedef uint16_t Orientation_t;       // 0.0054932479 degree
typedef uint8_t TransmissionState_t;  // enum
typedef uint16_t Velocity_t;          // 0.02 m/s
typedef uint16_t Heading_t;           // 0.0125 degree
typedef int8_t SteeringAngle_t;       // 1.5 degrees
typedef int16_t Acceleration_t;       // 0.01 m/s²
typedef int8_t VerticalAccel_t;       // 0.02 G
typedef int16_t YawRate_t;            // 0.01 deg/s
typedef uint8_t VehicleWidth_t;       // 1 cm
typedef uint16_t VehicleLength_t;     // 1 cm

// Positional Accuracy
typedef struct {
    SemiAxisAccuracy_t semiMajor;     // 0-254 (255=unavailable)
    SemiAxisAccuracy_t semiMinor;     // 0-254 (255=unavailable)
    Orientation_t orientation;         // 0-65535
} __attribute__((packed)) PositionalAccuracy_t;

// Brake System Status
typedef struct {
    uint8_t wheelBrakes      : 5;     // Bit field: 4=unavail, 3=LF, 2=LR, 1=RF, 0=RR
    uint8_t traction         : 2;     // 0=unavail, 1=off, 2=on, 3=engaged
    uint8_t abs              : 2;     // 0=unavail, 1=off, 2=on, 3=engaged
    uint8_t scs              : 2;     // Stability control
    uint8_t brakeBoost       : 2;     // Brake boost
    uint8_t auxBrakes        : 2;     // Auxiliary brakes
    uint8_t _padding         : 3;     // Align to byte boundary
} __attribute__((packed)) BrakeSystemStatus_t;

// 4-Way Acceleration Set
typedef struct {
    Acceleration_t longitudinal;       // -2000 to +2001 (0.01 m/s²)
    Acceleration_t lateral;            // -2000 to +2001 (0.01 m/s²)
    VerticalAccel_t vertical;          // -127 to +127 (0.02 G)
    YawRate_t yaw;                     // -32767 to +32767 (0.01 deg/s)
} __attribute__((packed)) AccelerationSet4Way_t;

// Vehicle Size
typedef struct {
    VehicleWidth_t width;              // 0-1023 cm
    VehicleLength_t length;            // 0-16383 cm
} __attribute__((packed)) VehicleSize_t;

// BSM Part I - Core Data Frame (ALL MANDATORY)
typedef struct {
    MsgCount_t msgCnt;                 // Message count (0-127)
    TemporaryID_t id;                  // 4-byte temporary ID
    DSecond_t secMark;                 // Millisecond of minute
    Latitude_t lat;                    // Latitude
    Longitude_t lon;                   // Longitude
    Elevation_t elev;                  // Elevation
    PositionalAccuracy_t accuracy;     // Positional accuracy
    TransmissionState_t transmission;  // Transmission state
    Velocity_t speed;                  // Speed
    Heading_t heading;                 // Heading
    SteeringAngle_t angle;             // Steering wheel angle
    AccelerationSet4Way_t accelSet;    // 4-way acceleration
    BrakeSystemStatus_t brakes;        // Brake status
    VehicleSize_t size;                // Vehicle dimensions
} __attribute__((packed)) BSMcoreData_t;

// Full BSM Message
typedef struct {
    BSMcoreData_t coreData;            // Part I (mandatory)
    // Part II would go here (optional, future)
    uint32_t timestamp_ms;             // Local timestamp (not transmitted)
    uint8_t checksum;                  // Message integrity check
} __attribute__((packed)) BSM_t;

// Constants
#define BSM_MSG_ID 0x14                // SAE J2735 BSM message ID = 20
#define BSM_PART1_SIZE sizeof(BSMcoreData_t)
#define BSM_TRANSMISSION_RATE_HZ 10    // 10 Hz per SAE J2735
#define BSM_UNAVAILABLE_LATLON 900000001
#define BSM_UNAVAILABLE_ELEV -4096
#define BSM_UNAVAILABLE_SPEED 8191
#define BSM_UNAVAILABLE_HEADING 28800

// Function Prototypes
void bsm_init(void);
void bsm_create_message(BSM_t* bsm);
bool bsm_validate_message(const BSM_t* bsm);
void bsm_update_msg_count(void);
void bsm_generate_temporary_id(void);

#endif // BSM_CORE_H
```

#### 4.2 Sensor Data Acquisition

```c
// hardware/src/sensors/gps_interface.h

#ifndef GPS_INTERFACE_H
#define GPS_INTERFACE_H

#include "../bsm/bsm_core.h"

typedef struct {
    double latitude;         // Decimal degrees
    double longitude;        // Decimal degrees
    float altitude;          // Meters above MSL
    float speed;             // m/s
    float course;            // Degrees from True North
    float hdop;              // Horizontal dilution of precision
    uint8_t satellites;      // Number of satellites
    uint32_t time_ms;        // Milliseconds within minute
    bool valid;              // GPS fix valid
} GPSData_t;

void gps_init(uint8_t rx_pin, uint8_t tx_pin);
bool gps_read_data(GPSData_t* gps_data);
void gps_to_bsm_position(const GPSData_t* gps, BSMcoreData_t* bsm);

#endif // GPS_INTERFACE_H
```

```c
// hardware/src/sensors/imu_interface.h

#ifndef IMU_INTERFACE_H
#define IMU_INTERFACE_H

#include "../bsm/bsm_core.h"

typedef struct {
    float accel_x;           // m/s² (vehicle forward axis)
    float accel_y;           // m/s² (vehicle lateral axis)
    float accel_z;           // m/s² (vehicle vertical axis)
    float gyro_z;            // deg/s (yaw rate)
    uint32_t timestamp_ms;
    bool valid;
} IMUData_t;

void imu_init(uint8_t sda_pin, uint8_t scl_pin);
bool imu_read_data(IMUData_t* imu_data);
void imu_calibrate(void);
void imu_to_bsm_accel(const IMUData_t* imu, AccelerationSet4Way_t* accelSet);

#endif // IMU_INTERFACE_H
```

### 5. MESSAGE ENCODING/DECODING

#### 5.1 Binary Encoding (UPER-like or Custom Compact Format)

```c
// hardware/src/bsm/bsm_encoder.h

#ifndef BSM_ENCODER_H
#define BSM_ENCODER_H

#include "bsm_core.h"

// Encode BSM to binary buffer
// Returns: Number of bytes written
size_t bsm_encode(const BSM_t* bsm, uint8_t* buffer, size_t buffer_size);

// Calculate checksum
uint8_t bsm_calculate_checksum(const uint8_t* data, size_t length);

#endif // BSM_ENCODER_H
```

Implementation requirements:
- Use little-endian byte order (ESP32 native)
- Pack all fields tightly (no padding except explicit alignment)
- Add CRC-8 or simple XOR checksum
- Target size: 38-42 bytes for Part I only
- Optimize for ESP32 performance (avoid bit shifting when possible)

#### 5.2 Binary Decoding

```c
// hardware/src/bsm/bsm_decoder.h

#ifndef BSM_DECODER_H
#define BSM_DECODER_H

#include "bsm_core.h"

// Decode binary buffer to BSM structure
// Returns: true if successful, false if invalid
bool bsm_decode(const uint8_t* buffer, size_t length, BSM_t* bsm);

// Verify checksum
bool bsm_verify_checksum(const uint8_t* data, size_t length, uint8_t checksum);

#endif // BSM_DECODER_H
```

### 6. VALIDATION AND COMPLIANCE

#### 6.1 Field Validation

```c
// hardware/src/bsm/bsm_validator.h

#ifndef BSM_VALIDATOR_H
#define BSM_VALIDATOR_H

#include "bsm_core.h"

typedef struct {
    bool msgCnt_valid;
    bool id_valid;
    bool secMark_valid;
    bool position_valid;
    bool accuracy_valid;
    bool transmission_valid;
    bool speed_valid;
    bool heading_valid;
    bool angle_valid;
    bool accel_valid;
    bool brakes_valid;
    bool size_valid;
    uint8_t error_count;
    char error_messages[10][64];
} BSMValidationReport_t;

// Validate entire BSM message
bool bsm_validate(const BSM_t* bsm, BSMValidationReport_t* report);

// Individual field validators
bool validate_latitude(Latitude_t lat);
bool validate_longitude(Longitude_t lon);
bool validate_elevation(Elevation_t elev);
bool validate_speed(Velocity_t speed);
bool validate_heading(Heading_t heading);
bool validate_sec_mark(DSecond_t secMark);

#endif // BSM_VALIDATOR_H
```

Implementation requirements for each validator:
- Check range bounds per SAE J2735 specification
- Verify "unavailable" values are used correctly
- Log validation failures with descriptive messages
- Support strict mode (reject any invalid field) and permissive mode (accept with warnings)

### 7. NETWORK INTEGRATION

#### 7.1 V2V Communication Layer

```c
// hardware/src/network/v2v_network.h

#ifndef V2V_NETWORK_H
#define V2V_NETWORK_H

#include <stdint.h>
#include <stdbool.h>

// V2V Network Configuration
#define V2V_CHANNEL 1              // WiFi channel for ESP-NOW
#define V2V_MAX_PEERS 20           // Maximum neighbor vehicles
#define V2V_BROADCAST_ADDR {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}

typedef void (*v2v_receive_callback_t)(const uint8_t* mac, const uint8_t* data, size_t len);

// Initialize V2V network (ESP-NOW or WiFi)
bool v2v_init(void);

// Send BSM broadcast
bool v2v_send_broadcast(const uint8_t* data, size_t length);

// Register receive callback
void v2v_register_callback(v2v_receive_callback_t callback);

// Get transmission statistics
typedef struct {
    uint32_t tx_count;
    uint32_t rx_count;
    uint32_t tx_failures;
    uint32_t rx_errors;
} V2VStats_t;

void v2v_get_stats(V2VStats_t* stats);

#endif // V2V_NETWORK_H
```

### 8. MAIN APPLICATION LOGIC

#### 8.1 BSM Transmission Scheduler

```c
// hardware/src/main/bsm_app.h

#ifndef BSM_APP_H
#define BSM_APP_H

#include "../bsm/bsm_core.h"

// BSM Application Configuration
#define BSM_TX_INTERVAL_MS (1000 / BSM_TRANSMISSION_RATE_HZ)  // 100ms for 10Hz
#define BSM_ID_ROTATION_INTERVAL_MS (5 * 60 * 1000)           // 5 minutes

// Initialize BSM application
void bsm_app_init(void);

// Main BSM task (call from FreeRTOS task or main loop)
void bsm_app_task(void* pvParameters);

// Receive handler for incoming BSMs
void bsm_app_receive_handler(const uint8_t* mac, const BSM_t* bsm);

// Get current vehicle BSM
void bsm_app_get_current_bsm(BSM_t* bsm);

#endif // BSM_APP_H
```

#### 8.2 Main ESP32 Application

```cpp
// hardware/src/main/main.cpp

#include <Arduino.h>
#include "bsm_app.h"
#include "../sensors/gps_interface.h"
#include "../sensors/imu_interface.h"
#include "../network/v2v_network.h"

// Pin Definitions
#define GPS_RX_PIN 16
#define GPS_TX_PIN 17
#define IMU_SDA_PIN 21
#define IMU_SCL_PIN 22
#define STATUS_LED_PIN 2

void setup() {
    Serial.begin(115200);
    Serial.println("RoadSense V2V BSM System Starting...");

    pinMode(STATUS_LED_PIN, OUTPUT);

    // Initialize GPS
    gps_init(GPS_RX_PIN, GPS_TX_PIN);

    // Initialize IMU
    imu_init(IMU_SDA_PIN, IMU_SCL_PIN);
    imu_calibrate();

    // Initialize V2V network
    if (!v2v_init()) {
        Serial.println("ERROR: V2V network initialization failed!");
        while(1) {
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
            delay(100);
        }
    }

    // Initialize BSM application
    bsm_app_init();

    Serial.println("System initialized. Broadcasting BSM...");
}

void loop() {
    // BSM task runs continuously
    bsm_app_task(NULL);
    delay(10);  // Small delay for watchdog
}
```

---

## IMPLEMENTATION REQUIREMENTS

### 9. CRITICAL IMPLEMENTATION RULES

#### 9.1 Memory Management
- **STACK ALLOCATION ONLY** for BSM structures (no dynamic malloc/free)
- Total RAM usage for BSM system: < 8 KB
- Use static buffers for encoding/decoding
- Implement circular buffer for received BSM history (last 50 messages)

#### 9.2 Timing Requirements
- BSM transmission rate: 10 Hz (every 100ms) per SAE J2735
- GPS update rate: minimum 5 Hz (prefer 10 Hz)
- IMU update rate: minimum 100 Hz (internal averaging to 10 Hz)
- Maximum latency from sensor read to BSM transmission: < 50ms

#### 9.3 Error Handling
- GPS signal loss: Mark position as "unavailable", continue transmitting with last known position + warning flag
- IMU failure: Mark acceleration/yaw as "unavailable"
- Network failure: Log errors, attempt recovery, maintain message counter
- Invalid sensor data: Reject outliers, use "unavailable" values

#### 9.4 Security Requirements
- Temporary ID rotation: Every 5 minutes (300 seconds)
- Random number generation: Use ESP32 hardware RNG seeded with MAC + timestamp
- NO personally identifiable information in messages
- NO permanent vehicle identifiers (VIN, license plate, etc.)

#### 9.5 Testing Requirements
- Unit tests for EVERY encoding/decoding function
- Integration test simulating 10 Hz transmission for 1 hour
- Compliance test suite verifying SAE J2735 field ranges
- Network stress test with 20 simultaneous vehicles
- Power consumption test (target: < 500mA average)

---

## STEP-BY-STEP IMPLEMENTATION SEQUENCE

### Phase 1: Core Data Structures (Day 1)
1. Create `hardware/src/bsm/bsm_core.h` with ALL struct definitions
2. Create `hardware/src/bsm/bsm_config.h` with compile-time constants
3. Implement `hardware/src/bsm/bsm_core.c` with initialization functions
4. Write unit tests for structure sizes and alignment

### Phase 2: Sensor Integration (Day 2)
1. Implement `hardware/src/sensors/gps_interface.c`
   - NMEA parser for $GPGGA and $GPRMC sentences
   - Extract lat/lon/alt/speed/course/time
   - Convert to SAE J2735 units
2. Implement `hardware/src/sensors/imu_interface.c`
   - I2C driver for MPU6050/MPU9250
   - Calibration routine (zero offset calculation)
   - Convert accelerometer/gyroscope to SAE units
3. Test sensor data acquisition independently

### Phase 3: Message Encoding/Decoding (Day 3)
1. Implement `hardware/src/bsm/bsm_encoder.c`
   - Pack BSMcoreData_t into byte array
   - Calculate and append checksum
2. Implement `hardware/src/bsm/bsm_decoder.c`
   - Unpack byte array into BSMcoreData_t
   - Verify checksum
3. Write round-trip tests (encode → decode → verify equality)

### Phase 4: Validation (Day 4)
1. Implement `hardware/src/bsm/bsm_validator.c`
   - Range validation for all 14 mandatory fields
   - Generate detailed error reports
2. Create compliance test suite
3. Verify against SAE J2735 specification tables

### Phase 5: Network Layer (Day 5)
1. Implement `hardware/src/network/v2v_network.c`
   - ESP-NOW initialization
   - Broadcast transmission
   - Receive callback registration
2. Test network transmission between 2 ESP32 devices
3. Measure packet loss and latency

### Phase 6: Application Integration (Day 6)
1. Implement `hardware/src/main/bsm_app.c`
   - 10 Hz transmission scheduler
   - Sensor data aggregation
   - Message counter management
   - Temporary ID rotation
2. Implement receive handler
3. Create main.cpp integration

### Phase 7: Testing and Validation (Day 7)
1. Run compliance test suite
2. Perform integration testing
3. Measure timing and resource usage
4. Document test results

### Phase 8: Documentation (Day 8)
1. Write `docs/BSM_IMPLEMENTATION.md`
2. Write `docs/BSM_COMPLIANCE_CHECKLIST.md`
3. Create API reference documentation
4. Add inline code comments

---

## VALIDATION CHECKLIST

After implementation, verify the following:

### Functional Requirements
- [ ] All 14 mandatory BSM Part I fields are implemented
- [ ] Message size is ≤ 50 bytes (Part I only)
- [ ] Messages transmit at 10 Hz ± 5ms
- [ ] GPS data updates at ≥ 5 Hz
- [ ] IMU data updates at ≥ 100 Hz
- [ ] Temporary ID rotates every 5 minutes
- [ ] Message counter increments correctly (0-127)
- [ ] "Unavailable" values used correctly when sensors fail

### SAE J2735 Compliance
- [ ] Latitude range: -900000000 to 900000001
- [ ] Longitude range: -1799999999 to 1800000001
- [ ] Elevation range: -4096 to 61439
- [ ] Speed range: 0 to 8191 (0.02 m/s units)
- [ ] Heading range: 0 to 28799 (0.0125° units)
- [ ] SecMark range: 0 to 60000 milliseconds
- [ ] Acceleration range: -2000 to 2001 (0.01 m/s² units)
- [ ] Brake status bitfield correct
- [ ] Vehicle size in centimeters

### Network Performance
- [ ] ESP-NOW successfully broadcasts to 255.255.255.255
- [ ] Packet loss < 1% at 10m distance
- [ ] Latency < 10ms for transmission
- [ ] Successfully receives BSMs from other vehicles
- [ ] Handles 20+ simultaneous transmitters

### Resource Usage
- [ ] RAM usage < 8 KB
- [ ] CPU usage < 30% (on single core)
- [ ] Power consumption < 500mA average
- [ ] No memory leaks after 24-hour test

### Code Quality
- [ ] All functions have Doxygen comments
- [ ] No compiler warnings (-Wall -Wextra)
- [ ] Code passes static analysis (cppcheck)
- [ ] Unit test coverage > 80%
- [ ] Integration tests pass

---

## DEBUGGING AND TROUBLESHOOTING

### Common Issues and Solutions

**Issue: GPS not getting fix**
- Solution: Ensure clear sky view, wait 2-5 minutes for cold start, verify UART baud rate (9600 or 115200)

**Issue: IMU readings drifting**
- Solution: Run calibration with vehicle stationary, implement complementary filter, check I2C pullup resistors

**Issue: ESP-NOW transmission fails**
- Solution: Verify WiFi mode is STA, check channel matching, ensure peer is added correctly

**Issue: Message counter not incrementing**
- Solution: Check modulo 128 logic, verify transmission timing

**Issue: High CPU usage**
- Solution: Optimize float-to-int conversions, use lookup tables for unit conversions, reduce Serial.print statements

---

## CONFIGURATION FILE SPECIFICATION

### BSM Configuration JSON
```json
{
  "bsm_config": {
    "version": "SAE_J2735_202409",
    "transmission_rate_hz": 10,
    "enable_part_ii": false,
    "vehicle_config": {
      "width_cm": 180,
      "length_cm": 450,
      "transmission_type": "forwardGears"
    },
    "sensor_config": {
      "gps": {
        "uart_num": 1,
        "baud_rate": 9600,
        "rx_pin": 16,
        "tx_pin": 17
      },
      "imu": {
        "type": "MPU6050",
        "i2c_addr": "0x68",
        "sda_pin": 21,
        "scl_pin": 22,
        "sample_rate_hz": 100
      }
    },
    "network_config": {
      "protocol": "ESP_NOW",
      "channel": 1,
      "max_peers": 20,
      "enable_encryption": false
    },
    "privacy_config": {
      "id_rotation_interval_sec": 300,
      "enable_random_id": true
    }
  }
}
```

---

## EXPECTED OUTPUT

### Serial Debug Output Example
```
[00:00:00.123] RoadSense V2V BSM System Starting...
[00:00:00.456] GPS: Initializing on UART1 (RX:16, TX:17)
[00:00:02.789] GPS: Waiting for fix... (Satellites: 3)
[00:00:08.234] GPS: Fix acquired (Satellites: 8, HDOP: 1.2)
[00:00:08.567] IMU: Initializing MPU6050 on I2C (SDA:21, SCL:22)
[00:00:09.123] IMU: Calibrating... (Keep vehicle stationary)
[00:00:11.456] IMU: Calibration complete (Offsets: X:0.02, Y:-0.01, Z:9.81)
[00:00:11.789] V2V: Initializing ESP-NOW on channel 1
[00:00:12.123] V2V: Network ready (MAC: AA:BB:CC:DD:EE:FF)
[00:00:12.456] BSM: Application initialized
[00:00:12.789] BSM: Generated temporary ID: 0x1A2B3C4D
[00:00:12.890] BSM: TX #000 | Lat:42.3456789 Lon:-71.0987654 Spd:15.5m/s Hdg:087°
[00:00:13.000] BSM: TX #001 | Lat:42.3456812 Lon:-71.0987632 Spd:15.6m/s Hdg:087°
[00:00:13.100] BSM: RX from AA:BB:CC:DD:EE:12 | Lat:42.3457123 Lon:-71.0986543 Spd:18.2m/s
```

---

## SUCCESS CRITERIA

Implementation is considered COMPLETE and SUCCESSFUL when:

1. ✅ All 14 mandatory BSM Part I fields are implemented correctly
2. ✅ Messages encode/decode without errors
3. ✅ All fields pass SAE J2735 range validation
4. ✅ Messages transmit at stable 10 Hz
5. ✅ ESP32 can receive BSMs from other vehicles
6. ✅ System runs for 24 hours without crashes
7. ✅ Memory usage stays < 8 KB
8. ✅ Unit tests have > 80% coverage
9. ✅ Integration tests pass
10. ✅ Documentation is complete

---

## ADDITIONAL RESOURCES AND REFERENCES

### SAE J2735 Key Sections
- Section 5.2: Basic Safety Message (BSM)
- Section 6.3: Data Frames (DF_BSMcoreData)
- Section 7: Data Elements (all DE_ definitions)
- Appendix A: ASN.1 Definitions

### ESP32 Libraries to Use
- `WiFi.h` - ESP-NOW network stack
- `Wire.h` - I2C for IMU
- `HardwareSerial.h` - UART for GPS
- `esp_system.h` - Hardware RNG

### GPS NMEA Sentences to Parse
- `$GPGGA` - Position, altitude, time, satellites
- `$GPRMC` - Position, speed, course, date
- `$GPGSA` - DOP (accuracy metrics)

### IMU Sensor Recommendations
- MPU6050: 6-axis (accel + gyro) - Budget option
- MPU9250: 9-axis (accel + gyro + mag) - Better heading
- ICM-20948: Latest 9-axis - Best performance

---

## POST-IMPLEMENTATION TASKS

After core BSM is working:

1. **Hazard Detection Integration**
   - Use received BSMs to calculate relative positions
   - Implement collision warning algorithms
   - Add machine learning hazard classification

2. **Performance Optimization**
   - Profile CPU usage per function
   - Optimize float-to-int conversions
   - Implement SIMD if needed

3. **Extended Features**
   - Add BSM Part II (path history)
   - Implement SPAT (Signal Phase and Timing) messages
   - Add MAP (intersection geometry) support

4. **Dashboard/UI**
   - Create web interface for monitoring
   - Real-time BSM visualization
   - Statistics and diagnostics

---

## FINAL NOTES FOR AI AGENT

**CRITICAL INSTRUCTIONS:**

1. **FOLLOW THE SPECIFICATION EXACTLY** - Do not deviate from SAE J2735 field definitions
2. **VERIFY EVERY FIELD** - Double-check ranges, units, and data types
3. **COMMENT EXTENSIVELY** - Every function needs purpose, parameters, returns documented
4. **TEST THOROUGHLY** - Write tests BEFORE implementing functions (TDD approach)
5. **OPTIMIZE FOR EMBEDDED** - Avoid dynamic allocation, minimize floating point
6. **HANDLE ERRORS GRACEFULLY** - Never crash, always use "unavailable" values when needed
7. **MAINTAIN STATE CAREFULLY** - Message counter and temporary ID must persist correctly
8. **RESPECT TIMING** - 10 Hz means EXACTLY 100ms intervals, use hardware timers
9. **VALIDATE CONTINUOUSLY** - Check sensor data before using it
10. **DOCUMENT EVERYTHING** - Code, tests, configuration, usage

**IMPLEMENTATION ORDER:**
Core structures → Sensors → Encoding → Validation → Network → Application → Testing → Documentation

**DO NOT:**
- Use malloc/free or new/delete
- Ignore error returns from functions
- Skip input validation
- Hardcode magic numbers (use #define constants)
- Leave TODO comments without issue tracking
- Commit code that doesn't compile
- Deploy without testing

**SUCCESS METRIC:**
If you can demonstrate two ESP32 devices successfully exchanging SAE J2735-compliant BSM messages at 10 Hz with <1% packet loss and all fields passing validation, the implementation is successful.

---

END OF SPECIFICATION
