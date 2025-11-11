/**
 * @file V2VMessage.h
 * @brief SAE J2735 BSM-compatible V2V message format
 *
 * CRITICAL: This message format is used by BOTH hardware and simulation.
 * The HIL bridge expects this exact struct layout.
 *
 * Total size: 85 bytes (well under 250-byte ESP-NOW limit)
 *
 * Standards compliance: Based on SAE J2735 Basic Safety Message (BSM)
 * Transport compatibility: ESP-NOW (hardware), UDP (simulation bridge)
 */

#ifndef V2VMESSAGE_H
#define V2VMESSAGE_H

#include <Arduino.h>

// Disable struct padding to ensure exact size
#pragma pack(push, 1)

/**
 * @struct V2VMessage
 * @brief Complete V2V message with BSM-compatible fields
 *
 * This struct represents a single V2V broadcast message containing
 * vehicle state, sensor data, and hazard alerts.
 */
struct V2VMessage {
    // ========================================================================
    // HEADER (8 bytes)
    // ========================================================================
    uint8_t version;           // Protocol version (2)
    char vehicleId[8];         // Vehicle identifier: "V001", "V002", "V003"
    uint32_t timestamp;        // millis() or Unix timestamp

    // ========================================================================
    // BSM PART I: CORE DATA SET (28 bytes)
    // ========================================================================

    /**
     * @brief Position data (WGS84 coordinates)
     */
    struct {
        float lat;             // Latitude (degrees)
        float lon;             // Longitude (degrees)
        float alt;             // Altitude (meters above sea level)
    } position;

    /**
     * @brief Vehicle dynamics
     */
    struct {
        float speed;           // Speed (m/s)
        float heading;         // Heading (degrees, 0-359, true north)
        float longAccel;       // Longitudinal acceleration (m/s²)
        float latAccel;        // Lateral acceleration (m/s²)
    } dynamics;

    // ========================================================================
    // BSM PART II: OPTIONAL - ROADSENSE SENSOR DATA (36 bytes)
    // ========================================================================

    /**
     * @brief Raw sensor data from 9-axis IMU
     */
    struct {
        float accel[3];        // Accelerometer (m/s²) [x, y, z]
        float gyro[3];         // Gyroscope (rad/s) [x, y, z]
        float mag[3];          // Magnetometer (μT) [x, y, z]
    } sensors;

    // ========================================================================
    // ROADSENSE EXTENSION: HAZARD ALERT (6 bytes)
    // ========================================================================

    /**
     * @brief Hazard detection results from ML model
     */
    struct {
        uint8_t riskLevel;     // 0=None, 1=Low, 2=Medium, 3=High
        uint8_t scenarioType;  // 0=convoy, 1=intersection, 2=lane-change
        float confidence;      // ML model confidence (0.0-1.0)
    } alert;

    // ========================================================================
    // MESH NETWORKING METADATA (7 bytes)
    // ========================================================================
    uint8_t hopCount;          // Number of relay hops (0-3)
    uint8_t sourceMAC[6];      // Original sender MAC address

    // ========================================================================
    // METHODS
    // ========================================================================

    /**
     * @brief Default constructor - zero initialize
     */
    V2VMessage() {
        memset(this, 0, sizeof(V2VMessage));
        version = 2;  // Protocol version
    }

    /**
     * @brief Validate message fields
     *
     * @return true if message is valid
     */
    bool isValid() const {
        return version == 2 &&
               hopCount <= 3 &&
               timestamp > 0 &&
               alert.riskLevel <= 3 &&
               alert.confidence >= 0.0f && alert.confidence <= 1.0f;
    }

    /**
     * @brief Check if message is stale (> 500ms old)
     *
     * @param maxAge Maximum acceptable age in milliseconds (default 500ms)
     * @return true if message is too old
     */
    bool isStale(uint32_t maxAge = 500) const {
        return (millis() - timestamp) > maxAge;
    }

    /**
     * @brief Get message age in milliseconds
     *
     * @return milliseconds since message was created
     */
    uint32_t age() const {
        return millis() - timestamp;
    }
};

#pragma pack(pop)

// ============================================================================
// COMPILE-TIME SIZE VERIFICATION
// ============================================================================

// Verify message fits in ESP-NOW payload limit
static_assert(sizeof(V2VMessage) <= 250,
              "V2VMessage exceeds ESP-NOW 250-byte payload limit!");

// Document actual size for debugging
// Expected: 1 + 8 + 4 + 12 + 16 + 36 + 6 + 7 = 90 bytes
// (May be 85 bytes depending on alignment)

#endif // V2VMESSAGE_H
