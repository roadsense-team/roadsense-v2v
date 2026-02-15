/**
 * @file RTTCommon.h
 * @brief Common definitions for RTT measurement system
 *
 * This header defines shared data structures used across:
 * - RTT packet encoding/decoding (RTTPacket.h)
 * - CSV logging (RTTLogging.h)
 * - Circular buffer (RTTBuffer.h)
 * - Timeout handling (RTTTimeout.h)
 */

#ifndef RTT_COMMON_H
#define RTT_COMMON_H

#include <Arduino.h>

/**
 * @brief RTT record structure - what we log to CSV
 *
 * Represents a single V2V ping packet for round-trip time measurement.
 * Used to characterize ESP-NOW latency and packet loss.
 *
 * Fields:
 * - sequence: Monotonic packet ID (0, 1, 2, ...)
 * - send_time_ms: millis() when packet was sent
 * - recv_time_ms: millis() when echo was received (0 if not received yet)
 * - lat, lon: GPS position when packet was sent
 * - speed: GPS speed in km/h
 * - heading: Compass heading in degrees
 * - accel_x, accel_y, accel_z: IMU acceleration in m/s²
 * - mag_x, mag_y, mag_z: Magnetometer field in microtesla
 * - received: true if echo was received, false if lost/pending
 */
struct RTTRecord {
    uint32_t sequence;
    uint32_t send_time_ms;
    uint32_t recv_time_ms;      // 0 if not received yet, set when echo arrives
    float lat, lon, speed;
    float heading;
    float accel_x, accel_y, accel_z;
    float mag_x, mag_y, mag_z;
    bool received;
};

// Circular buffer configuration
static const int MAX_PENDING = 100;

// Timeout configuration
static const uint32_t TIMEOUT_MS = 500;

#endif // RTT_COMMON_H
