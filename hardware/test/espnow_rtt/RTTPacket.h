/**
 * @file RTTPacket.h
 * @brief RTT packet structure for ESP-NOW characterization
 *
 * This packet is used to measure round-trip time (RTT) between two ESP32 units
 * for ESP-NOW network characterization. The packet includes GPS and IMU data
 * for correlation analysis.
 *
 * CRITICAL: Packet size MUST be exactly 90 bytes to match V2VMessage size
 * for realistic ESP-NOW performance measurement.
 */

#ifndef RTT_PACKET_H
#define RTT_PACKET_H

#include <Arduino.h>

#pragma pack(push, 1)
struct RTTPacket {
    // Network characterization fields
    uint32_t sequence;          // Packet sequence number (0, 1, 2, ...)  - 4 bytes
    uint32_t send_time_ms;      // Sender's millis() at send time        - 4 bytes

    // GPS fields (for distance correlation)
    float sender_lat;           // GPS latitude                          - 4 bytes
    float sender_lon;           // GPS longitude                         - 4 bytes
    float sender_speed;         // GPS speed (m/s)                       - 4 bytes
    float sender_heading;       // GPS heading (degrees)                 - 4 bytes

    // IMU fields (REQUIRED for sensor noise characterization)
    float accel_x;              // Accelerometer X (m/s²)                - 4 bytes
    float accel_y;              // Accelerometer Y (m/s²)                - 4 bytes
    float accel_z;              // Accelerometer Z (m/s²)                - 4 bytes

    // Padding to match V2VMessage size (90 bytes)
    uint8_t padding[54];        // 90 - 36 = 54 bytes
};
#pragma pack(pop)

// Compile-time size verification
static_assert(sizeof(RTTPacket) == 90, "RTTPacket must be exactly 90 bytes");

#endif // RTT_PACKET_H
