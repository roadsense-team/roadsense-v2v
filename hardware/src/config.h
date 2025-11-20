/**
 * @file config.h
 * @brief RoadSense V2V - Vehicle Configuration
 *
 * Hardware configuration for ESP32 vehicle units.
 * Change VEHICLE_ID for each unit: "V001", "V002", "V003"
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ============================================================================
// VEHICLE IDENTIFICATION
// ============================================================================

/**
 * Unique vehicle identifier
 * IMPORTANT: Change this for each ESP32 unit!
 * - Vehicle 1: "V001"
 * - Vehicle 2: "V002"
 * - Vehicle 3: "V003"
 */
#define VEHICLE_ID "V001"

// ============================================================================
// HARDWARE PIN ASSIGNMENTS
// ============================================================================

// MPU6500 6-Axis IMU (I2C) - ⚠️ Hardware: MPU6500 not MPU9250
#define I2C_SDA_PIN       21
#define I2C_SCL_PIN       22
#define I2C_CLOCK_HZ      400000  // 400 kHz (fast mode I2C)
#define MPU6500_I2C_ADDR  0x68    // MPU6500 I2C address (WHO_AM_I: 0x70)

// NEO-6M GPS (UART)
#define GPS_RX_PIN        16    // ESP32 RX ← GPS TX
#define GPS_TX_PIN        17    // ESP32 TX → GPS RX
#define GPS_BAUD_RATE     9600

// Status LED
#define LED_STATUS_PIN    2     // Built-in LED on most ESP32 boards

// Calibration button
#define BUTTON_CALIB_PIN  0     // BOOT button (GPIO 0)

// ============================================================================
// SENSOR CONFIGURATION
// ============================================================================

// IMU sampling rate
#define IMU_SAMPLE_RATE_HZ      10    // 10 Hz = 100ms interval
#define IMU_SAMPLE_INTERVAL_MS  (1000 / IMU_SAMPLE_RATE_HZ)

// IMU hardware configuration (MPU6500)
#define IMU_ACCEL_RANGE_G       4     // ±4g (covers vehicle dynamics: ±2g typical, 4g for hard braking)
#define IMU_GYRO_RANGE_DPS      500   // ±500 deg/s (covers aggressive turns: ~200 deg/s max)
#define IMU_DLPF_BANDWIDTH_HZ   5     // Digital low-pass filter bandwidth (DLPF_6 = 5 Hz)
#define IMU_SAMPLE_RATE_DIV     ((1000 / IMU_SAMPLE_RATE_HZ) - 1)  // Divider: 99 for 10 Hz

// GPS sampling rate
#define GPS_SAMPLE_RATE_HZ      1     // 1 Hz = 1000ms interval
#define GPS_CACHE_TIMEOUT_MS    30000 // 30 seconds

// ⚠️ Magnetometer calibration NOT APPLICABLE (MPU6500 has no magnetometer)
// GPS heading used instead of compass heading

// ============================================================================
// NETWORK CONFIGURATION
// ============================================================================

// ESP-NOW settings
#define ESPNOW_CHANNEL          1
#define ESPNOW_MAX_PAYLOAD      250   // bytes
#define ESPNOW_RETRY_COUNT      3

// V2V message settings
#define V2V_BROADCAST_RATE_HZ   10    // Normal: 10 Hz (100ms)
#define V2V_ALERT_RATE_HZ       20    // High risk: 20 Hz (50ms)
#define V2V_MESSAGE_TIMEOUT_MS  500   // Discard messages older than 500ms

// Mesh networking
#define MAX_HOP_COUNT           3     // Maximum relay hops
#define PEER_TIMEOUT_MS         60000 // 60 seconds

// Mesh message management
#define MAC_ADDRESS_LENGTH      6       // Standard 6-byte MAC address
#define MAC_STRING_LENGTH       18      // "XX:XX:XX:XX:XX:XX\0"
#define PACKAGE_TIMEOUT_MS      60000   // 60 seconds (message expiry)
#define MAX_PACKAGES_PER_SOURCE 3       // Limit packages per source MAC
#define MAX_TRACKED_MACS        20      // Maximum unique MAC addresses to track (3 vehicles × 3 = 9 expected, 20 allows safety margin)

// ============================================================================
// MACHINE LEARNING CONFIGURATION
// ============================================================================

// ML inference settings
#define ML_INFERENCE_INTERVAL_MS  100  // 10 Hz
#define ML_TENSOR_ARENA_SIZE      10240 // 10 KB (adjust if needed)

// Feature extraction
#define ML_FEATURE_COUNT        30    // 3 vehicles × 10 features each
#define ML_MAX_VEHICLES         3     // Maximum vehicles to track

// ============================================================================
// PERFORMANCE TARGETS
// ============================================================================

// Latency requirements
#define TARGET_SENSOR_LATENCY_MS    10   // Sensor read < 10ms
#define TARGET_ML_LATENCY_MS        100  // ML inference < 100ms
#define TARGET_NETWORK_LATENCY_MS   50   // ESP-NOW send < 50ms
#define TARGET_END_TO_END_MS        100  // Total system < 100ms

// Accuracy requirements
#define TARGET_DETECTION_ACCURACY   0.85  // 85% true positive rate
#define TARGET_FALSE_POSITIVE_RATE  0.10  // 10% false positive rate

// ============================================================================
// DEBUG SETTINGS
// ============================================================================

// Enable/disable debug output
#define DEBUG_SERIAL            true
#define DEBUG_SENSOR_DATA       false
#define DEBUG_NETWORK_MESSAGES  false
#define DEBUG_ML_INFERENCE      false

// Serial baud rate
#define SERIAL_BAUD_RATE        115200

// ============================================================================
// SYSTEM STATES
// ============================================================================

// LED blink patterns (milliseconds)
#define LED_BLINK_INIT          100   // Fast blink during initialization
#define LED_BLINK_GPS_WAIT      500   // Slow blink waiting for GPS fix
#define LED_BLINK_NORMAL        0     // Solid green (no risk)
#define LED_BLINK_LOW_RISK      1000  // Slow yellow blink
#define LED_BLINK_MEDIUM_RISK   300   // Fast yellow blink
#define LED_BLINK_HIGH_RISK     100   // Very fast red blink
#define LED_BLINK_ERROR         50    // Ultra fast blink (error state)

#endif // CONFIG_H
