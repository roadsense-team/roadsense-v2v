/**
 * @file NEO6M_Driver.h
 * @brief NEO-6M GPS module driver implementation
 *
 * Non-blocking GPS driver using TinyGPS++ library.
 * Implements intelligent caching to handle temporary GPS signal loss.
 *
 * Hardware: NEO-6M @ 9600 baud (UART1 remapped to GPIO 16/17)
 * Update rate: 1 Hz (NEO-6M default)
 *
 * Based on legacy RoadSense NEO6M_GPS_Service with improvements:
 * - Non-blocking operation (no delay() calls)
 * - GPS heading extraction (replaces magnetometer on MPU6500)
 * - Enhanced quality indicators (satellites, HDOP)
 * - 4-state cache logic (fresh/cached/stale/none)
 */

#ifndef NEO6M_DRIVER_H
#define NEO6M_DRIVER_H

#include <Arduino.h>
#include <HardwareSerial.h>
#include <TinyGPSPlus.h>
#include "IGpsSensor.h"
#include "utils/Logger.h"
#include "config.h"

/**
 * @class NEO6M_Driver
 * @brief Implementation of IGpsSensor for NEO-6M GPS module
 *
 * Non-blocking GPS driver using TinyGPS++ library.
 * Implements intelligent caching to handle temporary GPS signal loss.
 *
 * Hardware: NEO-6M @ 9600 baud (UART1 remapped to GPIO 16/17)
 * Update rate: 1 Hz (NEO-6M default)
 */
class NEO6M_Driver : public IGpsSensor {
private:
    HardwareSerial gpsSerial;  // Serial port for GPS
    TinyGPSPlus gps;           // TinyGPS++ parser

    bool initialized;
    bool everHadFix;

    // Cache (from legacy - handles GPS dropouts)
    GpsData lastValidReading;
    uint32_t lastValidReadingTime;

    /**
     * @brief Update internal cache when new valid GPS fix arrives
     *
     * Extracts position, speed, heading, quality from TinyGPS++.
     * Only updates heading if valid (moving fast enough).
     * Sets cached=false and timestamp=millis().
     */
    void updateCache();

public:
    /**
     * @brief Constructor
     *
     * Initializes GPS serial port using GPS_SERIAL_PORT constant (value = 1).
     * UART pins will be remapped in begin() to GPS_RX_PIN/GPS_TX_PIN (16/17).
     */
    NEO6M_Driver();

    /**
     * @brief Destructor
     */
    ~NEO6M_Driver() override = default;

    /**
     * @brief Initialize GPS module (non-blocking)
     *
     * - Initializes UART1 at 9600 baud on GPIO 16/17
     * - Does NOT wait for first fix (returns immediately)
     * - Cold start: 30-60 seconds to first fix
     *
     * @return true if serial port initialized, false otherwise
     */
    bool begin() override;

    /**
     * @brief Read cached GPS data (always non-blocking)
     *
     * Returns last valid reading with cache age metadata:
     * - <2s: valid=true, cached=false (fresh)
     * - 2-30s: valid=true, cached=true (cached but valid)
     * - >30s: valid=false, cached=true (stale/expired)
     * - Never had fix: valid=false, cached=false (zeros)
     *
     * @return GpsData struct with position, velocity, quality, metadata
     */
    GpsData read() override;

    /**
     * @brief Poll UART and feed bytes to TinyGPS++ parser
     *
     * CRITICAL: Must be called every loop iteration (~100Hz) to drain UART buffer.
     * Drains all available bytes without blocking.
     * Updates internal cache when complete NMEA sentence parsed.
     *
     * @return true if new GPS sentence was parsed
     */
    bool update() override;

    /**
     * @brief Check if GPS has ever achieved a fix
     *
     * Different from read().valid (which can toggle):
     * - hasEverHadFix(): true after first fix, stays true forever
     * - read().valid: true only if current fix is valid
     *
     * Used for LED status: Blinking = no fix yet, Solid = has fix
     *
     * @return true if at least one valid fix acquired since boot
     */
    bool hasEverHadFix() const override;

    /**
     * @brief Get sensor name for debugging
     *
     * @return "NEO-6M" identifier string
     */
    const char* getName() const override { return "NEO-6M"; }
};

#endif // NEO6M_DRIVER_H
