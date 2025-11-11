/**
 * @file IGpsSensor.h
 * @brief Abstract interface for GPS sensors
 *
 * Defines a hardware-independent interface for GPS/GNSS modules.
 * Implementations: NEO-6M (primary)
 *
 * Based on legacy RoadSense firmware with enhancements for
 * intermittent GPS lock handling.
 */

#ifndef IGPSSENSOR_H
#define IGPSSENSOR_H

#include <Arduino.h>

/**
 * @class IGpsSensor
 * @brief Abstract base class for GPS sensors
 *
 * Provides unified interface for reading GPS position, velocity, and timing.
 * Includes intelligent caching to handle temporary GPS signal loss.
 */
class IGpsSensor {
public:
    /**
     * @struct GpsData
     * @brief Container for GPS measurements
     */
    struct GpsData {
        // Position (WGS84 coordinates)
        double latitude;      // degrees (-90 to +90)
        double longitude;     // degrees (-180 to +180)
        float altitude;       // meters above sea level

        // Velocity
        float speed;          // m/s (ground speed)
        float heading;        // degrees (0-359, true north)

        // Quality indicators
        bool valid;           // true if GPS fix is valid
        uint8_t satellites;   // number of satellites in view
        float hdop;           // Horizontal Dilution of Precision

        // Timing
        uint32_t timestamp;   // millis() when data was read
        bool cached;          // true if using cached data (GPS lost)

        /**
         * @brief Default constructor - zero initialize
         */
        GpsData() : latitude(0.0), longitude(0.0), altitude(0.0f),
                    speed(0.0f), heading(0.0f), valid(false),
                    satellites(0), hdop(99.9f), timestamp(0), cached(false) {}

        /**
         * @brief Check if position data is usable
         *
         * @return true if valid AND not stale (< 30s old)
         */
        bool isUsable() const {
            return valid && (millis() - timestamp < 30000);
        }

        /**
         * @brief Get data age in milliseconds
         *
         * @return milliseconds since data was acquired
         */
        uint32_t age() const {
            return millis() - timestamp;
        }
    };

    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~IGpsSensor() = default;

    /**
     * @brief Initialize GPS sensor hardware
     *
     * Performs:
     * - UART communication setup
     * - Baud rate configuration (typically 9600)
     * - Cold start initialization
     *
     * Note: GPS may take 30-60 seconds to acquire first fix
     *
     * @return true if initialization successful, false otherwise
     */
    virtual bool begin() = 0;

    /**
     * @brief Read current GPS data
     *
     * Attempts to read fresh GPS data. If GPS fix is lost, returns
     * cached last valid reading (up to 30 seconds old).
     *
     * @return GpsData struct with current or cached position
     */
    virtual GpsData read() = 0;

    /**
     * @brief Update GPS parser (call frequently in loop)
     *
     * Feeds incoming UART bytes to GPS parser. Should be called
     * at least once per loop() iteration for reliable operation.
     *
     * @return true if new data was processed
     */
    virtual bool update() = 0;

    /**
     * @brief Check if GPS has ever achieved a fix
     *
     * @return true if at least one valid fix has been acquired
     */
    virtual bool hasEverHadFix() const = 0;

    /**
     * @brief Get sensor name for debugging
     *
     * @return String identifier (e.g., "NEO-6M")
     */
    virtual const char* getName() const = 0;
};

#endif // IGPSSENSOR_H
