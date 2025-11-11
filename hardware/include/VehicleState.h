/**
 * @file VehicleState.h
 * @brief Vehicle state data structure
 *
 * Represents the complete state of a vehicle including:
 * - Identity (vehicle ID)
 * - Position (GPS coordinates)
 * - Dynamics (velocity, acceleration, heading)
 * - Raw sensor data (IMU readings)
 * - Timing information
 */

#ifndef VEHICLESTATE_H
#define VEHICLESTATE_H

#include <Arduino.h>

/**
 * @struct VehicleState
 * @brief Complete vehicle state representation
 *
 * Used internally for own vehicle state and for storing peer vehicle states.
 */
struct VehicleState {
    // ========================================================================
    // IDENTITY
    // ========================================================================
    char vehicleId[8];        // "V001", "V002", "V003"

    // ========================================================================
    // POSITION (from GPS)
    // ========================================================================
    double lat;               // Latitude (degrees)
    double lon;               // Longitude (degrees)
    float alt;                // Altitude (meters)

    // ========================================================================
    // DYNAMICS (from GPS + IMU fusion)
    // ========================================================================
    float speed;              // Speed (m/s)
    float heading;            // Heading (degrees, 0-359, true north)
    float longAccel;          // Longitudinal acceleration (m/s²)
    float latAccel;           // Lateral acceleration (m/s²)

    // ========================================================================
    // RAW SENSOR DATA (from IMU)
    // ========================================================================
    float accel[3];           // Accelerometer (m/s²) [x, y, z]
    float gyro[3];            // Gyroscope (rad/s) [x, y, z]
    float mag[3];             // Magnetometer (μT) [x, y, z]

    // ========================================================================
    // TIMING
    // ========================================================================
    uint32_t timestamp;       // millis() when state was updated

    /**
     * @brief Default constructor - zero initialize all fields
     */
    VehicleState() : lat(0.0), lon(0.0), alt(0.0f),
                     speed(0.0f), heading(0.0f),
                     longAccel(0.0f), latAccel(0.0f),
                     timestamp(0) {
        memset(vehicleId, 0, sizeof(vehicleId));
        memset(accel, 0, sizeof(accel));
        memset(gyro, 0, sizeof(gyro));
        memset(mag, 0, sizeof(mag));
    }

    /**
     * @brief Get age of this state in milliseconds
     *
     * @return milliseconds since state was last updated
     */
    uint32_t age() const {
        return millis() - timestamp;
    }

    /**
     * @brief Check if state is stale (> 500ms old)
     *
     * @return true if state is too old to use
     */
    bool isStale() const {
        return age() > 500;
    }

    /**
     * @brief Check if state is valid (has meaningful data)
     *
     * @return true if vehicle ID is set and timestamp is recent
     */
    bool isValid() const {
        return (vehicleId[0] != '\0') && (timestamp > 0) && !isStale();
    }
};

#endif // VEHICLESTATE_H
