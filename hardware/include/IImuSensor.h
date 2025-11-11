/**
 * @file IImuSensor.h
 * @brief Abstract interface for 9-axis Inertial Measurement Units (IMU)
 *
 * Defines a hardware-independent interface for IMU sensors that provide:
 * - 3-axis accelerometer (m/s²)
 * - 3-axis gyroscope (rad/s)
 * - 3-axis magnetometer (μT)
 *
 * Implementations: MPU9250 (primary), MPU6050 (legacy compatibility)
 */

#ifndef IIMUSENSOR_H
#define IIMUSENSOR_H

#include <Arduino.h>

/**
 * @class IImuSensor
 * @brief Abstract base class for 9-axis IMU sensors
 *
 * Provides a unified interface for reading accelerometer, gyroscope,
 * and magnetometer data. Includes calibration support for magnetometer
 * hard-iron and soft-iron distortion correction.
 */
class IImuSensor {
public:
    /**
     * @struct ImuData
     * @brief Container for 9-axis IMU measurements
     */
    struct ImuData {
        // Accelerometer (m/s²) - vehicle frame
        // [0]=x (forward), [1]=y (left), [2]=z (up)
        float accel[3];

        // Gyroscope (rad/s) - angular velocity
        // [0]=roll rate, [1]=pitch rate, [2]=yaw rate
        float gyro[3];

        // Magnetometer (μT) - magnetic field
        // [0]=x (north), [1]=y (east), [2]=z (down) when calibrated
        float mag[3];

        // Timestamp (milliseconds since boot)
        uint32_t timestamp;

        /**
         * @brief Default constructor - zero initialize all fields
         */
        ImuData() : timestamp(0) {
            for (int i = 0; i < 3; i++) {
                accel[i] = 0.0f;
                gyro[i] = 0.0f;
                mag[i] = 0.0f;
            }
        }
    };

    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~IImuSensor() = default;

    /**
     * @brief Initialize the IMU sensor hardware
     *
     * Performs:
     * - I2C communication setup
     * - Sensor wake-up and configuration
     * - Range and sensitivity settings
     * - Sample rate configuration
     *
     * @return true if initialization successful, false otherwise
     */
    virtual bool begin() = 0;

    /**
     * @brief Read current IMU measurements
     *
     * Reads all 9 axes (accelerometer, gyroscope, magnetometer) in a single
     * operation. Apply calibration corrections if available.
     *
     * @return ImuData struct containing sensor readings and timestamp
     */
    virtual ImuData read() = 0;

    /**
     * @brief Perform magnetometer calibration
     *
     * Guides user through figure-8 pattern calibration to determine:
     * - Hard-iron offsets (constant magnetic interference)
     * - Soft-iron scale factors (directional distortion)
     *
     * Typically requires 30 seconds of movement in all orientations.
     *
     * @return true if calibration successful, false otherwise
     */
    virtual bool calibrate() = 0;

    /**
     * @brief Load calibration data from non-volatile storage (NVS)
     *
     * Retrieves previously saved calibration parameters from flash memory.
     * Should be called during begin() if calibration exists.
     *
     * @return true if calibration loaded successfully, false if not found
     */
    virtual bool loadCalibration() = 0;

    /**
     * @brief Save calibration data to non-volatile storage (NVS)
     *
     * Persists calibration parameters to flash memory for reuse after reboot.
     * Called automatically after successful calibrate().
     *
     * @return true if saved successfully, false otherwise
     */
    virtual bool saveCalibration() = 0;

    /**
     * @brief Check if sensor is calibrated
     *
     * @return true if magnetometer calibration has been performed and loaded
     */
    virtual bool isCalibrated() const = 0;

    /**
     * @brief Get sensor name for debugging
     *
     * @return String identifier (e.g., "MPU9250", "MPU6050")
     */
    virtual const char* getName() const = 0;
};

#endif // IIMUSENSOR_H
