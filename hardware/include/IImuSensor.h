/**
 * @file IImuSensor.h
 * @brief Abstract interface for IMU sensors
 *
 * Defines a hardware-independent interface for IMU sensors. On current hardware
 * (MPU6500) only 6 axes are available (accel + gyro); magnetometer fields will
 * be zeroed. Interface keeps mag fields for future 9-axis upgrades.
 */

#ifndef IIMUSENSOR_H
#define IIMUSENSOR_H

#include <Arduino.h>

/**
 * @class IImuSensor
 * @brief Abstract base class for 9-axis IMU sensors
 *
 * Provides a unified interface for reading accelerometer, gyroscope,
 * and magnetometer data. Current hardware (MPU6500) lacks a magnetometer, so
 * mag values will be zeros; interface remains for future 9-axis parts.
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
     * Reads available axes (accelerometer, gyroscope, and magnetometer where
     * supported). Apply calibration corrections if available.
     *
     * @return ImuData struct containing sensor readings and timestamp
     */
    virtual ImuData read() = 0;

    /**
     * @brief Perform sensor calibration
     *
     * For 6-axis hardware (MPU6500), this calibrates accelerometer/gyroscope
     * offsets only. Magnetometer fields are zeroed because the device lacks a
     * magnetometer. Future 9-axis implementations may include mag calibration.
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
     * @return true if available calibration data (accel/gyro or mag) has been loaded
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
