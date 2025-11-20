/**
 * @file MPU6500Driver.h
 * @brief Driver for MPU6500 6-axis IMU (accelerometer + gyroscope)
 *
 * ⚠️ HARDWARE NOTE: This sensor does NOT have a magnetometer
 * - MPU6500 is 6-axis only (accel + gyro, no magnetometer)
 * - ImuData.mag[3] will always be {0, 0, 0}
 * - calibrate() only calibrates accel/gyro offsets (not magnetometer)
 * - isCalibrated() returns true after accel/gyro calibration
 * - Use GPS heading instead of compass heading
 *
 * @date November 17, 2025
 * @version 1.0
 */

#ifndef MPU6500DRIVER_H
#define MPU6500DRIVER_H

#include <Arduino.h>
#include <Wire.h>
#include <Preferences.h>
#include <MPU6500_WE.h>  // Use MPU6500_WE class (WHO_AM_I = 0x70)
#include "IImuSensor.h"   // PlatformIO adds include/ to search path
#include "utils/Logger.h"
#include "config.h"

/**
 * @class MPU6500Driver
 * @brief Implementation of IImuSensor for MPU6500 6-axis IMU
 *
 * Provides accelerometer and gyroscope data with calibration support.
 * Magnetometer field is always zero (hardware limitation).
 *
 * Hardware: MPU6500 @ 0x68 (WHO_AM_I: 0x70)
 * Library: MPU6500_WE (Wolfgang Ewald)
 */
class MPU6500Driver : public IImuSensor {
private:
    MPU6500_WE mpu;           // Library instance (accepts WHO_AM_I = 0x70)
    bool initialized;
    bool calibrated;
    static_assert(sizeof(xyzFloat) == 12, "xyzFloat must be 12 bytes");

    /**
     * @struct CalibrationData
     * @brief Calibration offsets stored in NVS
     */
    struct CalibrationData {
        xyzFloat accelOffset;  // Accel offsets (raw values)
        xyzFloat gyroOffset;   // Gyro offsets (raw values)
        bool valid;            // True if calibration data exists
    } calibData;

    /**
     * @brief Apply loaded calibration offsets to sensor
     * @note CRITICAL: Must be called after loadCalibration() to apply offsets
     */
    void applyCalibration();

public:
    /**
     * @brief Constructor
     * @param i2cAddr I2C address (default: 0x68, alt: 0x69 if AD0 high)
     */
    MPU6500Driver(uint8_t i2cAddr = MPU6500_I2C_ADDR);

    /**
     * @brief Destructor
     */
    ~MPU6500Driver() override = default;

    /**
     * @brief Initialize IMU sensor
     *
     * Performs:
     * - I2C initialization (SDA=21, SCL=22, 400 kHz)
     * - MPU6500 wake-up and configuration
     * - Accel range: ±4g, Gyro range: ±500 deg/s
     * - Digital low-pass filter (DLPF) enabled
     * - Sample rate: 10 Hz (via divider)
     * - Load calibration from NVS if exists
     * - Auto-calibrate if no stored calibration
     *
     * @return true if initialization successful, false otherwise
     */
    bool begin() override;

    /**
     * @brief Read current IMU measurements
     *
     * Returns 6-axis data (accel + gyro) with unit conversions:
     * - Accelerometer: g-force → m/s² (multiply by 9.80665)
     * - Gyroscope: deg/s → rad/s (multiply by DEG_TO_RAD)
     * - Magnetometer: ALWAYS {0, 0, 0} (hardware not present)
     *
     * @return ImuData struct with sensor readings and timestamp
     */
    ImuData read() override;

    /**
     * @brief Perform accel/gyro offset calibration
     *
     * ⚠️ INTERFACE NOTE: IImuSensor::calibrate() was designed for magnetometer
     * (figure-8 pattern), but MPU6500 has NO magnetometer. This method
     * calibrates accel/gyro offsets only.
     *
     * Requirements:
     * - Sensor must be placed FLAT and STATIONARY
     * - Takes ~3 seconds (500 samples @ 1ms intervals)
     * - Automatically saves calibration to NVS
     *
     * @return true if calibration successful and saved, false otherwise
     */
    bool calibrate() override;

    /**
     * @brief Load calibration data from NVS and apply to sensor
     *
     * ⚠️ CRITICAL: After loading, calls applyCalibration() to write offsets
     * to sensor registers. Without this, calibration is not active!
     *
     * @return true if calibration loaded and applied, false if not found
     */
    bool loadCalibration() override;

    /**
     * @brief Save calibration data to NVS (non-volatile storage)
     *
     * Persists accel/gyro offsets to flash memory for reuse after reboot.
     * Called automatically by calibrate().
     *
     * @return true if saved successfully, false otherwise
     */
    bool saveCalibration() override;

    /**
     * @brief Check if sensor is calibrated
     *
     * @return true after accel/gyro calibration (does NOT wait for magnetometer)
     */
    bool isCalibrated() const override;

    /**
     * @brief Get sensor name for debugging
     *
     * @return "MPU6500" identifier string
     */
    const char* getName() const override { return "MPU6500"; }
};

#endif // MPU6500DRIVER_H
