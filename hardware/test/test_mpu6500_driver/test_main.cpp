#include <unity.h>
#include <Arduino.h>
#include "sensors/imu/MPU6500Driver.h"
#include "config.h"

MPU6500Driver imu;

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

void test_imu_initialization() {
    // This runs begin(), which checks WHO_AM_I (0x70)
    // and initializes I2C connection
    bool result = imu.begin();
    
    if (!result) {
        TEST_FAIL_MESSAGE("IMU initialization failed! Check wiring (SDA=21, SCL=22) or WHO_AM_I value.");
    }
    TEST_ASSERT_TRUE(result);
}

void test_accel_readings_stationary() {
    // Read sensor data
    IImuSensor::ImuData data = imu.read();
    
    // Verify data is not all zeros (driver returning valid data)
    // Note: Absolute zero is statistically impossible for noise
    bool isNonZero = (abs(data.accel[0]) > 0.001f) || 
                     (abs(data.accel[1]) > 0.001f) || 
                     (abs(data.accel[2]) > 0.001f);
    TEST_ASSERT_TRUE_MESSAGE(isNonZero, "Accelerometer data is all zeros!");

    // Verify Z-axis is roughly 1G (9.8 m/s^2) assuming flat on desk
    // Allow generous tolerance (±2 m/s^2) as board might be tilted
    TEST_ASSERT_FLOAT_WITHIN(3.0f, 9.81f, abs(data.accel[2]));
}

void test_gyro_readings_stationary() {
    IImuSensor::ImuData data = imu.read();
    
    // Verify gyro is roughly zero (stationary)
    // Tolerance: 0.2 rad/s (~11 degrees/s)
    TEST_ASSERT_FLOAT_WITHIN(0.2f, 0.0f, data.gyro[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.2f, 0.0f, data.gyro[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.2f, 0.0f, data.gyro[2]);
}

void test_magnetometer_missing() {
    IImuSensor::ImuData data = imu.read();
    
    // MPU6500 has no magnetometer, driver must return 0s
    TEST_ASSERT_EQUAL_FLOAT(0.0f, data.mag[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, data.mag[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, data.mag[2]);
}

void test_calibration_save_load() {
    // Test that we can save and load calibration without crashing
    // We won't run full calibration (requires board flat + still) 
    // but we can check the load/save logic
    
    // Note: MPU6500Driver::saveCalibration() implementation writes to NVS
    bool saveResult = imu.saveCalibration();
    TEST_ASSERT_TRUE(saveResult);
    
    bool loadResult = imu.loadCalibration();
    TEST_ASSERT_TRUE(loadResult);
    
    TEST_ASSERT_TRUE(imu.isCalibrated());
}

void setup() {
    delay(2000); // Wait for board stability
    UNITY_BEGIN();
    
    RUN_TEST(test_imu_initialization);
    RUN_TEST(test_accel_readings_stationary);
    RUN_TEST(test_gyro_readings_stationary);
    RUN_TEST(test_magnetometer_missing);
    RUN_TEST(test_calibration_save_load);
    
    UNITY_END();
}

void loop() {
}
