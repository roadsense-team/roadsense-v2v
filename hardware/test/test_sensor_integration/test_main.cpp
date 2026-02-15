#include <unity.h>
#include <Arduino.h>
#include <cmath>
#include "sensors/imu/MPU6500Driver.h"
#include "sensors/mag/QMC5883LDriver.h"
#include "sensors/gps/NEO6M_Driver.h"
#include "network/protocol/V2VMessage.h"
#include "config.h"

MPU6500Driver imu;
QMC5883LDriver mag;
NEO6M_Driver gps;

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

void test_dual_initialization() {
    // Initialize both sensors to check for resource conflicts
    bool imuResult = imu.begin();
    bool magResult = mag.begin(Wire);
    bool gpsResult = gps.begin();
    
    TEST_ASSERT_TRUE_MESSAGE(imuResult, "IMU init failed during dual-test");
    TEST_ASSERT_TRUE_MESSAGE(magResult, "Magnetometer init failed during dual-test");
    TEST_ASSERT_TRUE_MESSAGE(gpsResult, "GPS init failed during dual-test");
}

void test_concurrent_update() {
    // Simulate main loop cycle
    
    unsigned long start = millis();
    
    // GPS needs frequent updates to drain UART
    gps.update(); 
    
    // IMU is polled via read() usually, but let's just do a read here to simulate load
    imu.read();
    
    unsigned long duration = millis() - start;
    
    // Ensure updates are efficient
    TEST_ASSERT_LESS_THAN_UINT32(20, duration); // allow 20ms for I2C transaction
}

void test_data_to_v2v_message() {
    // Read from sensors
    IImuSensor::ImuData imuData = imu.read();
    IGpsSensor::GpsData gpsData = gps.read();
    float magX = 0.0f;
    float magY = 0.0f;
    float magZ = 0.0f;
    bool magReadOk = mag.read(magX, magY, magZ);
    
    // Create and populate message
    V2VMessage msg;
    msg.version = 2;
    strncpy(msg.vehicleId, "TEST01", 8);
    msg.timestamp = millis();
    
    // Populate GPS fields
    msg.position.lat = gpsData.latitude;
    msg.position.lon = gpsData.longitude;
    msg.dynamics.speed = gpsData.speed;
    msg.dynamics.heading = gpsData.heading;
    
    // Populate IMU fields
    memcpy(msg.sensors.accel, imuData.accel, sizeof(float)*3);
    memcpy(msg.sensors.gyro, imuData.gyro, sizeof(float)*3);
    msg.sensors.mag[0] = magX;
    msg.sensors.mag[1] = magY;
    msg.sensors.mag[2] = magZ;
    
    // Validation
    TEST_ASSERT_EQUAL_FLOAT(gpsData.latitude, msg.position.lat);
    TEST_ASSERT_EQUAL_FLOAT(imuData.accel[0], msg.sensors.accel[0]);
    TEST_ASSERT_TRUE_MESSAGE(magReadOk, "Mag read failed while building V2V message");
    TEST_ASSERT_TRUE_MESSAGE((fabs(msg.sensors.mag[0]) > 0.001f) ||
                             (fabs(msg.sensors.mag[1]) > 0.001f) ||
                             (fabs(msg.sensors.mag[2]) > 0.001f),
                             "V2V message magnetometer fields are all zero");
    
    // Size check again just to be sure
    TEST_ASSERT_EQUAL_INT(90, sizeof(V2VMessage));
}

void setup() {
    delay(2000);
    UNITY_BEGIN();
    
    RUN_TEST(test_dual_initialization);
    RUN_TEST(test_concurrent_update);
    RUN_TEST(test_data_to_v2v_message);
    
    UNITY_END();
}

void loop() {
    // Keep polling GPS
    gps.update();
}
