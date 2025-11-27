#include <unity.h>
#include <Arduino.h>
#include "sensors/gps/NEO6M_Driver.h"
#include "config.h"

NEO6M_Driver gps;

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

void test_gps_initialization() {
    // Should return true if Serial1 initialized
    bool result = gps.begin();
    TEST_ASSERT_TRUE(result);
}

void test_gps_read_structure() {
    // Call update (simulates main loop)
    gps.update();
    
    // Read data
    IGpsSensor::GpsData data = gps.read();
    
    // Verify data structure integrity (defaults should be 0/false if no fix)
    // If we have no fix, these should be false
    // If we DO have a fix (outdoors), they might be true
    // So we just assert that the read() command didn't crash and returned a struct
    
    // Check that we get a valid object back
    // (In C++, structs are values, so just checking access)
    float lat = data.latitude;
    float lon = data.longitude;
    (void)lat; (void)lon; // Suppress unused warning
}

void test_gps_caching_logic() {
    // Test the 4-state logic (Fresh -> Cached -> Stale -> None)
    // Without a real GPS signal simulator, we can't easily force a "Fix" state
    // But we can verify the "No Fix" state behavior
    
    IGpsSensor::GpsData data = gps.read();
    
    if (!data.valid && !data.cached) {
        // If no fix, speed and altitude should be 0
        TEST_ASSERT_EQUAL_FLOAT(0.0f, data.speed);
        TEST_ASSERT_EQUAL_FLOAT(0.0f, data.altitude);
    }
}

void test_heading_validation_logic() {
    // We can't inject speed into the driver easily without a mock
    // But we can verify the constants are set correctly in the build
    TEST_ASSERT_EQUAL_FLOAT(1.4f, GPS_HEADING_MIN_SPEED_MPS);
}

void setup() {
    delay(2000);
    UNITY_BEGIN();
    
    RUN_TEST(test_gps_initialization);
    RUN_TEST(test_gps_read_structure);
    RUN_TEST(test_gps_caching_logic);
    RUN_TEST(test_heading_validation_logic);
    
    UNITY_END();
}

void loop() {
    // Must call update() to keep UART buffer drained during tests if they run long
    gps.update();
}
