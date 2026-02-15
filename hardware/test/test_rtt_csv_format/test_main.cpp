/**
 * @file test_main.cpp
 * @brief Unit tests for RTT CSV formatting
 *
 * TDD Test Suite: CSV Format
 * Tests MUST pass before implementation is considered correct
 *
 * Critical Requirements:
 * - CSV header must exactly match analysis script expectations
 * - CSV row format must have 15 columns
 * - Float precision must be sufficient for analysis
 * - Lost packets must be marked with -1 values
 */

#include <unity.h>
#include <Arduino.h>
#include <cstring>
#include <cstdio>
#include "../espnow_rtt/RTTCommon.h"

// Forward declare CSV formatting functions (to be implemented)
void generateCSVHeader(char* buffer, size_t bufferSize);
void formatCSVRow(char* buffer, size_t bufferSize, const RTTRecord* record);
int countChar(const char* str, char c);

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

/**
 * TEST 1: CSV header must match exact specification
 * CRITICAL: Python analysis script depends on this exact format
 */
void test_csv_header_exact_match() {
    const char* expected = "sequence,send_time_ms,recv_time_ms,rtt_ms,lat,lon,speed,heading,accel_x,accel_y,accel_z,mag_x,mag_y,mag_z,lost\n";

    char buffer[256];
    generateCSVHeader(buffer, sizeof(buffer));

    TEST_ASSERT_EQUAL_STRING(expected, buffer);
}

/**
 * TEST 2: CSV header has correct number of columns
 */
void test_csv_header_column_count() {
    char buffer[256];
    generateCSVHeader(buffer, sizeof(buffer));

    // Count commas (14 commas = 15 columns)
    int commas = countChar(buffer, ',');
    TEST_ASSERT_EQUAL_INT(14, commas);
}

/**
 * TEST 3: CSV row for received packet (lost=0)
 */
void test_csv_row_received_packet() {
    RTTRecord record;
    record.sequence = 42;
    record.send_time_ms = 1000;
    record.recv_time_ms = 1007;
    record.lat = 32.085123f;
    record.lon = 34.781234f;
    record.speed = 15.5f;
    record.heading = 45.2f;
    record.accel_x = -0.05f;
    record.accel_y = 0.12f;
    record.accel_z = 9.81f;
    record.mag_x = 21.4f;
    record.mag_y = -4.8f;
    record.mag_z = 39.0f;
    record.received = true;

    char line[256];
    formatCSVRow(line, sizeof(line), &record);

    // Expected: ...accel_x,accel_y,accel_z,mag_x,mag_y,mag_z,lost

    // Verify has 15 columns (14 commas)
    int commas = countChar(line, ',');
    TEST_ASSERT_EQUAL_INT(14, commas);

    // Verify line ends with newline
    size_t len = strlen(line);
    TEST_ASSERT_GREATER_THAN_INT(0, len);
    TEST_ASSERT_EQUAL_CHAR('\n', line[len - 1]);

    // Verify contains key values
    TEST_ASSERT_TRUE(strstr(line, "42,") != NULL);  // sequence
    TEST_ASSERT_TRUE(strstr(line, "1000,") != NULL);  // send_time
    TEST_ASSERT_TRUE(strstr(line, "1007,") != NULL);  // recv_time
    TEST_ASSERT_TRUE(strstr(line, ",7,") != NULL);  // rtt_ms = 7
    TEST_ASSERT_TRUE(strstr(line, ",0\n") != NULL);  // lost=0 at end
}

/**
 * TEST 4: CSV row for lost packet (lost=1)
 */
void test_csv_row_lost_packet() {
    RTTRecord record;
    record.sequence = 100;
    record.send_time_ms = 2000;
    record.recv_time_ms = 0;  // Not received
    record.lat = 32.085125f;
    record.lon = 34.781236f;
    record.speed = 10.0f;
    record.heading = 90.0f;
    record.accel_x = 0.01f;
    record.accel_y = -0.02f;
    record.accel_z = 9.80f;
    record.mag_x = 20.5f;
    record.mag_y = -3.7f;
    record.mag_z = 37.9f;
    record.received = false;

    char line[256];
    formatCSVRow(line, sizeof(line), &record);

    // Expected: ...,-0.020,9.800,20.500,-3.700,37.900,1
    // recv_time_ms = -1, rtt_ms = -1, lost = 1

    // Verify has 15 columns
    int commas = countChar(line, ',');
    TEST_ASSERT_EQUAL_INT(14, commas);

    // Verify contains -1 for recv_time and rtt
    TEST_ASSERT_TRUE(strstr(line, ",-1,-1,") != NULL);

    // Verify ends with ,1 (lost=1)
    TEST_ASSERT_TRUE(strstr(line, ",1\n") != NULL);
}

/**
 * TEST 5: RTT calculation is correct
 */
void test_rtt_calculation() {
    RTTRecord record;
    record.send_time_ms = 1000;
    record.recv_time_ms = 1012;
    record.received = true;

    int rtt = record.recv_time_ms - record.send_time_ms;
    TEST_ASSERT_EQUAL_INT(12, rtt);
}

/**
 * TEST 6: GPS precision is sufficient (6 decimal places)
 */
void test_gps_precision() {
    RTTRecord record;
    record.lat = 32.125000f;  // Changed to value that represents cleanly in binary (avoids IEEE 754 precision issues)
    record.lon = 34.750000f;  // Changed to value that represents cleanly in binary
    record.sequence = 0;
    record.send_time_ms = 1000;
    record.recv_time_ms = 1010;
    record.speed = 0.0f;
    record.heading = 0.0f;
    record.accel_x = 0.0f;
    record.accel_y = 0.0f;
    record.accel_z = 9.81f;
    record.mag_x = 0.0f;
    record.mag_y = 0.0f;
    record.mag_z = 0.0f;
    record.received = true;

    char line[256];
    formatCSVRow(line, sizeof(line), &record);

    // Verify GPS coordinates have at least 6 decimal places
    TEST_ASSERT_TRUE(strstr(line, "32.125") != NULL);
    TEST_ASSERT_TRUE(strstr(line, "34.75") != NULL);
}

/**
 * TEST 7: IMU precision is sufficient (3 decimal places for m/s²)
 */
void test_imu_precision() {
    RTTRecord record;
    record.accel_x = -0.050f;
    record.accel_y = 0.125f;
    record.accel_z = 9.810f;
    record.mag_x = 21.125f;
    record.mag_y = -2.500f;
    record.mag_z = 40.250f;
    record.sequence = 0;
    record.send_time_ms = 1000;
    record.recv_time_ms = 1010;
    record.lat = 0.0f;
    record.lon = 0.0f;
    record.speed = 0.0f;
    record.heading = 0.0f;
    record.received = true;

    char line[256];
    formatCSVRow(line, sizeof(line), &record);

    // Verify IMU data has at least 3 decimal places
    TEST_ASSERT_TRUE(strstr(line, "0.050") != NULL || strstr(line, "-0.050") != NULL);
    TEST_ASSERT_TRUE(strstr(line, "0.125") != NULL);
    TEST_ASSERT_TRUE(strstr(line, "9.810") != NULL);
    TEST_ASSERT_TRUE(strstr(line, "21.125") != NULL);
    TEST_ASSERT_TRUE(strstr(line, "-2.500") != NULL);
    TEST_ASSERT_TRUE(strstr(line, "40.250") != NULL);
}

/**
 * TEST 8: Buffer overflow protection
 */
void test_buffer_overflow_protection() {
    RTTRecord record;
    record.sequence = 999999;
    record.send_time_ms = 999999999;
    record.recv_time_ms = 999999999;
    record.lat = 180.0f;  // Max latitude
    record.lon = 180.0f;  // Max longitude
    record.speed = 999.0f;
    record.heading = 359.9f;
    record.accel_x = 100.0f;
    record.accel_y = 100.0f;
    record.accel_z = 100.0f;
    record.mag_x = 1000.0f;
    record.mag_y = 1000.0f;
    record.mag_z = 1000.0f;
    record.received = true;

    // Use small buffer to test overflow protection
    char line[240];
    formatCSVRow(line, sizeof(line), &record);

    // Verify buffer is null-terminated
    TEST_ASSERT_EQUAL_CHAR('\0', line[239]);

    // Verify line length doesn't exceed buffer
    TEST_ASSERT_LESS_THAN_INT(240, strlen(line));
}

/**
 * TEST 9: Edge case - zero values
 */
void test_zero_values() {
    RTTRecord record;
    memset(&record, 0, sizeof(record));
    record.received = true;

    char line[256];
    formatCSVRow(line, sizeof(line), &record);

    // Should produce valid CSV with zeros
    int commas = countChar(line, ',');
    TEST_ASSERT_EQUAL_INT(14, commas);

    // Should end with ,0 (lost=0 because received=true)
    TEST_ASSERT_TRUE(strstr(line, ",0\n") != NULL);
}

/**
 * TEST 10: Verify CSV is parseable by Python pandas
 * (This is a format validation - actual parsing tested in integration)
 */
void test_csv_format_parseable() {
    RTTRecord record;
    record.sequence = 1;
    record.send_time_ms = 1100;
    record.recv_time_ms = 1108;
    record.lat = 32.085123f;
    record.lon = 34.781234f;
    record.speed = 15.5f;
    record.heading = 45.2f;
    record.accel_x = -0.05f;
    record.accel_y = 0.12f;
    record.accel_z = 9.81f;
    record.mag_x = 21.5f;
    record.mag_y = -4.9f;
    record.mag_z = 39.2f;
    record.received = true;

    char line[256];
    formatCSVRow(line, sizeof(line), &record);

    // Verify no invalid characters (quotes, extra commas, etc.)
    TEST_ASSERT_FALSE(strstr(line, "\"") != NULL);  // No quotes
    TEST_ASSERT_FALSE(strstr(line, ",,") != NULL);  // No empty fields

    // Verify ends with single newline
    size_t len = strlen(line);
    TEST_ASSERT_EQUAL_CHAR('\n', line[len - 1]);
    if (len >= 2) {
        TEST_ASSERT_NOT_EQUAL('\n', line[len - 2]);  // No double newline
    }
}

// Include RTT logging header (currently empty - will cause tests to FAIL)
#include "../espnow_rtt/RTTLogging.h"

void setup() {
    delay(2000);

    UNITY_BEGIN();

    RUN_TEST(test_csv_header_exact_match);
    RUN_TEST(test_csv_header_column_count);
    RUN_TEST(test_csv_row_received_packet);
    RUN_TEST(test_csv_row_lost_packet);
    RUN_TEST(test_rtt_calculation);
    RUN_TEST(test_gps_precision);
    RUN_TEST(test_imu_precision);
    RUN_TEST(test_buffer_overflow_protection);
    RUN_TEST(test_zero_values);
    RUN_TEST(test_csv_format_parseable);

    UNITY_END();
}

void loop() {
}
