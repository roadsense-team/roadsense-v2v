/**
 * @file test_main.cpp
 * @brief Unit tests for RTT packet structure and protocol
 *
 * TDD Test Suite: RTT Packet Protocol
 * Tests MUST pass before implementation is considered correct
 *
 * Critical Requirements:
 * - Packet size MUST be exactly 90 bytes (matches V2VMessage)
 * - Packet must be properly packed (#pragma pack)
 * - Fields must serialize/deserialize correctly
 * - IMU data fields are REQUIRED (for sensor noise characterization)
 */

#include <unity.h>
#include <Arduino.h>
#include <cstring>

// Include RTT packet header (currently empty - will cause tests to FAIL)
#include "../espnow_rtt/RTTPacket.h"

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

/**
 * TEST 1: Verify packet size is exactly 90 bytes
 * CRITICAL: Must match V2VMessage size for realistic ESP-NOW measurement
 */
void test_rtt_packet_size_is_exactly_90_bytes() {
    TEST_ASSERT_EQUAL_INT(90, sizeof(RTTPacket));
}

/**
 * TEST 2: Verify packet is properly packed (no padding between fields)
 */
void test_rtt_packet_is_properly_packed() {
    RTTPacket pkt;

    // Verify field offsets
    TEST_ASSERT_EQUAL_INT(0, offsetof(RTTPacket, sequence));
    TEST_ASSERT_EQUAL_INT(4, offsetof(RTTPacket, send_time_ms));
    TEST_ASSERT_EQUAL_INT(8, offsetof(RTTPacket, sender_lat));
    TEST_ASSERT_EQUAL_INT(12, offsetof(RTTPacket, sender_lon));
    TEST_ASSERT_EQUAL_INT(16, offsetof(RTTPacket, sender_speed));
    TEST_ASSERT_EQUAL_INT(20, offsetof(RTTPacket, sender_heading));
    TEST_ASSERT_EQUAL_INT(24, offsetof(RTTPacket, accel_x));
    TEST_ASSERT_EQUAL_INT(28, offsetof(RTTPacket, accel_y));
    TEST_ASSERT_EQUAL_INT(32, offsetof(RTTPacket, accel_z));
    TEST_ASSERT_EQUAL_INT(36, offsetof(RTTPacket, padding));
}

/**
 * TEST 3: Verify packet initialization
 */
void test_rtt_packet_initialization() {
    RTTPacket pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.sequence = 42;
    pkt.send_time_ms = 1000;
    pkt.sender_lat = 32.085123f;
    pkt.sender_lon = 34.781234f;
    pkt.sender_speed = 15.5f;
    pkt.sender_heading = 45.2f;
    pkt.accel_x = -0.05f;
    pkt.accel_y = 0.12f;
    pkt.accel_z = 9.81f;
    memset(pkt.padding, 0xAA, sizeof(pkt.padding));

    // Verify all fields
    TEST_ASSERT_EQUAL_UINT32(42, pkt.sequence);
    TEST_ASSERT_EQUAL_UINT32(1000, pkt.send_time_ms);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 32.085123f, pkt.sender_lat);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 34.781234f, pkt.sender_lon);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 15.5f, pkt.sender_speed);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 45.2f, pkt.sender_heading);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -0.05f, pkt.accel_x);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.12f, pkt.accel_y);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 9.81f, pkt.accel_z);
    TEST_ASSERT_EQUAL_UINT8(0xAA, pkt.padding[0]);
    TEST_ASSERT_EQUAL_UINT8(0xAA, pkt.padding[53]);
}

/**
 * TEST 4: Verify packet serialization/deserialization
 * CRITICAL: Packet must survive ESP-NOW transmission unchanged
 */
void test_rtt_packet_serialization() {
    // Create source packet
    RTTPacket sourcePacket;
    memset(&sourcePacket, 0, sizeof(sourcePacket));

    sourcePacket.sequence = 123;
    sourcePacket.send_time_ms = 5000;
    sourcePacket.sender_lat = 32.085123f;
    sourcePacket.sender_lon = 34.781234f;
    sourcePacket.sender_speed = 20.5f;
    sourcePacket.sender_heading = 90.0f;
    sourcePacket.accel_x = 0.10f;
    sourcePacket.accel_y = -0.05f;
    sourcePacket.accel_z = 9.78f;
    memset(sourcePacket.padding, 0xBB, sizeof(sourcePacket.padding));

    // Simulate ESP-NOW transmission (serialize to byte buffer)
    uint8_t buffer[sizeof(RTTPacket)];
    memcpy(buffer, &sourcePacket, sizeof(RTTPacket));

    // Simulate reception (deserialize from byte buffer)
    RTTPacket receivedPacket;
    memcpy(&receivedPacket, buffer, sizeof(RTTPacket));

    // Verify all fields match exactly
    TEST_ASSERT_EQUAL_UINT32(123, receivedPacket.sequence);
    TEST_ASSERT_EQUAL_UINT32(5000, receivedPacket.send_time_ms);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 32.085123f, receivedPacket.sender_lat);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 34.781234f, receivedPacket.sender_lon);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 20.5f, receivedPacket.sender_speed);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 90.0f, receivedPacket.sender_heading);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.10f, receivedPacket.accel_x);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -0.05f, receivedPacket.accel_y);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 9.78f, receivedPacket.accel_z);
    TEST_ASSERT_EQUAL_UINT8(0xBB, receivedPacket.padding[0]);
}

/**
 * TEST 5: Verify IMU fields are present (REQUIRED for sensor noise characterization)
 */
void test_rtt_packet_has_imu_fields() {
    RTTPacket pkt;

    // Set IMU data
    pkt.accel_x = 1.0f;
    pkt.accel_y = 2.0f;
    pkt.accel_z = 9.81f;

    // Verify fields exist and can be set
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, pkt.accel_x);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 2.0f, pkt.accel_y);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 9.81f, pkt.accel_z);
}

/**
 * TEST 6: Verify GPS fields are present (REQUIRED for distance correlation)
 */
void test_rtt_packet_has_gps_fields() {
    RTTPacket pkt;

    // Set GPS data
    pkt.sender_lat = 32.0f;
    pkt.sender_lon = 34.0f;
    pkt.sender_speed = 10.0f;
    pkt.sender_heading = 45.0f;

    // Verify fields exist and can be set
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 32.0f, pkt.sender_lat);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 34.0f, pkt.sender_lon);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 10.0f, pkt.sender_speed);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 45.0f, pkt.sender_heading);
}

/**
 * TEST 7: Verify padding size is correct
 */
void test_rtt_packet_padding_size() {
    // Total packet size: 90 bytes
    // Fields: 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 = 36 bytes
    // Padding: 90 - 36 = 54 bytes
    RTTPacket pkt;
    TEST_ASSERT_EQUAL_INT(54, sizeof(pkt.padding));
}

/**
 * TEST 8: Verify packet can be used with ESP-NOW send/receive
 * (Mock test - actual ESP-NOW tested in integration)
 */
void test_rtt_packet_espnow_compatibility() {
    RTTPacket pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.sequence = 999;
    pkt.send_time_ms = 12345;

    // Simulate ESP-NOW send (cast to uint8_t*)
    uint8_t* pktBytes = (uint8_t*)&pkt;
    TEST_ASSERT_EQUAL_UINT8(231, pktBytes[0]);  // 999 & 0xFF = 231
    TEST_ASSERT_EQUAL_UINT8(3, pktBytes[1]);    // (999 >> 8) & 0xFF = 3

    // Verify size for ESP-NOW (must be <= 250 bytes)
    TEST_ASSERT_LESS_OR_EQUAL_INT(250, sizeof(RTTPacket));
}

void setup() {
    delay(2000);

    UNITY_BEGIN();

    RUN_TEST(test_rtt_packet_size_is_exactly_90_bytes);
    RUN_TEST(test_rtt_packet_is_properly_packed);
    RUN_TEST(test_rtt_packet_initialization);
    RUN_TEST(test_rtt_packet_serialization);
    RUN_TEST(test_rtt_packet_has_imu_fields);
    RUN_TEST(test_rtt_packet_has_gps_fields);
    RUN_TEST(test_rtt_packet_padding_size);
    RUN_TEST(test_rtt_packet_espnow_compatibility);

    UNITY_END();
}

void loop() {
}
