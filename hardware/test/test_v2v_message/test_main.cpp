#include <unity.h>
#include <Arduino.h>
#include "network/protocol/V2VMessage.h"

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

void test_v2v_message_size() {
    // CRITICAL: Must be exactly 90 bytes for bridge compatibility
    TEST_ASSERT_EQUAL_INT(90, sizeof(V2VMessage));
}

void test_v2v_message_initialization() {
    V2VMessage msg;
    
    // Constructor should set version to 2
    TEST_ASSERT_EQUAL_UINT8(2, msg.version);
    
    // Other fields should be zeroed
    TEST_ASSERT_EQUAL_FLOAT(0.0f, msg.position.lat);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, msg.dynamics.speed);
    TEST_ASSERT_EQUAL_UINT8(0, msg.hopCount);
}

void test_v2v_message_serialization() {
    V2VMessage sourceMsg;
    sourceMsg.version = 2;
    strncpy(sourceMsg.vehicleId, "V001", 8);
    sourceMsg.timestamp = 12345678;
    sourceMsg.position.lat = 32.123f;
    sourceMsg.position.lon = 34.567f;
    sourceMsg.dynamics.speed = 15.5f;
    sourceMsg.alert.riskLevel = 2;
    
    // Simulate serialization to byte buffer (network transmission)
    uint8_t buffer[sizeof(V2VMessage)];
    memcpy(buffer, &sourceMsg, sizeof(V2VMessage));
    
    // Simulate deserialization
    V2VMessage destMsg;
    memcpy(&destMsg, buffer, sizeof(V2VMessage));
    
    // Verify fields match exactly
    TEST_ASSERT_EQUAL_UINT8(2, destMsg.version);
    TEST_ASSERT_EQUAL_STRING("V001", destMsg.vehicleId);
    TEST_ASSERT_EQUAL_UINT32(12345678, destMsg.timestamp);
    TEST_ASSERT_EQUAL_FLOAT(32.123f, destMsg.position.lat);
    TEST_ASSERT_EQUAL_FLOAT(34.567f, destMsg.position.lon);
    TEST_ASSERT_EQUAL_FLOAT(15.5f, destMsg.dynamics.speed);
    TEST_ASSERT_EQUAL_UINT8(2, destMsg.alert.riskLevel);
}

void test_v2v_message_validation() {
    V2VMessage msg;
    msg.version = 2;
    msg.hopCount = 0;
    msg.timestamp = 1000;
    msg.alert.riskLevel = 0;
    msg.alert.confidence = 0.5f;
    
    TEST_ASSERT_TRUE(msg.isValid());
    
    // Invalid version
    msg.version = 1;
    TEST_ASSERT_FALSE(msg.isValid());
    msg.version = 2;
    
    // Invalid hop count
    msg.hopCount = 4;
    TEST_ASSERT_FALSE(msg.isValid());
    msg.hopCount = 3;
    
    // Invalid risk level
    msg.alert.riskLevel = 4;
    TEST_ASSERT_FALSE(msg.isValid());
    msg.alert.riskLevel = 3;
    
    // Invalid confidence
    msg.alert.confidence = 1.5f;
    TEST_ASSERT_FALSE(msg.isValid());
}

void setup() {
    delay(2000);
    
    UNITY_BEGIN();
    
    RUN_TEST(test_v2v_message_size);
    RUN_TEST(test_v2v_message_initialization);
    RUN_TEST(test_v2v_message_serialization);
    RUN_TEST(test_v2v_message_validation);
    
    UNITY_END();
}

void loop() {
}
