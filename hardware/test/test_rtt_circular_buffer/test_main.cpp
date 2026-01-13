/**
 * @file test_main.cpp
 * @brief Unit tests for RTT circular buffer (pending packets tracking)
 *
 * TDD Test Suite: Circular Buffer
 * Tests MUST pass before implementation is considered correct
 *
 * Critical Requirements:
 * - Buffer must handle wraparound correctly
 * - Collisions must be handled (old packets overwritten)
 * - Index calculation must be correct (sequence % MAX_PENDING)
 * - Lost packet detection must work after timeout
 */

#include <unity.h>
#include <Arduino.h>
#include <cstring>
#include "../espnow_rtt/RTTCommon.h"

// Global circular buffer (to be used by implementation)
RTTRecord pendingPackets[MAX_PENDING];

// Forward declare buffer management functions (to be implemented)
void initCircularBuffer();
void addPendingPacket(const RTTRecord* record);
void markPacketReceived(uint32_t sequence, uint32_t recv_time_ms);
bool isPacketPending(uint32_t sequence);
bool isPacketReceived(uint32_t sequence);
int getBufferIndex(uint32_t sequence);

void setUp(void) {
    // Clear buffer before each test
    memset(pendingPackets, 0, sizeof(pendingPackets));
    initCircularBuffer();
}

void tearDown(void) {
    // Runs after each test
}

/**
 * TEST 1: Buffer initialization clears all slots
 */
void test_buffer_initialization() {
    initCircularBuffer();

    for (int i = 0; i < MAX_PENDING; i++) {
        TEST_ASSERT_EQUAL_UINT32(0, pendingPackets[i].sequence);
        TEST_ASSERT_EQUAL_UINT32(0, pendingPackets[i].send_time_ms);
        TEST_ASSERT_FALSE(pendingPackets[i].received);
    }
}

/**
 * TEST 2: Index calculation is correct
 */
void test_buffer_index_calculation() {
    TEST_ASSERT_EQUAL_INT(0, getBufferIndex(0));
    TEST_ASSERT_EQUAL_INT(1, getBufferIndex(1));
    TEST_ASSERT_EQUAL_INT(99, getBufferIndex(99));
    TEST_ASSERT_EQUAL_INT(0, getBufferIndex(100));  // Wraparound
    TEST_ASSERT_EQUAL_INT(1, getBufferIndex(101));
    TEST_ASSERT_EQUAL_INT(42, getBufferIndex(142));
    TEST_ASSERT_EQUAL_INT(0, getBufferIndex(1000));  // 1000 % 100 = 0
}

/**
 * TEST 3: Add pending packet to buffer
 */
void test_add_pending_packet() {
    RTTRecord record;
    record.sequence = 42;
    record.send_time_ms = 1000;
    record.recv_time_ms = 0;
    record.lat = 32.0f;
    record.lon = 34.0f;
    record.speed = 10.0f;
    record.heading = 45.0f;
    record.accel_x = 0.1f;
    record.accel_y = -0.05f;
    record.accel_z = 9.81f;
    record.received = false;

    addPendingPacket(&record);

    // Verify packet is in correct slot (42 % 100 = 42)
    TEST_ASSERT_EQUAL_UINT32(42, pendingPackets[42].sequence);
    TEST_ASSERT_EQUAL_UINT32(1000, pendingPackets[42].send_time_ms);
    TEST_ASSERT_FALSE(pendingPackets[42].received);
}

/**
 * TEST 4: Mark packet as received
 */
void test_mark_packet_received() {
    // Add pending packet
    RTTRecord record;
    record.sequence = 10;
    record.send_time_ms = 1000;
    record.received = false;

    addPendingPacket(&record);
    TEST_ASSERT_FALSE(pendingPackets[10].received);

    // Mark as received
    markPacketReceived(10, 1007);

    TEST_ASSERT_TRUE(pendingPackets[10].received);
    TEST_ASSERT_EQUAL_UINT32(1007, pendingPackets[10].recv_time_ms);
}

/**
 * TEST 5: Buffer wraparound (sequence 100 overwrites slot 0)
 */
void test_buffer_wraparound() {
    // Add packet with sequence 0
    RTTRecord record0;
    record0.sequence = 0;
    record0.send_time_ms = 1000;
    record0.received = false;

    addPendingPacket(&record0);
    TEST_ASSERT_EQUAL_UINT32(0, pendingPackets[0].sequence);
    TEST_ASSERT_EQUAL_UINT32(1000, pendingPackets[0].send_time_ms);

    // Add packet with sequence 100 (same slot: 100 % 100 = 0)
    RTTRecord record100;
    record100.sequence = 100;
    record100.send_time_ms = 2000;
    record100.received = false;

    addPendingPacket(&record100);

    // Slot 0 should now contain sequence 100 (overwrites 0)
    TEST_ASSERT_EQUAL_UINT32(100, pendingPackets[0].sequence);
    TEST_ASSERT_EQUAL_UINT32(2000, pendingPackets[0].send_time_ms);
}

/**
 * TEST 6: Detect packet is pending
 */
void test_is_packet_pending() {
    RTTRecord record;
    record.sequence = 50;
    record.send_time_ms = 1500;
    record.received = false;

    addPendingPacket(&record);

    TEST_ASSERT_TRUE(isPacketPending(50));
    TEST_ASSERT_FALSE(isPacketPending(51));  // Not added
}

/**
 * TEST 7: Detect packet is received
 */
void test_is_packet_received() {
    RTTRecord record;
    record.sequence = 60;
    record.send_time_ms = 1600;
    record.received = false;

    addPendingPacket(&record);
    TEST_ASSERT_FALSE(isPacketReceived(60));

    markPacketReceived(60, 1610);
    TEST_ASSERT_TRUE(isPacketReceived(60));
}

/**
 * TEST 8: Multiple packets in buffer
 */
void test_multiple_packets() {
    // Add 10 packets
    for (uint32_t i = 0; i < 10; i++) {
        RTTRecord record;
        record.sequence = i;
        record.send_time_ms = 1000 + (i * 100);
        record.received = false;

        addPendingPacket(&record);
    }

    // Verify all packets are in buffer
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_EQUAL_UINT32(i, pendingPackets[i].sequence);
        TEST_ASSERT_EQUAL_UINT32(1000 + (i * 100), pendingPackets[i].send_time_ms);
        TEST_ASSERT_FALSE(pendingPackets[i].received);
    }
}

/**
 * TEST 9: Buffer full scenario (100 packets)
 */
void test_buffer_full() {
    // Fill buffer completely
    for (uint32_t i = 0; i < MAX_PENDING; i++) {
        RTTRecord record;
        record.sequence = i;
        record.send_time_ms = 1000 + (i * 10);
        record.received = false;

        addPendingPacket(&record);
    }

    // Verify all 100 slots are filled
    for (int i = 0; i < MAX_PENDING; i++) {
        TEST_ASSERT_EQUAL_UINT32(i, pendingPackets[i].sequence);
    }

    // Add packet 100 (overwrites slot 0)
    RTTRecord record100;
    record100.sequence = 100;
    record100.send_time_ms = 2000;
    record100.received = false;

    addPendingPacket(&record100);

    TEST_ASSERT_EQUAL_UINT32(100, pendingPackets[0].sequence);
    TEST_ASSERT_EQUAL_UINT32(2000, pendingPackets[0].send_time_ms);
}

/**
 * TEST 10: Collision handling (new packet overwrites old unreceived packet)
 */
void test_collision_overwrites_old_packet() {
    // Add packet 0
    RTTRecord record0;
    record0.sequence = 0;
    record0.send_time_ms = 1000;
    record0.lat = 32.0f;
    record0.received = false;

    addPendingPacket(&record0);

    // Add packet 100 (collides with slot 0)
    RTTRecord record100;
    record100.sequence = 100;
    record100.send_time_ms = 2000;
    record100.lat = 33.0f;  // Different value
    record100.received = false;

    addPendingPacket(&record100);

    // Verify packet 0 was overwritten
    TEST_ASSERT_EQUAL_UINT32(100, pendingPackets[0].sequence);
    TEST_ASSERT_EQUAL_UINT32(2000, pendingPackets[0].send_time_ms);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 33.0f, pendingPackets[0].lat);
}

/**
 * TEST 11: Received packets should not match wrong sequence
 */
void test_received_packet_sequence_verification() {
    // Add packet 50
    RTTRecord record50;
    record50.sequence = 50;
    record50.send_time_ms = 1500;
    record50.received = false;

    addPendingPacket(&record50);

    // Add packet 150 (same slot: 150 % 100 = 50, overwrites packet 50)
    RTTRecord record150;
    record150.sequence = 150;
    record150.send_time_ms = 2500;
    record150.received = false;

    addPendingPacket(&record150);

    // Try to mark packet 50 as received (but slot now contains 150)
    // This should NOT work - sequence mismatch
    markPacketReceived(50, 1510);

    // Verify packet 150 is still pending (not marked received)
    TEST_ASSERT_FALSE(pendingPackets[50].received);
    TEST_ASSERT_EQUAL_UINT32(150, pendingPackets[50].sequence);
}

/**
 * TEST 12: Buffer stress test - rapid adds
 */
void test_buffer_stress_rapid_adds() {
    // Add 250 packets rapidly (2.5x buffer size)
    for (uint32_t i = 0; i < 250; i++) {
        RTTRecord record;
        record.sequence = i;
        record.send_time_ms = 1000 + i;
        record.received = false;

        addPendingPacket(&record);
    }

    // Verify final buffer state after wraparound
    // For modulo-based indexing (seq % 100):
    // - Index 0 receives: 0 -> 100 -> 200 (final: 200)
    // - Index 49 receives: 49 -> 149 -> 249 (final: 249)
    // - Index 50 receives: 50 -> 150 (final: 150, never overwritten)
    // - Index 99 receives: 99 -> 199 (final: 199, never overwritten)
    for (int i = 0; i < MAX_PENDING; i++) {
        uint32_t expected_seq;
        if (i < 50) {
            // Slots 0-49: sequences 200-249 (overwritten twice)
            expected_seq = 200 + i;
        } else {
            // Slots 50-99: sequences 150-199 (overwritten once)
            expected_seq = 150 + (i - 50);
        }
        TEST_ASSERT_EQUAL_UINT32(expected_seq, pendingPackets[i].sequence);
    }
}

// Include RTT buffer header (currently empty - will cause tests to FAIL)
#include "../espnow_rtt/RTTBuffer.h"

void setup() {
    delay(2000);

    UNITY_BEGIN();

    RUN_TEST(test_buffer_initialization);
    RUN_TEST(test_buffer_index_calculation);
    RUN_TEST(test_add_pending_packet);
    RUN_TEST(test_mark_packet_received);
    RUN_TEST(test_buffer_wraparound);
    RUN_TEST(test_is_packet_pending);
    RUN_TEST(test_is_packet_received);
    RUN_TEST(test_multiple_packets);
    RUN_TEST(test_buffer_full);
    RUN_TEST(test_collision_overwrites_old_packet);
    RUN_TEST(test_received_packet_sequence_verification);
    RUN_TEST(test_buffer_stress_rapid_adds);

    UNITY_END();
}

void loop() {
}
