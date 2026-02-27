#include <unity.h>

#include <cstdint>
#include <cstring>

#ifdef ARDUINO
#include <Arduino.h>
#endif

#include "inference/ConeFilter.h"
#include "network/mesh/MeshRelayPolicy.h"
#include "network/protocol/V2VMessage.h"
#include "utils/MACHelper.h"

namespace {

struct TestMessage {
    uint8_t hopCount;
    uint8_t sourceMAC[6];
    uint32_t timestamp;
};

// Helper: build a V2VMessage with specific fields for testing
V2VMessage makeV2VMessage(const char* vehicleId,
                          uint32_t timestamp,
                          float lat, float lon,
                          float heading, float speed,
                          uint8_t hopCount,
                          const uint8_t sourceMac[6]) {
    V2VMessage msg;
    strncpy(msg.vehicleId, vehicleId, 8);
    msg.timestamp = timestamp;
    msg.position.lat = lat;
    msg.position.lon = lon;
    msg.dynamics.heading = heading;
    msg.dynamics.speed = speed;
    msg.hopCount = hopCount;
    std::memcpy(msg.sourceMAC, sourceMac, 6);
    return msg;
}

void test_cone_filter_cpp_peer_ahead_returns_true() {
    const bool inCone = ConeFilter::isInCone(
        0.0f,      // heading north
        32.0000,   // ego lat
        34.0000,   // ego lon
        32.0010,   // peer lat slightly north
        34.0000,   // peer lon same
        45.0f
    );
    TEST_ASSERT_TRUE(inCone);
}

void test_cone_filter_cpp_peer_behind_returns_false() {
    const bool inCone = ConeFilter::isInCone(
        0.0f,
        32.0000,
        34.0000,
        31.9990,   // south
        34.0000,
        45.0f
    );
    TEST_ASSERT_FALSE(inCone);
}

void test_cone_filter_cpp_matches_python_implementation() {
    const bool edgeIncluded = ConeFilter::isInCone(0.0f, 0.0, 0.0, 1.0, 1.0, 45.0f);
    const bool outsideExcluded = ConeFilter::isInCone(0.0f, 0.0, 0.0, 1.0, 1.1, 45.0f);
    const bool wrapAroundIncluded = ConeFilter::isInCone(359.0f, 0.0, 0.0, 1.0, -0.01, 45.0f);

    TEST_ASSERT_TRUE(edgeIncluded);
    TEST_ASSERT_FALSE(outsideExcluded);
    TEST_ASSERT_TRUE(wrapAroundIncluded);
}

void test_rebroadcast_increments_hop_count() {
    TEST_ASSERT_EQUAL_UINT8(1, MeshRelayPolicy::computeRelayedHopCount(0, 3));
    TEST_ASSERT_EQUAL_UINT8(2, MeshRelayPolicy::computeRelayedHopCount(1, 3));
}

void test_rebroadcast_only_front_cone_peers() {
    TEST_ASSERT_TRUE(MeshRelayPolicy::shouldRelayMessage(0, false, true, 3));
    TEST_ASSERT_FALSE(MeshRelayPolicy::shouldRelayMessage(0, false, false, 3));
}

void test_rebroadcast_blocks_self_and_max_hop() {
    TEST_ASSERT_FALSE(MeshRelayPolicy::shouldRelayMessage(0, true, true, 3));
    TEST_ASSERT_FALSE(MeshRelayPolicy::shouldRelayMessage(3, false, true, 3));
}

void test_package_manager_called_on_receive() {
    const uint8_t senderMac[6] = {1, 2, 3, 4, 5, 6};

    TestMessage msg{};
    msg.hopCount = 1;
    msg.timestamp = 1234;
    std::memcpy(msg.sourceMAC, senderMac, sizeof(senderMac));

    bool addCalled = false;
    uint8_t observedHop = 0;

    const bool parsed = MeshRelayPolicy::parseAndStoreReceivedMessage<TestMessage>(
        senderMac,
        reinterpret_cast<const uint8_t*>(&msg),
        sizeof(msg),
        [&](const uint8_t* mac, const TestMessage& parsedMsg, uint8_t hopCount) {
            addCalled = true;
            observedHop = hopCount;
            TEST_ASSERT_EQUAL_UINT8_ARRAY(senderMac, mac, 6);
            TEST_ASSERT_EQUAL_UINT8(1, parsedMsg.hopCount);
        }
    );

    TEST_ASSERT_TRUE(parsed);
    TEST_ASSERT_TRUE(addCalled);
    TEST_ASSERT_EQUAL_UINT8(1, observedHop);
}

void test_invalid_payload_not_stored() {
    const uint8_t senderMac[6] = {1, 2, 3, 4, 5, 6};
    const uint8_t badPayload[3] = {0, 1, 2};
    bool addCalled = false;

    const bool parsed = MeshRelayPolicy::parseAndStoreReceivedMessage<TestMessage>(
        senderMac,
        badPayload,
        sizeof(badPayload),
        [&](const uint8_t*, const TestMessage&, uint8_t) { addCalled = true; }
    );

    TEST_ASSERT_FALSE(parsed);
    TEST_ASSERT_FALSE(addCalled);
}

// =========================================================================
// GAP 1: Hop count saturation at MAX_HOP_COUNT
// =========================================================================

void test_hop_count_saturates_at_max() {
    // At max (3) -> should return max, not overflow
    TEST_ASSERT_EQUAL_UINT8(3, MeshRelayPolicy::computeRelayedHopCount(3, 3));
    // Above max (shouldn't happen, but defensive)
    TEST_ASSERT_EQUAL_UINT8(3, MeshRelayPolicy::computeRelayedHopCount(4, 3));
    // Max = 1 edge case
    TEST_ASSERT_EQUAL_UINT8(1, MeshRelayPolicy::computeRelayedHopCount(0, 1));
    TEST_ASSERT_EQUAL_UINT8(1, MeshRelayPolicy::computeRelayedHopCount(1, 1));
}

// =========================================================================
// GAP 2: Cone filter at Israel's latitude (~32N)
//   cos(32°) ≈ 0.848, so longitude is compressed ~15%.
//   A peer at exactly 45° bearing in flat coords will appear
//   at a DIFFERENT bearing after projection. Verify the filter
//   handles this correctly.
// =========================================================================

void test_cone_filter_israel_latitude_peer_northeast() {
    // Ego in Tel Aviv area heading North.
    // Peer offset: +0.001 lat (north), +0.001 lon (east).
    // Flat bearing = 45°, but after cos(32°) projection:
    //   east_proj = 0.001 * cos(32°) ≈ 0.000848
    //   north     = 0.001
    //   bearing   = atan2(0.000848, 0.001) ≈ 40.3°
    // 40.3° < 45° → should be IN cone.
    const bool inCone = ConeFilter::isInCone(
        0.0f, 32.0000, 34.8000, 32.0010, 34.8010, 45.0f);
    TEST_ASSERT_TRUE(inCone);
}

void test_cone_filter_israel_latitude_peer_just_outside() {
    // At 32°N, a peer with dLon much larger than dLat gets
    // pushed to a larger bearing by the cos(lat) compression.
    // dLat=+0.0004, dLon=+0.001
    //   east_proj = 0.001 * cos(32°) ≈ 0.000848
    //   north     = 0.0004
    //   bearing   = atan2(0.000848, 0.0004) ≈ 64.8°
    // 64.8° > 45° → should be OUT of cone.
    const bool inCone = ConeFilter::isInCone(
        0.0f, 32.0000, 34.8000, 32.0004, 34.8010, 45.0f);
    TEST_ASSERT_FALSE(inCone);
}

// =========================================================================
// GAP 3: parseAndStoreReceivedMessage with real V2VMessage struct
//   Verifies the actual 90-byte struct is parsed correctly off the wire.
// =========================================================================

void test_parse_real_v2v_message_preserves_fields() {
    const uint8_t senderMac[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01};
    const uint8_t sourceMac[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

    V2VMessage original = makeV2VMessage(
        "V002", 99000,
        32.0800f, 34.7800f,
        90.0f, 15.5f,
        1, sourceMac);

    // Simulate wire: raw bytes
    const uint8_t* wire = reinterpret_cast<const uint8_t*>(&original);

    V2VMessage parsed;
    bool addCalled = false;

    const bool ok = MeshRelayPolicy::parseAndStoreReceivedMessage<V2VMessage>(
        senderMac, wire, sizeof(V2VMessage),
        [&](const uint8_t* mac, const V2VMessage& msg, uint8_t hop) {
            addCalled = true;

            // hopCount comes from msg.hopCount (the parsed message)
            TEST_ASSERT_EQUAL_UINT8(1, hop);

            // sourceMAC should be the ORIGINAL sender, not senderMac
            TEST_ASSERT_EQUAL_UINT8_ARRAY(sourceMac, msg.sourceMAC, 6);

            // Key fields survived the memcpy
            TEST_ASSERT_EQUAL_STRING("V002", msg.vehicleId);
            TEST_ASSERT_EQUAL_UINT32(99000, msg.timestamp);
            TEST_ASSERT_FLOAT_WITHIN(0.001f, 32.08f, msg.position.lat);
            TEST_ASSERT_FLOAT_WITHIN(0.001f, 34.78f, msg.position.lon);
            TEST_ASSERT_FLOAT_WITHIN(0.1f, 90.0f, msg.dynamics.heading);
            TEST_ASSERT_FLOAT_WITHIN(0.1f, 15.5f, msg.dynamics.speed);
        },
        &parsed
    );

    TEST_ASSERT_TRUE(ok);
    TEST_ASSERT_TRUE(addCalled);

    // The out-param should also have correct fields
    TEST_ASSERT_EQUAL_STRING("V002", parsed.vehicleId);
    TEST_ASSERT_EQUAL_UINT8(1, parsed.hopCount);
}

void test_parse_v2v_message_wrong_size_rejected() {
    const uint8_t senderMac[6] = {1, 2, 3, 4, 5, 6};
    // 89 bytes — one byte short of sizeof(V2VMessage)
    uint8_t shortPayload[89];
    std::memset(shortPayload, 0, sizeof(shortPayload));
    bool addCalled = false;

    const bool ok = MeshRelayPolicy::parseAndStoreReceivedMessage<V2VMessage>(
        senderMac, shortPayload, sizeof(shortPayload),
        [&](const uint8_t*, const V2VMessage&, uint8_t) { addCalled = true; }
    );

    TEST_ASSERT_FALSE(ok);
    TEST_ASSERT_FALSE(addCalled);
}

// =========================================================================
// GAP 4: E2E relay chain — A → B → C
//   Three vehicles in a convoy heading North.
//   A is furthest ahead, B is in the middle, C is ego (rear).
//   Verifies that:
//   - A's message is in B's front cone → B should relay
//   - B increments hop count from 0 → 1
//   - C receives the relayed message with hop=1, sourceMAC=A
//   - C does NOT relay A's message further (A is behind C? No — A is
//     AHEAD of C too, so C would relay if maxHop allows. Test both.)
// =========================================================================

void test_e2e_relay_chain_a_through_b_to_c() {
    // MAC addresses
    const uint8_t macA[6] = {0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5};
    const uint8_t macB[6] = {0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5};
    const uint8_t macC[6] = {0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5};

    // All heading North (0°). Positions along a north-south line.
    // A is furthest north (ahead), C is furthest south (ego, rear).
    //   A: lat=32.003  (front)
    //   B: lat=32.002  (middle)
    //   C: lat=32.001  (rear / ego)
    const float lon = 34.800f;

    V2VMessage msgA = makeV2VMessage("V003", 1000, 32.003f, lon, 0.0f, 20.0f, 0, macA);
    V2VMessage msgB = makeV2VMessage("V002", 1000, 32.002f, lon, 0.0f, 20.0f, 0, macB);
    V2VMessage msgC = makeV2VMessage("V001", 1000, 32.001f, lon, 0.0f, 20.0f, 0, macC);

    // --- Step 1: B receives A's broadcast (hop=0). ---
    // B checks: is A in B's front cone?
    const bool aInBCone = ConeFilter::isInCone(
        msgB.dynamics.heading,
        msgB.position.lat, msgB.position.lon,
        msgA.position.lat, msgA.position.lon);
    TEST_ASSERT_TRUE_MESSAGE(aInBCone, "A should be in B's front cone (A is ahead of B)");

    // B decides to relay
    const bool bShouldRelay = MeshRelayPolicy::shouldRelayMessage(
        msgA.hopCount,   // 0
        false,           // A is not B
        aInBCone,        // true
        3);              // maxHop=3
    TEST_ASSERT_TRUE_MESSAGE(bShouldRelay, "B should relay A's message");

    // B prepares relayed message
    V2VMessage relayedByB = msgA;  // copy A's message
    relayedByB.hopCount = MeshRelayPolicy::computeRelayedHopCount(msgA.hopCount, 3);
    TEST_ASSERT_EQUAL_UINT8(1, relayedByB.hopCount);

    // sourceMAC must still be A's (NOT B's)
    TEST_ASSERT_EQUAL_UINT8_ARRAY(macA, relayedByB.sourceMAC, 6);

    // --- Step 2: C (ego) receives the relayed message from B (hop=1). ---
    // C checks: is the ORIGINAL source (A) in C's front cone?
    const bool aInCCone = ConeFilter::isInCone(
        msgC.dynamics.heading,
        msgC.position.lat, msgC.position.lon,
        relayedByB.position.lat, relayedByB.position.lon);
    TEST_ASSERT_TRUE_MESSAGE(aInCCone, "A should be in C's front cone (A is ahead of C)");

    // C could relay further (A is in front, hop=1 < maxHop=3)
    const bool sourceIsSelfC = MACHelper::compareMACAddresses(relayedByB.sourceMAC, macC);
    TEST_ASSERT_FALSE_MESSAGE(sourceIsSelfC, "A's MAC should not match C's MAC");

    const bool cShouldRelay = MeshRelayPolicy::shouldRelayMessage(
        relayedByB.hopCount, sourceIsSelfC, aInCCone, 3);
    TEST_ASSERT_TRUE_MESSAGE(cShouldRelay, "C should relay A's message onward (hop=1 < 3)");

    // C increments hop
    uint8_t cRelayHop = MeshRelayPolicy::computeRelayedHopCount(relayedByB.hopCount, 3);
    TEST_ASSERT_EQUAL_UINT8(2, cRelayHop);
}

void test_e2e_relay_blocked_when_source_behind() {
    // Same convoy but D is BEHIND B. B should NOT relay D's message.
    const uint8_t macB[6] = {0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5};
    const uint8_t macD[6] = {0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5};

    const float lon = 34.800f;
    V2VMessage msgB = makeV2VMessage("V002", 1000, 32.002f, lon, 0.0f, 20.0f, 0, macB);
    V2VMessage msgD = makeV2VMessage("V004", 1000, 32.000f, lon, 0.0f, 20.0f, 0, macD);

    // D is south of B → behind B (heading North)
    const bool dInBCone = ConeFilter::isInCone(
        msgB.dynamics.heading,
        msgB.position.lat, msgB.position.lon,
        msgD.position.lat, msgD.position.lon);
    TEST_ASSERT_FALSE_MESSAGE(dInBCone, "D should NOT be in B's front cone (D is behind B)");

    const bool bShouldRelay = MeshRelayPolicy::shouldRelayMessage(
        msgD.hopCount, false, dInBCone, 3);
    TEST_ASSERT_FALSE_MESSAGE(bShouldRelay, "B should NOT relay D's message");
}

void test_e2e_relay_blocked_at_max_hop() {
    // A's message has already been relayed twice (hop=2). B receives it.
    // With maxHop=2, B should NOT relay further.
    const uint8_t macA[6] = {0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5};

    const bool shouldRelay = MeshRelayPolicy::shouldRelayMessage(
        2,      // already at hop 2
        false,  // not self
        true,   // in front cone
        2);     // maxHop = 2
    TEST_ASSERT_FALSE_MESSAGE(shouldRelay, "Should NOT relay at max hop count");

    // Verify hop count saturates
    TEST_ASSERT_EQUAL_UINT8(2, MeshRelayPolicy::computeRelayedHopCount(2, 2));
}

void test_e2e_self_message_not_relayed() {
    // Vehicle B receives its OWN message echoed back (same sourceMAC).
    const uint8_t macB[6] = {0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5};

    const bool sourceIsSelf = MACHelper::compareMACAddresses(macB, macB);
    TEST_ASSERT_TRUE(sourceIsSelf);

    const bool shouldRelay = MeshRelayPolicy::shouldRelayMessage(
        0, sourceIsSelf, true, 3);
    TEST_ASSERT_FALSE_MESSAGE(shouldRelay, "Should never relay own message");
}

// =========================================================================
// GAP 5: normalizeAngleDeg edge cases
// =========================================================================

void test_cone_filter_normalize_boundary_180() {
    // Peer exactly behind (180° from heading).
    // Heading=0, peer due south. diff should be ±180 → outside 45° cone.
    const bool inCone = ConeFilter::isInCone(
        0.0f, 32.0000, 34.8000, 31.9990, 34.8000, 45.0f);
    TEST_ASSERT_FALSE(inCone);
}

void test_cone_filter_normalize_boundary_negative_heading() {
    // Heading=1°, peer at bearing 359° (2° to the left).
    // diff should be -2° → abs(2) <= 45 → in cone.
    // Peer slightly west of north from ego, heading almost north.
    // At lat 32, a small negative dLon with large positive dLat → bearing ~359°.
    const bool inCone = ConeFilter::isInCone(
        1.0f, 32.0000, 34.8000, 32.0010, 34.7999, 45.0f);
    TEST_ASSERT_TRUE(inCone);
}

} // namespace

void setUp(void) {}
void tearDown(void) {}

static void run_all_tests() {
    // --- Original tests ---
    RUN_TEST(test_cone_filter_cpp_peer_ahead_returns_true);
    RUN_TEST(test_cone_filter_cpp_peer_behind_returns_false);
    RUN_TEST(test_cone_filter_cpp_matches_python_implementation);
    RUN_TEST(test_rebroadcast_increments_hop_count);
    RUN_TEST(test_rebroadcast_only_front_cone_peers);
    RUN_TEST(test_rebroadcast_blocks_self_and_max_hop);
    RUN_TEST(test_package_manager_called_on_receive);
    RUN_TEST(test_invalid_payload_not_stored);

    // --- Gap 1: Hop count saturation ---
    RUN_TEST(test_hop_count_saturates_at_max);

    // --- Gap 2: Cone filter at Israel latitude ---
    RUN_TEST(test_cone_filter_israel_latitude_peer_northeast);
    RUN_TEST(test_cone_filter_israel_latitude_peer_just_outside);

    // --- Gap 3: Real V2VMessage wire parsing ---
    RUN_TEST(test_parse_real_v2v_message_preserves_fields);
    RUN_TEST(test_parse_v2v_message_wrong_size_rejected);

    // --- Gap 4: E2E relay chain ---
    RUN_TEST(test_e2e_relay_chain_a_through_b_to_c);
    RUN_TEST(test_e2e_relay_blocked_when_source_behind);
    RUN_TEST(test_e2e_relay_blocked_at_max_hop);
    RUN_TEST(test_e2e_self_message_not_relayed);

    // --- Gap 5: normalizeAngleDeg boundaries ---
    RUN_TEST(test_cone_filter_normalize_boundary_180);
    RUN_TEST(test_cone_filter_normalize_boundary_negative_heading);
}

#ifdef ARDUINO
void setup() {
    delay(2000);
    UNITY_BEGIN();
    run_all_tests();
    UNITY_END();
}

void loop() {}
#else
int main(int, char**) {
    UNITY_BEGIN();
    run_all_tests();
    return UNITY_END();
}
#endif
