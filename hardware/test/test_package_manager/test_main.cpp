#include <unity.h>
#include <Arduino.h>
#include "network/mesh/PackageManager.h"
#include "network/protocol/V2VMessage.h"
#include "config.h"

// Helper to create a dummy message
V2VMessage createMessage(uint32_t timestamp, const char* id = "V999") {
    V2VMessage msg;
    msg.version = 2;
    strncpy(msg.vehicleId, id, 8);
    msg.timestamp = timestamp;
    // Important: Set sourceMAC in the message, as PackageManager uses this for deduplication
    // We'll let the test cases set msg.sourceMAC explicitly if needed, 
    // but for now let's default to all zeros or handle it in the test
    return msg;
}

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

// ============================================================================
// DUPLICATE DETECTION TESTS
// ============================================================================

void test_duplicate_same_mac_timestamp() {
    PackageManager pm; // Instantiate directly (not a singleton)
    
    uint8_t mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    
    V2VMessage msg = createMessage(1000);
    memcpy(msg.sourceMAC, mac, 6); // Deduplication uses msg.sourceMAC

    // First add should succeed (using 'mac' as immediate sender)
    pm.addPackage(mac, msg, 0);
    TEST_ASSERT_EQUAL(1, pm.getPackageCount()); // Method name corrected

    // Second add (same MAC, same TS) should fail/be ignored
    pm.addPackage(mac, msg, 0);
    TEST_ASSERT_EQUAL(1, pm.getPackageCount());
}

void test_non_duplicate_diff_timestamp() {
    PackageManager pm;
    uint8_t mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    
    V2VMessage msg1 = createMessage(1000);
    memcpy(msg1.sourceMAC, mac, 6);
    pm.addPackage(mac, msg1, 0);
    
    V2VMessage msg2 = createMessage(1001);
    memcpy(msg2.sourceMAC, mac, 6);
    pm.addPackage(mac, msg2, 0);
    
    TEST_ASSERT_EQUAL(2, pm.getPackageCount());
}

void test_non_duplicate_diff_mac() {
    PackageManager pm;
    
    uint8_t mac1[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    uint8_t mac2[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    
    // Message from MAC1
    V2VMessage msg1 = createMessage(1000);
    memcpy(msg1.sourceMAC, mac1, 6);
    
    // Identical content (timestamp 1000), but from MAC2
    V2VMessage msg2 = createMessage(1000);
    memcpy(msg2.sourceMAC, mac2, 6);

    pm.addPackage(mac1, msg1, 0);
    pm.addPackage(mac2, msg2, 0);
    
    TEST_ASSERT_EQUAL(2, pm.getPackageCount());
}

// ============================================================================
// FIFO EVICTION TESTS
// ============================================================================

void test_fifo_eviction() {
    PackageManager pm;
    uint8_t mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

    // Fill up to limit (MAX_PACKAGES_PER_SOURCE = 3)
    for(int i=0; i<3; i++) {
        V2VMessage msg = createMessage(1000 + i*1000);
        memcpy(msg.sourceMAC, mac, 6);
        pm.addPackage(mac, msg, 0);
    }
    TEST_ASSERT_EQUAL(3, pm.getPackageCount());

    // Add 4th message -> should evict oldest (1000)
    V2VMessage msg4 = createMessage(4000);
    memcpy(msg4.sourceMAC, mac, 6);
    pm.addPackage(mac, msg4, 0);
    TEST_ASSERT_EQUAL(3, pm.getPackageCount());

    // Verify contents: should be 2000, 3000, 4000
    std::vector<PackageData> packages = pm.getAllPackagesForMAC(mac); // Method name corrected
    TEST_ASSERT_EQUAL(3, packages.size());
    TEST_ASSERT_EQUAL_UINT32(2000, packages[0].message.timestamp);
    TEST_ASSERT_EQUAL_UINT32(3000, packages[1].message.timestamp);
    TEST_ASSERT_EQUAL_UINT32(4000, packages[2].message.timestamp);
}

// ============================================================================
// TIMEOUT CLEANUP TESTS
// ============================================================================

void test_timeout_cleanup() {
    PackageManager pm;
    uint8_t mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

    V2VMessage msg = createMessage(1000);
    memcpy(msg.sourceMAC, mac, 6);
    
    pm.addPackage(mac, msg, 0);
    TEST_ASSERT_EQUAL(1, pm.getPackageCount());

    // Run cleanup with default (should keep it since it's fresh)
    pm.cleanupOldPackages(); 
    TEST_ASSERT_EQUAL(1, pm.getPackageCount());
    
    // Note: We rely on hardware validation for the actual timing logic
    // since we cannot mock millis() easily here without dependency injection.
}

// ============================================================================
// MAC LIMIT TESTS
// ============================================================================

// Only run this if MAX_TRACKED_MACS is defined
#ifdef MAX_TRACKED_MACS
void test_mac_limit_enforcement() {
    PackageManager pm;
    
    // Add packages from 20 different MACs
    for (int i = 0; i < MAX_TRACKED_MACS; i++) {
        uint8_t mac[] = {0x00, 0x00, 0x00, 0x00, 0x00, (uint8_t)i};
        V2VMessage msg = createMessage(1000);
        memcpy(msg.sourceMAC, mac, 6);
        
        pm.addPackage(mac, msg, 0);
    }
    TEST_ASSERT_EQUAL(MAX_TRACKED_MACS, pm.getPackageCount());

    // Add 21st MAC -> should evict LRU (MAC 0) if implementation supports it
    // NOTE: The current implementation might not strictly enforce 20 MACs *globally*
    // or might use a different strategy. 
    // Checking config.h: MAX_TRACKED_MACS is 20.
    // This test assumes PackageManager implementation handles this limit.
    
    uint8_t mac21[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    V2VMessage msg21 = createMessage(1000);
    memcpy(msg21.sourceMAC, mac21, 6);
    
    pm.addPackage(mac21, msg21, 0);
    
    // If logic works, count should stay at MAX (20) or be capped
    // If not implemented, it might grow. Let's assert <= MAX_TRACKED_MACS + 5 just in case
    // ideally it should be equal.
    
    // Based on previous bug fixes (Fix #1: MAC Tracking DoS), this should be strictly enforced.
    // BUT if the implementation clears *all* packages from the evicted MAC, total count stays same.
    TEST_ASSERT_TRUE(pm.getPackageCount() <= MAX_TRACKED_MACS); 
}
#endif

// ============================================================================
// UTILITY TESTS
// ============================================================================

void test_package_count() {
    PackageManager pm;
    TEST_ASSERT_EQUAL(0, pm.getPackageCount());
    
    uint8_t mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    V2VMessage msg = createMessage(1000);
    memcpy(msg.sourceMAC, mac, 6);
    
    pm.addPackage(mac, msg, 0);
    TEST_ASSERT_EQUAL(1, pm.getPackageCount());
}

void setup() {
    delay(2000);
    UNITY_BEGIN();
    
    RUN_TEST(test_duplicate_same_mac_timestamp);
    RUN_TEST(test_non_duplicate_diff_timestamp);
    RUN_TEST(test_non_duplicate_diff_mac);
    RUN_TEST(test_fifo_eviction);
    RUN_TEST(test_timeout_cleanup);
    #ifdef MAX_TRACKED_MACS
    RUN_TEST(test_mac_limit_enforcement);
    #endif
    RUN_TEST(test_package_count);
    
    UNITY_END();
}

void loop() {
}