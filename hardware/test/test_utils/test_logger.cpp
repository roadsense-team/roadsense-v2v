#include <unity.h>
#include <Arduino.h>
#include "utils/Logger.h"

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

void test_logger_singleton() {
    Logger& log1 = Logger::getInstance();
    Logger& log2 = Logger::getInstance();
    
    // Check if addresses are the same
    TEST_ASSERT_EQUAL_PTR(&log1, &log2);
}

void test_logger_initialization() {
    Logger& log = Logger::getInstance();
    
    // Should not crash
    log.begin(115200);
    log.info("TEST", "Logger initialized test message");
}

void test_log_levels() {
    Logger& log = Logger::getInstance();
    
    // Just verify these calls don't crash the system
    log.setLogLevel(LOG_DEBUG);
    log.debug("TEST", "This is a debug message");
    
    log.setLogLevel(LOG_ERROR);
    log.debug("TEST", "This debug message should be suppressed");
    log.error("TEST", "This is an error message");
}

void setup() {
    // Wait for serial port to be available for test output
    delay(2000); 
    
    UNITY_BEGIN();
    
    RUN_TEST(test_logger_singleton);
    RUN_TEST(test_logger_initialization);
    RUN_TEST(test_log_levels);
    
    UNITY_END();
}

void loop() {
    // Tests run once in setup()
}
