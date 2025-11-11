/**
 * @file main.cpp
 * @brief RoadSense V2V - Main firmware entry point
 *
 * ESP32 firmware for Vehicle-to-Vehicle hazard detection system
 *
 * Phase 1: Foundation & Structure (CURRENT)
 * - Directory structure created
 * - Interfaces defined
 * - Utilities ported
 * - This skeleton compiles successfully
 *
 * Next phases will implement:
 * - Phase 2: Network layer (ESP-NOW mesh)
 * - Phase 3: Sensor layer (MPU9250, GPS)
 * - Phase 4: ML integration (TensorFlow Lite)
 * - Phase 5: Core application logic
 * - Phase 6: Testing and validation
 */

#include <Arduino.h>
#include "config.h"
#include "utils/Logger.h"

// Include interfaces (compile-time check)
#include "../include/IImuSensor.h"
#include "../include/IGpsSensor.h"
#include "../include/ITransport.h"
#include "../include/VehicleState.h"
#include "network/protocol/V2VMessage.h"

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

Logger& logger = Logger::getInstance();

// LED blink state (for visual feedback during development)
unsigned long lastBlinkTime = 0;
bool ledState = false;

// ============================================================================
// SETUP - Called once at boot
// ============================================================================

void setup() {
    // Initialize serial logging
    logger.begin(SERIAL_BAUD_RATE);
    logger.info("MAIN", "========================================");
    logger.info("MAIN", "  RoadSense V2V System - Phase 1      ");
    logger.info("MAIN", "  Vehicle ID: " + String(VEHICLE_ID));
    logger.info("MAIN", "========================================");

    // Initialize status LED
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN, LOW);
    logger.info("MAIN", "Status LED initialized (GPIO " + String(LED_STATUS_PIN) + ")");

    // Test V2VMessage size
    logger.info("MAIN", "V2VMessage size: " + String(sizeof(V2VMessage)) + " bytes");
    if (sizeof(V2VMessage) <= 250) {
        logger.info("MAIN", "✓ V2VMessage fits in ESP-NOW payload limit (250 bytes)");
    } else {
        logger.error("MAIN", "✗ V2VMessage exceeds ESP-NOW limit!");
    }

    // Test VehicleState
    VehicleState state;
    strncpy(state.vehicleId, VEHICLE_ID, sizeof(state.vehicleId));
    state.timestamp = millis();
    logger.info("MAIN", "VehicleState initialized for " + String(state.vehicleId));

    // Initialization complete
    logger.info("MAIN", "");
    logger.info("MAIN", "Phase 1 Foundation - COMPLETE ✓");
    logger.info("MAIN", "Entering main loop...");
    logger.info("MAIN", "");
}

// ============================================================================
// LOOP - Called repeatedly
// ============================================================================

void loop() {
    // Blink LED to show firmware is running
    unsigned long currentTime = millis();
    if (currentTime - lastBlinkTime >= 1000) {  // 1 Hz blink
        lastBlinkTime = currentTime;
        ledState = !ledState;
        digitalWrite(LED_STATUS_PIN, ledState ? HIGH : LOW);

        // Log heartbeat every 5 seconds
        static int loopCount = 0;
        loopCount++;
        if (loopCount % 5 == 0) {
            logger.debug("MAIN", "Heartbeat - System running OK");
        }
    }

    // Phase 1: Nothing else to do yet
    // Future phases will add:
    // - Sensor reading
    // - ML inference
    // - Network communication
    // - Hazard detection

    delay(10);  // Small delay to prevent watchdog issues
}
