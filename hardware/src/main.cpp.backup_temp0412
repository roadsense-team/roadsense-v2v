/**
 * @file main.cpp
 * @brief RoadSense V2V Unified Firmware
 *
 * Integrates:
 * - MPU6500 (6-axis IMU)
 * - NEO-6M (GPS)
 * - ESP-NOW (V2V Communication)
 *
 * Loop Rate: ~100Hz (Sensors), 10Hz (Broadcast)
 */

#ifndef UNIT_TEST // Only compile main loop if NOT running unit tests

#include <Arduino.h>
#include "config.h"
#include "utils/Logger.h"
#include "sensors/imu/MPU6500Driver.h"
#include "sensors/gps/NEO6M_Driver.h"
#include "network/transport/EspNowTransport.h"
#include "network/protocol/V2VMessage.h"
#include "logging/DataLogger.h"

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================
MPU6500Driver imu;
NEO6M_Driver gps;
EspNowTransport transport;
DataLogger dataLogger;

// Timer variables
uint32_t lastBroadcastTime = 0;
uint32_t lastPrintTime = 0;
uint32_t lastButtonCheck = 0;

// Button state
bool lastButtonState = HIGH;
uint32_t lastButtonPress = 0;
const uint32_t BUTTON_DEBOUNCE_MS = 500;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Update LED based on system state
 */
void updateLED() {
    static uint32_t lastLedToggle = 0;
    static bool ledState = false;
    uint32_t now = millis();

    auto gpsData = gps.read();

    if (dataLogger.isLogging()) {
        // Solid ON when logging
        digitalWrite(LED_STATUS_PIN, HIGH);
    } else if (!gpsData.valid && !gpsData.cached) {
        // Fast blink when waiting for GPS fix (200ms)
        if (now - lastLedToggle >= 200) {
            ledState = !ledState;
            digitalWrite(LED_STATUS_PIN, ledState);
            lastLedToggle = now;
        }
    } else {
        // OFF when not logging and GPS ready
        digitalWrite(LED_STATUS_PIN, LOW);
    }
}

/**
 * @brief Handle button press (start/stop logging)
 */
void handleButton() {
    uint32_t now = millis();
    bool buttonState = digitalRead(BUTTON_CALIB_PIN);

    // Debounce check
    if (buttonState == LOW && lastButtonState == HIGH) {
        if (now - lastButtonPress > BUTTON_DEBOUNCE_MS) {
            lastButtonPress = now;

            // Toggle logging
            auto gpsData = gps.read();
            if (dataLogger.isLogging()) {
                // Stop logging
                dataLogger.stopLogging();
                Logger::getInstance().info("MAIN", "📝 Logging STOPPED");
            } else {
                // Start logging
                bool gpsReady = gpsData.valid || gpsData.cached;
                if (dataLogger.startLogging(VEHICLE_ID, gpsReady)) {
                    Logger::getInstance().info("MAIN", "📝 Logging STARTED (Session " + String(dataLogger.getSessionNumber()) + ")");
                } else {
                    Logger::getInstance().error("MAIN", "❌ Failed to start logging");
                }
            }
        }
    }

    lastButtonState = buttonState;
}

/**
 * @brief Populate V2VMessage with current sensor data
 */
V2VMessage buildV2VMessage() {
    V2VMessage msg;
    
    // Header
    msg.version = 2;
    strncpy(msg.vehicleId, VEHICLE_ID, 8);
    msg.timestamp = millis();
    
    // Position (GPS)
    auto gpsData = gps.read();
    msg.position.lat = gpsData.latitude;
    msg.position.lon = gpsData.longitude;
    msg.position.alt = gpsData.altitude;
    
    // Dynamics (GPS + IMU)
    msg.dynamics.speed = gpsData.speed;
    msg.dynamics.heading = gpsData.heading;
    
    auto imuData = imu.read();
    // Convert from IMU frame to vehicle frame if needed
    // For now, assume sensor is mounted aligned with vehicle
    msg.dynamics.longAccel = imuData.accel[0]; // X-axis = Longitudinal
    msg.dynamics.latAccel = imuData.accel[1];  // Y-axis = Lateral
    
    // Raw Sensor Data (RoadSense Extension)
    memcpy(msg.sensors.accel, imuData.accel, sizeof(float) * 3);
    memcpy(msg.sensors.gyro, imuData.gyro, sizeof(float) * 3);
    memset(msg.sensors.mag, 0, sizeof(float) * 3); // No magnetometer
    
    // Hazard Alert (Placeholder / Future ML)
    msg.alert.riskLevel = 0; // None
    msg.alert.scenarioType = 0;
    msg.alert.confidence = 0.0f;
    
    // Mesh Metadata
    msg.hopCount = 0;
    WiFi.macAddress(msg.sourceMAC);
    
    return msg;
}

/**
 * @brief Callback for received messages
 */
void onDataReceived(const uint8_t* data, size_t len) {
    if (len != sizeof(V2VMessage)) {
        Logger::getInstance().warning("V2V", "Received invalid size: " + String(len));
        return;
    }
    
    const V2VMessage* msg = (const V2VMessage*)data;
    
    // Simple log for now
    String logMsg = "Rx from " + String(msg->vehicleId) + 
                    " | Speed: " + String(msg->dynamics.speed) + 
                    " | Lat: " + String(msg->position.lat, 6);
    Logger::getInstance().info("V2V", logMsg);
}

// ============================================================================
// MAIN SETUP
// ============================================================================
void setup() {
    // 1. Initialize Logger
    Logger& log = Logger::getInstance();
    log.begin(115200);
    log.setLogLevel(LOG_INFO);

    delay(1000); // Stability delay

    log.info("MAIN", "═══════════════════════════════════");
    log.info("MAIN", "  RoadSense V2V Unified Firmware");
    log.info("MAIN", "  Vehicle ID: " + String(VEHICLE_ID));
    log.info("MAIN", "═══════════════════════════════════");

    // 1.5. Initialize GPIO
    pinMode(BUTTON_CALIB_PIN, INPUT_PULLUP);
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN, LOW);
    log.info("MAIN", "GPIO initialized: Button=" + String(BUTTON_CALIB_PIN) + ", LED=" + String(LED_STATUS_PIN));

    // 2. Initialize Sensors
    log.info("MAIN", "Initializing Sensors...");
    
    if (!imu.begin()) {
        log.error("MAIN", "❌ IMU Initialization FAILED! (Check wiring)");
        // Don't halt, try to continue with other systems
    } else {
        log.info("MAIN", "✅ IMU Initialized");
    }
    
    if (!gps.begin()) {
        log.error("MAIN", "❌ GPS Initialization FAILED! (Check wiring)");
    } else {
        log.info("MAIN", "✅ GPS Initialized");
    }

    // 3. Initialize Network
    log.info("MAIN", "Initializing Network...");
    
    // Register receive callback BEFORE begin() if possible, 
    // or pass it to a handler. EspNowTransport uses singleton callback logic.
    // We'll use the transport interface method.
    
    if (!transport.begin()) {
        log.error("MAIN", "❌ Network Initialization FAILED!");
        while(1) delay(1000); // Network is critical, halt if failed
    }
    
    transport.onReceive(onDataReceived);
    log.info("MAIN", "✅ Network Initialized (ESP-NOW)");

    // 4. Initialize DataLogger
    log.info("MAIN", "Initializing SD Card...");
    if (!dataLogger.begin()) {
        log.error("MAIN", "❌ SD Card Initialization FAILED!");
        log.error("MAIN", "Data logging disabled (system will still function)");
    } else {
        log.info("MAIN", "✅ SD Card Initialized (Session " + String(dataLogger.getSessionNumber() + 1) + " ready)");
        log.info("MAIN", "Press button (GPIO " + String(BUTTON_CALIB_PIN) + ") to start/stop logging");
    }

    log.info("MAIN", "✅ System Ready. Broadcasting at 10Hz.");
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
    uint32_t now = millis();

    // 1. Handle button input
    handleButton();

    // 2. Update LED feedback
    updateLED();

    // 3. Update Sensors (High Frequency polling)
    gps.update(); // Drains UART buffer
    // imu.update() is not needed, we poll via read()

    // 4. Broadcast V2V Message (10 Hz) + Log if active
    if (now - lastBroadcastTime >= 100) {
        V2VMessage msg = buildV2VMessage();

        // Send broadcast (to FF:FF:FF:FF:FF:FF)
        bool success = transport.send((uint8_t*)&msg, sizeof(msg));

        if (!success) {
            // Logger::getInstance().warning("V2V", "Send failed");
        }

        // Log sample if logging active
        if (dataLogger.isLogging()) {
            auto gpsData = gps.read();
            dataLogger.logSample(msg, gpsData);
        }

        lastBroadcastTime = now;
    }
    
    // 5. Debug Print (1 Hz)
    if (now - lastPrintTime >= 1000) {
        IGpsSensor::GpsData gpsInfo = gps.read();

        String statusMsg = "";

        if (gpsInfo.valid || gpsInfo.cached) {
            statusMsg = "GPS Fix: YES | Sats: " + String(gpsInfo.satellites) +
                       " | Speed: " + String(gpsInfo.speed) + " m/s";
        } else {
            statusMsg = "GPS Fix: NO  | Searching...";
        }

        // Add logging status
        if (dataLogger.isLogging()) {
            statusMsg += " | 📝 LOGGING (" + String(dataLogger.getRowCount()) + " rows)";
        } else {
            statusMsg += " | ⏸ Not logging";
        }

        Logger::getInstance().info("STATUS", statusMsg);

        lastPrintTime = now;
    }
}

#endif // UNIT_TEST