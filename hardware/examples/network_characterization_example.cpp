/**
 * @file network_characterization_example.cpp
 * @brief Example: ESP-NOW Network Characterization for Emulator Parameters
 *
 * PURPOSE:
 * --------
 * Collect REAL ESP-NOW performance data (latency, packet loss, jitter)
 * to calibrate the Python ESP-NOW emulator for SUMO training.
 *
 * HARDWARE SETUP:
 * ---------------
 * - 2 ESP32 boards (V001 and V002)
 * - Each with: MPU6500 + QMC5883L + NEO6M GPS + SD card
 * - Flash this firmware on BOTH boards
 * - Change VEHICLE_ID to "V001" on first board, "V002" on second
 *
 * TEST PROCEDURE:
 * ---------------
 * 1. Insert SD cards in both boards
 * 2. Power on both boards
 * 3. Wait for GPS fix (~60 seconds)
 * 4. Press button on BOTH boards → logging starts
 * 5. Drive around for 20-30 minutes:
 *    - Start close (5m)
 *    - Move apart to 50m
 *    - Move apart to 100m
 *    - Return close
 *    - Include obstacles (buildings, cars)
 * 6. Press button on BOTH boards → logging stops
 * 7. Extract SD cards → get 4 files:
 *    - v001_tx_001.csv, v001_rx_001.csv
 *    - v002_tx_001.csv, v002_rx_001.csv
 *
 * OUTPUT FILES:
 * -------------
 * TX Log: timestamp_local_ms,msg_timestamp,vehicle_id,lat,lon,speed,heading
 * RX Log: timestamp_local_ms,msg_timestamp,from_vehicle_id,lat,lon,speed,heading
 *
 * POST-PROCESSING:
 * ----------------
 * Run: python scripts/analyze_espnow_performance.py
 * Output: emulator_params.json (latency, loss, jitter distributions)
 */

#include <Arduino.h>
#include "config.h"
#include "logging/DataLogger.h"
#include "sensors/MPU6500Driver.h"
#include "sensors/QMC5883LDriver.h"  // ✅ Magnetometer (validated Dec 22, 2025)
#include "sensors/NEO6M_Driver.h"
#include "network/mesh/NetworkManager.h"
#include "network/mesh/PackageManager.h"
#include "network/transport/EspNowTransport.h"
#include "network/protocol/V2VMessage.h"
#include "utils/Logger.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

#define VEHICLE_ID "V001"  // ⚠️ CHANGE TO "V002" FOR SECOND BOARD!

#define BUTTON_LOG_PIN 4    // Button to start/stop logging
#define LED_STATUS_PIN 2    // LED status indicator

#define BROADCAST_INTERVAL_MS 100  // 10 Hz broadcast rate

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

DataLogger dataLogger;
MPU6500Driver imu;
QMC5883LDriver mag;  // ✅ Magnetometer (3-axis compass)
NEO6M_Driver gps;
NetworkManager networkManager;
PackageManager packageManager;
EspNowTransport espNowTransport;

// State
bool logging = false;
bool buttonLastState = HIGH;
unsigned long lastBroadcastTime = 0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);

    Logger& log = Logger::getInstance();
    log.info("Main", "RoadSense V2V - Network Characterization Mode");
    log.info("Main", "Vehicle ID: " + String(VEHICLE_ID));

    // 1. Initialize button and LED
    pinMode(BUTTON_LOG_PIN, INPUT_PULLUP);
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN, LOW);

    // 2. Initialize sensors
    if (!imu.begin()) {
        log.error("Main", "❌ MPU6500 init failed");
        while (1) { delay(1000); }
    }
    log.info("Main", "✅ MPU6500 initialized");

    if (!mag.begin()) {
        log.error("Main", "❌ QMC5883L init failed");
        while (1) { delay(1000); }
    }
    log.info("Main", "✅ QMC5883L initialized");

    if (!gps.begin(GPS_RX_PIN, GPS_TX_PIN)) {
        log.error("Main", "❌ GPS init failed");
        while (1) { delay(1000); }
    }
    log.info("Main", "✅ GPS initialized");

    // 3. Initialize ESP-NOW network
    if (!espNowTransport.begin()) {
        log.error("Main", "❌ ESP-NOW init failed");
        while (1) { delay(1000); }
    }
    log.info("Main", "✅ ESP-NOW initialized");

    // Register ESP-NOW receive callback (we'll implement this below)
    espNowTransport.setReceiveCallback(onV2VMessageReceived);

    // 4. Initialize DataLogger
    if (!dataLogger.begin()) {
        log.error("Main", "❌ DataLogger init failed - check SD card!");
        while (1) { delay(1000); }
    }
    log.info("Main", "✅ DataLogger initialized");

    // Set to Mode 1 (Network Characterization)
    dataLogger.setMode(MODE_NETWORK_CHARACTERIZATION);

    log.info("Main", "🚀 System ready!");
    log.info("Main", "   Press button to start/stop logging");
    log.info("Main", "   LED ON = logging active");

    digitalWrite(LED_STATUS_PIN, HIGH);  // Blink once to show ready
    delay(200);
    digitalWrite(LED_STATUS_PIN, LOW);
}

// ============================================================================
// LOOP
// ============================================================================

void loop() {
    unsigned long now = millis();

    // 1. Update sensors
    gps.update();
    imu.update();
    mag.update();

    // 2. Handle button (start/stop logging)
    handleButton();

    // 3. Broadcast V2V message at 10 Hz
    if (now - lastBroadcastTime >= BROADCAST_INTERVAL_MS) {
        broadcastV2VMessage();
        lastBroadcastTime = now;
    }

    // 4. Blink LED if logging
    if (logging) {
        // Fast blink = logging active
        digitalWrite(LED_STATUS_PIN, (now / 500) % 2);
    }

    delay(10);  // Small delay for stability
}

// ============================================================================
// V2V MESSAGE HANDLING
// ============================================================================

/**
 * Build V2V message from current sensor state
 */
V2VMessage buildV2VMessage() {
    V2VMessage msg;

    strcpy(msg.vehicleId, VEHICLE_ID);
    msg.timestamp = millis();

    // Get GPS data
    auto gpsData = gps.getData();
    msg.position.lat = gpsData.lat;
    msg.position.lon = gpsData.lon;
    msg.position.alt = gpsData.alt;
    msg.dynamics.speed = gpsData.speed;
    msg.dynamics.heading = gpsData.heading;

    // Get IMU data
    auto accelData = imu.getAcceleration();
    msg.sensors.accel[0] = accelData.x;
    msg.sensors.accel[1] = accelData.y;
    msg.sensors.accel[2] = accelData.z;

    auto gyroData = imu.getAngularVelocity();
    msg.sensors.gyro[0] = gyroData.x;
    msg.sensors.gyro[1] = gyroData.y;
    msg.sensors.gyro[2] = gyroData.z;

    // Get Magnetometer data
    auto magData = mag.getMagneticField();
    msg.sensors.mag[0] = magData.x;
    msg.sensors.mag[1] = magData.y;
    msg.sensors.mag[2] = magData.z;

    // Dynamics (simple calculation from accel)
    msg.dynamics.longAccel = accelData.x;
    msg.dynamics.latAccel = accelData.y;

    return msg;
}

/**
 * Broadcast V2V message via ESP-NOW
 */
void broadcastV2VMessage() {
    V2VMessage msg = buildV2VMessage();

    // CRITICAL: Log TX BEFORE sending (to capture exact timestamp)
    if (logging) {
        dataLogger.logTxMessage(msg);
    }

    // Send via ESP-NOW (broadcast to all peers)
    espNowTransport.broadcast((uint8_t*)&msg, sizeof(V2VMessage));
}

/**
 * ESP-NOW receive callback - called when message received
 */
void onV2VMessageReceived(const uint8_t* macAddr, const uint8_t* data, int len) {
    if (len != sizeof(V2VMessage)) {
        Logger::getInstance().warning("Main", "Invalid message size: " + String(len));
        return;
    }

    // Parse message
    V2VMessage msg;
    memcpy(&msg, data, sizeof(V2VMessage));

    // CRITICAL: Log RX immediately (timestamp = now)
    if (logging) {
        dataLogger.logRxMessage(msg);
    }

    // Optional: Add to PackageManager for deduplication (not needed for characterization)
    // packageManager.addPackage(macAddr, msg, msg.hopCount);
}

// ============================================================================
// BUTTON HANDLING
// ============================================================================

void handleButton() {
    bool buttonState = digitalRead(BUTTON_LOG_PIN);

    // Detect button press (falling edge)
    if (buttonState == LOW && buttonLastState == HIGH) {
        delay(50);  // Debounce
        if (digitalRead(BUTTON_LOG_PIN) == LOW) {
            toggleLogging();
        }
    }

    buttonLastState = buttonState;
}

/**
 * Toggle logging on/off
 */
void toggleLogging() {
    Logger& log = Logger::getInstance();

    if (!logging) {
        // Start logging
        if (dataLogger.startCharacterizationLogging(VEHICLE_ID)) {
            logging = true;
            log.info("Main", "✅ Logging STARTED");
            digitalWrite(LED_STATUS_PIN, HIGH);
        } else {
            log.error("Main", "❌ Failed to start logging");
        }
    } else {
        // Stop logging
        dataLogger.stopLogging();
        logging = false;
        log.info("Main", "✅ Logging STOPPED");
        digitalWrite(LED_STATUS_PIN, LOW);
    }
}
