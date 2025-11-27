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

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================
MPU6500Driver imu;
NEO6M_Driver gps;
EspNowTransport transport;

// Timer variables
uint32_t lastBroadcastTime = 0;
uint32_t lastPrintTime = 0;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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
    
    log.info("MAIN", "✅ System Ready. Broadcasting at 10Hz.");
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
    uint32_t now = millis();
    
    // 1. Update Sensors (High Frequency polling)
    gps.update(); // Drains UART buffer
    // imu.update() is not needed, we poll via read()
    
    // 2. Broadcast V2V Message (10 Hz)
    if (now - lastBroadcastTime >= 100) {
        V2VMessage msg = buildV2VMessage();
        
        // Send broadcast (to FF:FF:FF:FF:FF:FF)
        bool success = transport.send((uint8_t*)&msg, sizeof(msg));
        
        if (!success) {
            // Logger::getInstance().warning("V2V", "Send failed");
        }
        
        lastBroadcastTime = now;
    }
    
    // 3. Debug Print (1 Hz)
    if (now - lastPrintTime >= 1000) {
        // Blink LED if available
        #ifdef LED_BUILTIN
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        #endif
        
        IGpsSensor::GpsData gpsInfo = gps.read();
        
        if (gpsInfo.valid || gpsInfo.cached) {
             Logger::getInstance().info("STATUS", 
                "GPS Fix: YES | Sats: " + String(gpsInfo.satellites) + 
                " | Speed: " + String(gpsInfo.speed) + " m/s");
        } else {
             Logger::getInstance().info("STATUS", "GPS Fix: NO  | Searching...");
        }
        
        lastPrintTime = now;
    }
}

#endif // UNIT_TEST