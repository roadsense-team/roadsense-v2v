/**
 * @file sender_main.cpp
 * @brief RTT Sender - ESP-NOW round-trip time measurement
 *
 * Sends RTT packets at 10Hz, logs results to SD card for ESP-NOW characterization.
 * Uses verified components: RTTCommon, RTTPacket, RTTLogging, RTTBuffer, RTTTimeout.
 *
 * Hardware requirements:
 * - ESP32 DevKit (Sender unit - typically V001)
 * - MPU6500 IMU @ I2C 0x68
 * - NEO-6M GPS @ UART2 (TX=16, RX=17)
 * - MicroSD card module @ SPI (CS=5)
 *
 * @date December 29, 2025
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <SPI.h>
#include <SdFat.h>
#include <TinyGPSPlus.h>
#include <Wire.h>

// Verified RTT components
#include "RTTCommon.h"
#include "RTTPacket.h"
#include "RTTLogging.h"
#include "RTTBuffer.h"
#include "RTTTimeout.h"

// Project drivers
#include "sensors/imu/MPU6500Driver.h"
#include "utils/Logger.h"

// =============================================================================
// Pin Definitions (from config.h)
// =============================================================================

// I2C (MPU6500)
#define I2C_SDA_PIN       21
#define I2C_SCL_PIN       22

// GPS (UART2)
#define GPS_RX_PIN        16
#define GPS_TX_PIN        17
#define GPS_BAUD_RATE     9600

// SD Card (SPI) - CRITICAL: Use 10 MHz for road reliability
#define SD_CS_PIN         5
#undef SD_SPI_MHZ         // Override config.h value
#define SD_SPI_MHZ        10  // 10 MHz (directive requirement, not 25 MHz)

// Type alias for convenience
using ImuData = IImuSensor::ImuData;

// ESP-NOW
#define ESPNOW_CHANNEL    1

// =============================================================================
// RTT Configuration
// =============================================================================

#define RTT_SEND_RATE_HZ      10                          // 10 Hz send rate
#define RTT_SEND_INTERVAL_MS  (1000 / RTT_SEND_RATE_HZ)   // 100ms
#define RTT_LOG_FILENAME      "/rtt_log.csv"

// =============================================================================
// Global State
// =============================================================================

// Circular buffer for pending packets (required by RTTBuffer.h)
RTTRecord pendingPackets[MAX_PENDING];

// Hardware drivers
MPU6500Driver imu;
TinyGPSPlus gps;
HardwareSerial& gpsSerial = Serial2;
SdFat sd;
SdFile logFile;

// Logging
Logger& logger = Logger::getInstance();

// State
uint32_t sequenceNumber = 0;
uint32_t lastSendTime = 0;
bool sdInitialized = false;
bool headerWritten = false;

// =============================================================================
// ESP-NOW Callbacks
// =============================================================================

/**
 * @brief ESP-NOW send callback
 * Called when send completes (success or fail)
 */
void onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    // Minimal logging in callback context
    if (status != ESP_NOW_SEND_SUCCESS) {
        // Log failures only
        logger.warning("ESP-NOW", "Send failed");
    }
}

/**
 * @brief ESP-NOW receive callback
 * Called when echo packet received from reflector
 */
void onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    uint32_t recvTime = millis();

    // Validate packet size
    if (len != sizeof(RTTPacket)) {
        return;  // Ignore malformed packets
    }

    // Extract sequence number from received packet
    const RTTPacket* packet = reinterpret_cast<const RTTPacket*>(data);
    uint32_t seq = packet->sequence;

    // Mark packet as received in circular buffer
    markPacketReceived(seq, recvTime);
}

// =============================================================================
// ESP-NOW Initialization (EXACT sequence from EspNowTransport.cpp)
// =============================================================================

bool initEspNow() {
    logger.info("ESP-NOW", "Initializing ESP-NOW transport...");

    // Step 1: Set WiFi to station mode
    WiFi.mode(WIFI_STA);
    logger.debug("ESP-NOW", "WiFi mode: WIFI_STA");

    // Step 2: Disconnect from any saved WiFi networks
    // CRITICAL: This removes STA auto-connect interference
    WiFi.disconnect();
    logger.debug("ESP-NOW", "WiFi disconnected (prevents auto-connect)");

    // Step 3: Set WiFi channel
    esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
    logger.info("ESP-NOW", "WiFi channel set to " + String(ESPNOW_CHANNEL));

    // Step 4: Initialize ESP-NOW
    if (esp_now_init() != ESP_OK) {
        logger.error("ESP-NOW", "esp_now_init() failed!");
        return false;
    }
    logger.debug("ESP-NOW", "esp_now_init() successful");

    // Step 5: Register send callback
    esp_err_t sendResult = esp_now_register_send_cb(onDataSent);
    if (sendResult != ESP_OK) {
        logger.error("ESP-NOW", "Failed to register send callback");
        esp_now_deinit();
        return false;
    }
    logger.debug("ESP-NOW", "Send callback registered");

    // Step 6: Register receive callback
    esp_err_t recvResult = esp_now_register_recv_cb(onDataRecv);
    if (recvResult != ESP_OK) {
        logger.error("ESP-NOW", "Failed to register recv callback");
        esp_now_deinit();
        return false;
    }
    logger.debug("ESP-NOW", "Receive callback registered");

    // Step 7: Add broadcast peer (FF:FF:FF:FF:FF:FF)
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memset(peerInfo.peer_addr, 0xFF, 6);  // Broadcast address
    peerInfo.channel = ESPNOW_CHANNEL;
    peerInfo.encrypt = false;

    esp_err_t peerResult = esp_now_add_peer(&peerInfo);
    if (peerResult != ESP_OK) {
        logger.error("ESP-NOW", "Failed to add broadcast peer");
        esp_now_deinit();
        return false;
    }
    logger.debug("ESP-NOW", "Broadcast peer added (FF:FF:FF:FF:FF:FF)");

    // Log local MAC address
    uint8_t mac[6];
    WiFi.macAddress(mac);
    char macStr[18];
    snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    logger.info("ESP-NOW", "Local MAC: " + String(macStr));

    logger.info("ESP-NOW", "ESP-NOW initialized successfully");
    return true;
}

// =============================================================================
// SD Card Initialization
// =============================================================================

bool initSD() {
    logger.info("SD", "Initializing SD card...");

    // Initialize SPI with explicit pins
    SPI.begin(18, 19, 23, SD_CS_PIN);  // SCK, MISO, MOSI, CS

    // Initialize SD card at 10 MHz (CRITICAL for road reliability)
    if (!sd.begin(SD_CS_PIN, SD_SCK_MHZ(SD_SPI_MHZ))) {
        logger.error("SD", "SD card initialization failed!");
        return false;
    }

    // Open/create log file (APPEND mode - preserves existing data)
    if (!logFile.open(RTT_LOG_FILENAME, O_RDWR | O_CREAT | O_AT_END)) {
        logger.error("SD", "Failed to open log file");
        return false;
    }

    // Check if file is empty (new file) - only write header if so
    uint32_t fileSize = logFile.fileSize();
    if (fileSize == 0) {
        logger.info("SD", "New log file, header will be written");
    } else {
        // File has existing data - skip header, mark as already written
        headerWritten = true;
        logger.info("SD", "Appending to existing log (" + String(fileSize) + " bytes)");
    }

    logger.info("SD", "SD card initialized: " + String(RTT_LOG_FILENAME));
    return true;
}

// =============================================================================
// Sensor Initialization
// =============================================================================

bool initSensors() {
    // Initialize IMU
    logger.info("IMU", "Initializing MPU6500...");
    if (!imu.begin()) {
        logger.error("IMU", "MPU6500 initialization failed!");
        return false;
    }
    logger.info("IMU", "MPU6500 initialized");

    // Initialize GPS
    logger.info("GPS", "Initializing NEO-6M...");
    gpsSerial.begin(GPS_BAUD_RATE, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    logger.info("GPS", "GPS UART initialized (waiting for fix...)");

    return true;
}

// =============================================================================
// Main Logic
// =============================================================================

void setup() {
    // Initialize logger
    logger.begin(115200);
    delay(100);

    logger.info("RTT", "========================================");
    logger.info("RTT", "RTT Sender Starting...");
    logger.info("RTT", "========================================");

    // Initialize circular buffer
    initCircularBuffer();
    logger.debug("RTT", "Circular buffer initialized");

    // Initialize ESP-NOW
    if (!initEspNow()) {
        logger.error("RTT", "FATAL: ESP-NOW init failed, halting");
        while (1) delay(1000);
    }

    // Initialize sensors
    if (!initSensors()) {
        logger.error("RTT", "FATAL: Sensor init failed, halting");
        while (1) delay(1000);
    }

    // Initialize SD card
    sdInitialized = initSD();
    if (!sdInitialized) {
        logger.warning("RTT", "SD card not available - running without logging");
    }

    logger.info("RTT", "Setup complete, starting RTT measurement");
}

void loop() {
    uint32_t currentTime = millis();

    // Feed GPS data
    while (gpsSerial.available()) {
        gps.encode(gpsSerial.read());
    }

    // === 10 Hz Send Loop ===
    if (currentTime - lastSendTime >= RTT_SEND_INTERVAL_MS) {
        lastSendTime = currentTime;

        // Read sensors
        ImuData imuData = imu.read();

        // Build RTT packet
        RTTPacket packet;
        memset(&packet, 0, sizeof(packet));

        packet.sequence = sequenceNumber;
        packet.send_time_ms = currentTime;

        // GPS data (use last known position, or 0 if no fix)
        if (gps.location.isValid()) {
            packet.sender_lat = gps.location.lat();
            packet.sender_lon = gps.location.lng();
        }
        if (gps.speed.isValid()) {
            packet.sender_speed = gps.speed.mps();  // m/s
        }
        if (gps.course.isValid()) {
            packet.sender_heading = gps.course.deg();
        }

        // IMU data
        packet.accel_x = imuData.accel[0];
        packet.accel_y = imuData.accel[1];
        packet.accel_z = imuData.accel[2];

        // Create RTTRecord for buffer
        RTTRecord record;
        memset(&record, 0, sizeof(record));
        record.sequence = sequenceNumber;
        record.send_time_ms = currentTime;
        record.recv_time_ms = 0;  // Not received yet
        record.lat = packet.sender_lat;
        record.lon = packet.sender_lon;
        record.speed = packet.sender_speed;
        record.heading = packet.sender_heading;
        record.accel_x = packet.accel_x;
        record.accel_y = packet.accel_y;
        record.accel_z = packet.accel_z;
        record.received = false;

        // Add to circular buffer
        addPendingPacket(&record);

        // Send via ESP-NOW (broadcast)
        uint8_t broadcastAddr[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
        esp_now_send(broadcastAddr, (uint8_t*)&packet, sizeof(packet));

        sequenceNumber++;

        // Debug print every 10 packets (1 second)
        if (sequenceNumber % 10 == 0) {
            logger.debug("RTT", "Sent seq=" + String(sequenceNumber - 1) +
                        " GPS=" + String(gps.location.isValid() ? "OK" : "NO"));
        }
    }

    // === Check Timeouts & Write to SD ===
    if (sdInitialized) {
        // Write CSV header if needed
        if (!headerWritten) {
            char header[256];
            generateCSVHeader(header, sizeof(header));
            logFile.print(header);
            headerWritten = true;
        }

        // Scan buffer for records to write
        for (int i = 0; i < MAX_PENDING; i++) {
            RTTRecord* record = &pendingPackets[i];

            if (shouldWriteRecord(record, currentTime)) {
                // Format and write CSV row
                char row[256];
                formatCSVRow(row, sizeof(row), record);
                logFile.print(row);

                // Clear slot after writing
                record->send_time_ms = 0;
            }
        }

        // Flush every second
        static uint32_t lastFlush = 0;
        if (currentTime - lastFlush >= 1000) {
            logFile.sync();
            lastFlush = currentTime;
        }
    }
}
