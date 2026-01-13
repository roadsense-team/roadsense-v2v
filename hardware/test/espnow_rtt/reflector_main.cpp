/**
 * @file reflector_main.cpp
 * @brief RTT Reflector - Simple ESP-NOW echo for round-trip time measurement
 *
 * Receives RTT packets and immediately echoes them back.
 * Dumb reflector - no processing, no SD, no GPS, no sensors.
 *
 * Hardware requirements:
 * - ESP32 DevKit (Reflector unit - typically V002 or V003)
 *
 * @date December 29, 2025
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// RTT packet structure (for size validation only)
#include "RTTPacket.h"

// Logger for debug output
#include "utils/Logger.h"

// =============================================================================
// Configuration
// =============================================================================

#define ESPNOW_CHANNEL    1

// =============================================================================
// Global State
// =============================================================================

Logger& logger = Logger::getInstance();
uint32_t echoCount = 0;

// =============================================================================
// ESP-NOW Callbacks
// =============================================================================

/**
 * @brief ESP-NOW send callback
 * Minimal - just track failures
 */
void onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    // Minimal logging - avoid heap allocation in callback
    if (status != ESP_NOW_SEND_SUCCESS) {
        // Count failures silently
    }
}

/**
 * @brief ESP-NOW receive callback
 * Receives packet -> echoes immediately
 * ZERO PROCESSING: Do not unpack, do not validate sequence, just echo.
 */
void onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    // Size validation only
    if (len != sizeof(RTTPacket)) {
        return;  // Ignore non-RTT packets
    }

    // Echo back to sender (use broadcast - sender will receive)
    uint8_t broadcastAddr[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    esp_now_send(broadcastAddr, data, len);

    echoCount++;
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
// Main Logic
// =============================================================================

void setup() {
    // Initialize logger
    logger.begin(115200);
    delay(100);

    logger.info("RTT", "========================================");
    logger.info("RTT", "RTT Reflector Starting...");
    logger.info("RTT", "========================================");

    // Initialize ESP-NOW
    if (!initEspNow()) {
        logger.error("RTT", "FATAL: ESP-NOW init failed, halting");
        while (1) delay(1000);
    }

    logger.info("RTT", "Setup complete, waiting for packets to echo");
}

void loop() {
    // Print stats every 10 seconds (debug only)
    static uint32_t lastPrint = 0;
    uint32_t now = millis();

    if (now - lastPrint >= 10000) {
        logger.info("RTT", "Echoed " + String(echoCount) + " packets");
        lastPrint = now;
    }

    // Nothing else to do - all work happens in callback
    delay(10);
}
