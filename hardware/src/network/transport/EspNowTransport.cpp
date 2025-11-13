/**
 * @file EspNowTransport.cpp
 * @brief ESP-NOW transport implementation
 */

#include "EspNowTransport.h"

// Initialize static instance
EspNowTransport* EspNowTransport::instance = nullptr;

EspNowTransport::EspNowTransport(uint8_t channel)
    : channel(channel),
      initialized(false),
      receiveCallback(nullptr),
      sendCount(0),
      receiveCount(0),
      failCount(0),
      logger(Logger::getInstance()) {
    instance = this;
}

EspNowTransport::~EspNowTransport() {
    if (initialized) {
        esp_now_deinit();
    }
    if (instance == this) {
        instance = nullptr;
    }
}

bool EspNowTransport::begin() {
    if (initialized) {
        logger.warning("ESP-NOW", "Already initialized");
        return true;
    }

    logger.info("ESP-NOW", "Initializing ESP-NOW transport...");

    // CRITICAL: This initialization sequence order must not be changed!
    // Step 1: Set WiFi to station mode
    WiFi.mode(WIFI_STA);
    logger.debug("ESP-NOW", "WiFi mode: WIFI_STA");

    // Step 2: Disconnect from any saved WiFi networks
    // CRITICAL: This removes STA auto-connect interference
    WiFi.disconnect();
    logger.debug("ESP-NOW", "WiFi disconnected (prevents auto-connect)");

    // Step 3: Set WiFi channel
    esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
    logger.info("ESP-NOW", "WiFi channel set to " + String(channel));

    // Step 4: Initialize ESP-NOW
    if (esp_now_init() != ESP_OK) {
        logger.error("ESP-NOW", "esp_now_init() failed!");
        return false;
    }
    logger.debug("ESP-NOW", "esp_now_init() successful");

    // Step 5: Register send callback
    esp_err_t sendResult = esp_now_register_send_cb(onDataSent);
    if (sendResult != ESP_OK) {
        logger.error("ESP-NOW", "Failed to register send callback: " + String(sendResult));
        esp_now_deinit();
        return false;
    }
    logger.debug("ESP-NOW", "Send callback registered");

    // Step 6: Register receive callback
    esp_err_t recvResult = esp_now_register_recv_cb(onDataRecv);
    if (recvResult != ESP_OK) {
        logger.error("ESP-NOW", "Failed to register recv callback: " + String(recvResult));
        esp_now_deinit();
        return false;
    }
    logger.debug("ESP-NOW", "Receive callback registered");

    // Step 7: Add broadcast peer (FF:FF:FF:FF:FF:FF)
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memset(peerInfo.peer_addr, 0xFF, 6);  // Broadcast address
    peerInfo.channel = channel;
    peerInfo.encrypt = false;

    esp_err_t peerResult = esp_now_add_peer(&peerInfo);
    if (peerResult != ESP_OK) {
        logger.error("ESP-NOW", "Failed to add broadcast peer: " + String(peerResult));
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

    initialized = true;
    logger.info("ESP-NOW", "✓ ESP-NOW transport initialized successfully");

    return true;
}

bool EspNowTransport::send(const uint8_t* data, size_t len) {
    if (!initialized) {
        logger.error("ESP-NOW", "Cannot send: not initialized");
        return false;
    }

    if (len > 250) {
        logger.error("ESP-NOW", "Payload too large: " + String(len) + " > 250 bytes");
        return false;
    }

    if (data == nullptr) {
        logger.error("ESP-NOW", "Cannot send: data is null");
        return false;
    }

    // Send to broadcast address (FF:FF:FF:FF:FF:FF)
    uint8_t broadcastAddress[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    esp_err_t result = esp_now_send(broadcastAddress, data, len);
    if (result == ESP_OK) {
        logger.debug("ESP-NOW", "Send queued (" + String(len) + " bytes)");
        return true;
    } else {
        logger.error("ESP-NOW", "Send failed: " + String(result));
        failCount++;
        return false;
    }
}

void EspNowTransport::onReceive(ReceiveCallback callback) {
    receiveCallback = callback;
    if (callback) {
        logger.debug("ESP-NOW", "Receive callback registered");
    } else {
        logger.debug("ESP-NOW", "Receive callback cleared");
    }
}

void EspNowTransport::getLocalMAC(uint8_t* mac) const {
    WiFi.macAddress(mac);
}

// Static callbacks (route to instance methods)

void EspNowTransport::onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    if (instance) {
        instance->handleDataSent(mac, status);
    }
}

void EspNowTransport::onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    if (instance) {
        instance->handleDataRecv(mac, data, len);
    }
}

// Instance callback handlers

void EspNowTransport::handleDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) {
        sendCount++;
        logger.debug("ESP-NOW", "✓ Send confirmed (total: " + String(sendCount) + ")");
    } else {
        failCount++;
        logger.warning("ESP-NOW", "✗ Send failed (total failures: " + String(failCount) + ")");
    }
}

void EspNowTransport::handleDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    receiveCount++;

    // Log sender MAC
    char macStr[18];
    snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    logger.debug("ESP-NOW", "Received " + String(len) + " bytes from " + String(macStr));

    // Call user callback if registered
    if (receiveCallback) {
        receiveCallback(data, len);
    }
}
