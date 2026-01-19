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

    // SINGLETON ENFORCEMENT: Prevent multiple instances
    if (instance != nullptr) {
        logger.error("ESP-NOW", "FATAL: EspNowTransport instance already exists!");
        logger.error("ESP-NOW", "Only ONE instance allowed (singleton pattern)");
        logger.error("ESP-NOW", "Existing instance must be destroyed before creating new one");

        // Halt execution - multiple instances would cause callback routing errors
        while(1) {
            delay(1000);
            logger.error("ESP-NOW", "System halted due to singleton violation");
        }
    }

    instance = this;
    logger.debug("ESP-NOW", "EspNowTransport instance created");
}

EspNowTransport::~EspNowTransport() {
    logger.debug("ESP-NOW", "Destroying EspNowTransport instance");

    // CRITICAL: Unregister callbacks FIRST (while object still valid)
    // This prevents race condition where WiFi callback fires after destruction
    if (initialized) {
        logger.debug("ESP-NOW", "Unregistering ESP-NOW callbacks");

        // Unregister callbacks before deinit (prevents use-after-free)
        esp_now_unregister_send_cb();
        esp_now_unregister_recv_cb();

        // Small delay to ensure pending callbacks complete
        delay(10);

        // Now safe to deinit
        esp_now_deinit();
        initialized = false;

        logger.debug("ESP-NOW", "ESP-NOW deinitialized");
    }

    // Clear instance pointer LAST (after callbacks unregistered)
    if (instance == this) {
        instance = nullptr;
        logger.debug("ESP-NOW", "Instance pointer cleared");
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

    // Step 3.5: Enable Long Range (LR) Mode and Max TX Power
    // =========================================================================
    // WIFI_PROTOCOL_LR: Proprietary Espressif mode (~250-500kbps)
    // - Increases receiver sensitivity by +5-10 dB
    // - Potentially doubles/triples range compared to standard 802.11
    // - CRITICAL: Both sender and receiver MUST use LR mode
    // Reference: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#lr
    // =========================================================================
    esp_err_t lr_result = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_LR);
    if (lr_result == ESP_OK) {
        logger.info("ESP-NOW", "Long Range (LR) mode ENABLED");
    } else {
        logger.warning("ESP-NOW", "Failed to set LR mode (err=" + String(lr_result) + "), using standard protocol");
    }

    // Set maximum TX power: 84 × 0.25 = 21 dBm (hardware caps to ~19.5 dBm)
    esp_err_t tx_result = esp_wifi_set_max_tx_power(84);
    if (tx_result == ESP_OK) {
        logger.info("ESP-NOW", "TX power set to maximum (84 = ~21dBm)");
    } else {
        logger.warning("ESP-NOW", "Failed to set TX power (err=" + String(tx_result) + ")");
    }

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

/**
 * CALLBACK ROUTING SAFETY:
 *
 * These static callbacks are registered with ESP-NOW framework and
 * route to instance methods via singleton pointer.
 *
 * RACE CONDITION PREVENTION:
 * - Constructor enforces singleton (only one instance exists)
 * - Destructor unregisters callbacks BEFORE clearing instance pointer
 * - 10ms delay in destructor allows pending callbacks to complete
 *
 * This design is safe because:
 * 1. Only one instance can exist (enforced in constructor)
 * 2. Callbacks are unregistered before object destruction
 * 3. Instance pointer is cleared AFTER callbacks unregistered
 *
 * NOTE: If instance is null here, it means ESP-NOW fired callback
 * after unregister (ESP-IDF race condition). We safely ignore it.
 */

void EspNowTransport::onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    if (instance) {
        instance->handleDataSent(mac, status);
    }
    // If instance is null, callbacks were unregistered - safe to ignore
}

void EspNowTransport::onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    if (instance) {
        instance->handleDataRecv(mac, data, len);
    }
    // If instance is null, callbacks were unregistered - safe to ignore
}

// Instance callback handlers

void EspNowTransport::handleDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) {
        sendCount++;
        // Minimal logging - avoid heap allocation in WiFi callback
        // Full stats available via getSendCount()
    } else {
        failCount++;
        // Log failures only (critical info, worth the allocation cost)
        logger.warning("ESP-NOW", "Send failed");
    }
}

void EspNowTransport::handleDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    receiveCount++;

    // Minimal logging - avoid heap allocation in WiFi callback
    // Detailed packet inspection should be done in application layer
    // Statistics available via getReceiveCount()

    // Call user callback if registered
    if (receiveCallback) {
        receiveCallback(data, len);
    }
}
