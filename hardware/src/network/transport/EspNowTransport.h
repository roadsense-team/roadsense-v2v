/**
 * @file EspNowTransport.h
 * @brief ESP-NOW transport layer implementation
 *
 * Implements ITransport interface for ESP-NOW wireless communication.
 * Preserves the critical initialization sequence from legacy firmware.
 *
 * CRITICAL: ESP-NOW initialization order is fragile - do not modify without testing!
 *
 * Network constraints:
 * - Max payload: 250 bytes
 * - Broadcast-only (FF:FF:FF:FF:FF:FF)
 * - Channel: 1 (configurable)
 * - No encryption (for maximum compatibility)
 */

#ifndef ESPNOW_TRANSPORT_H
#define ESPNOW_TRANSPORT_H

#include "../../../include/ITransport.h"
#include "../../utils/Logger.h"
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

/**
 * @class EspNowTransport
 * @brief ESP-NOW wireless transport implementation
 *
 * Provides broadcast-based V2V communication using ESP-NOW protocol.
 * Follows the proven initialization sequence from legacy firmware.
 *
 * SINGLETON PATTERN:
 * This class uses a singleton pattern because ESP-NOW callbacks
 * are static C functions that need to route to instance methods.
 *
 * USAGE:
 * @code
 * // Correct usage (single instance):
 * EspNowTransport* transport = new EspNowTransport(1);
 * transport->begin();
 * // ... use transport ...
 * delete transport;  // Safe - callbacks unregistered in destructor
 *
 * // WRONG - will halt system:
 * EspNowTransport transport1(1);
 * EspNowTransport transport2(6);  // ← FATAL: Second instance not allowed
 * @endcode
 *
 * SAFETY GUARANTEES:
 * - Only one instance can exist (enforced in constructor)
 * - Destructor unregisters callbacks before cleanup (prevents use-after-free)
 * - Safe to destroy instance even during active communication
 */
class EspNowTransport : public ITransport {
public:
    /**
     * @brief Constructor
     * @param channel WiFi channel (1-13, default 1)
     */
    EspNowTransport(uint8_t channel = 1);

    /**
     * @brief Destructor - cleanup ESP-NOW
     */
    ~EspNowTransport();

    /**
     * @brief Initialize ESP-NOW transport
     *
     * CRITICAL: Initialization sequence order must not be changed!
     * 1. WiFi.mode(WIFI_STA)
     * 2. WiFi.disconnect()  ← CRITICAL: removes auto-connect interference
     * 3. esp_wifi_set_channel()
     * 4. esp_now_init()
     * 5. Register callbacks
     * 6. Add broadcast peer
     *
     * @return true if initialization successful
     */
    bool begin() override;

    /**
     * @brief Send data via ESP-NOW broadcast
     * @param data Pointer to data buffer
     * @param len Length of data (max 250 bytes)
     * @return true if send queued successfully
     */
    bool send(const uint8_t* data, size_t len) override;

    /**
     * @brief Register callback for received data
     * @param callback Function to call when data received
     */
    void onReceive(ReceiveCallback callback) override;

    /**
     * @brief Get maximum payload size
     * @return 250 bytes (ESP-NOW limit)
     */
    size_t getMaxPayload() const override { return 250; }

    /**
     * @brief Get transport name
     * @return "ESP-NOW"
     */
    const char* getName() const override { return "ESP-NOW"; }

    /**
     * @brief Check if transport is ready
     * @return true if initialized and operational
     */
    bool isReady() const override { return initialized; }

    /**
     * @brief Get local MAC address
     * @param mac Buffer to store MAC (must be 6 bytes)
     */
    void getLocalMAC(uint8_t* mac) const;

    /**
     * @brief Get send statistics
     * @return Number of successful sends
     */
    uint32_t getSendCount() const { return sendCount; }

    /**
     * @brief Get receive statistics
     * @return Number of received messages
     */
    uint32_t getReceiveCount() const { return receiveCount; }

    /**
     * @brief Get failed send count
     * @return Number of failed sends
     */
    uint32_t getFailCount() const { return failCount; }

private:
    /**
     * @brief ESP-NOW send callback (static)
     *
     * Called by ESP-NOW when send completes.
     * Routes to instance method via singleton pattern.
     *
     * @param mac Destination MAC address
     * @param status Send status (ESP_NOW_SEND_SUCCESS or ESP_NOW_SEND_FAIL)
     */
    static void onDataSent(const uint8_t* mac, esp_now_send_status_t status);

    /**
     * @brief ESP-NOW receive callback (static)
     *
     * Called by ESP-NOW when data received.
     * Routes to instance method via singleton pattern.
     *
     * @param mac Sender MAC address
     * @param data Received data buffer
     * @param len Data length
     */
    static void onDataRecv(const uint8_t* mac, const uint8_t* data, int len);

    /**
     * @brief Instance send callback handler
     *
     * IMPORTANT: Minimal logging to avoid heap fragmentation.
     * This runs in WiFi task context at high frequency (10+ Hz).
     * String allocations here cause heap fragmentation over time.
     * Use getSendCount()/getFailCount() for statistics.
     */
    void handleDataSent(const uint8_t* mac, esp_now_send_status_t status);

    /**
     * @brief Instance receive callback handler
     *
     * IMPORTANT: Minimal logging to avoid heap fragmentation.
     * This runs in WiFi task context at high frequency (10+ Hz).
     * String allocations here cause heap fragmentation over time.
     * Use getReceiveCount() for statistics.
     */
    void handleDataRecv(const uint8_t* mac, const uint8_t* data, int len);

    // Configuration
    uint8_t channel;           ///< WiFi channel
    bool initialized;          ///< Initialization status

    // Callbacks
    ReceiveCallback receiveCallback;  ///< User receive callback

    // Statistics
    uint32_t sendCount;        ///< Successful sends
    uint32_t receiveCount;     ///< Received messages
    uint32_t failCount;        ///< Failed sends

    // Logger
    Logger& logger;

    // Singleton instance for callbacks
    static EspNowTransport* instance;
};

#endif // ESPNOW_TRANSPORT_H
