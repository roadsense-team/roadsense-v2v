/**
 * @file ITransport.h
 * @brief Abstract interface for network transport layers
 *
 * Provides a hardware-independent interface for V2V communication.
 * Implementations: ESP-NOW (primary), UDP (for HIL bridge)
 *
 * This abstraction allows swapping transport layers without changing
 * application code (e.g., ESP-NOW for hardware, UDP for simulation).
 */

#ifndef ITRANSPORT_H
#define ITRANSPORT_H

#include <Arduino.h>
#include <functional>

/**
 * @class ITransport
 * @brief Abstract base class for network transport layers
 *
 * Defines a simple send/receive interface for broadcasting V2V messages.
 * Transport layer is responsible for:
 * - Physical layer communication (radio, WiFi, etc.)
 * - Basic error handling
 * - Callback-based receive notification
 */
class ITransport {
public:
    /**
     * @typedef ReceiveCallback
     * @brief Callback function type for incoming messages
     *
     * @param data Pointer to received data buffer
     * @param len Length of received data in bytes
     */
    using ReceiveCallback = std::function<void(const uint8_t* data, size_t len)>;

    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~ITransport() = default;

    /**
     * @brief Initialize transport layer
     *
     * Performs hardware-specific initialization:
     * - ESP-NOW: WiFi setup, peer registration
     * - UDP: Socket creation, binding
     *
     * @return true if initialization successful, false otherwise
     */
    virtual bool begin() = 0;

    /**
     * @brief Send data to all peers (broadcast)
     *
     * Transmits data to all reachable peers using transport-specific
     * broadcast mechanism:
     * - ESP-NOW: Broadcast to FF:FF:FF:FF:FF:FF
     * - UDP: Multicast or broadcast to subnet
     *
     * @param data Pointer to data buffer to send
     * @param len Length of data in bytes
     * @return true if send initiated successfully, false on error
     *
     * @note ESP-NOW max payload is 250 bytes
     * @note This is non-blocking, actual transmission may take time
     */
    virtual bool send(const uint8_t* data, size_t len) = 0;

    /**
     * @brief Register callback for incoming messages
     *
     * Sets up a callback function that will be invoked whenever
     * a message is received from a peer.
     *
     * @param callback Function to call with received data
     *
     * @note Callback may be invoked from interrupt context (ESP-NOW)
     * @note Keep callback processing short and non-blocking
     */
    virtual void onReceive(ReceiveCallback callback) = 0;

    /**
     * @brief Get maximum payload size supported by transport
     *
     * Different transport layers have different size limits:
     * - ESP-NOW: 250 bytes
     * - UDP: Typically 1472 bytes (ethernet MTU)
     *
     * @return Maximum number of bytes that can be sent in one message
     */
    virtual size_t getMaxPayload() const = 0;

    /**
     * @brief Get transport layer name for debugging
     *
     * @return String identifier (e.g., "ESP-NOW", "UDP")
     */
    virtual const char* getName() const = 0;

    /**
     * @brief Check if transport is ready for operation
     *
     * @return true if initialized and ready to send/receive
     */
    virtual bool isReady() const = 0;
};

#endif // ITRANSPORT_H
