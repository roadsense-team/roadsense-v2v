/**
 * @file RTTBuffer.h
 * @brief Circular buffer for pending RTT packets
 */

#ifndef RTT_BUFFER_H
#define RTT_BUFFER_H

#include <Arduino.h>
#include "RTTCommon.h"
#include <cstring>

// External global buffer (defined in test or main firmware)
extern RTTRecord pendingPackets[MAX_PENDING];

/**
 * @brief Initialize circular buffer
 *
 * Clears all buffer slots by zeroing memory.
 * Should be called once at startup.
 */
void initCircularBuffer() {
    memset(pendingPackets, 0, sizeof(RTTRecord) * MAX_PENDING);
}

/**
 * @brief Calculate buffer index from sequence number
 *
 * Uses modulo to map sequence numbers to buffer slots.
 * Handles wraparound: sequence 100 -> slot 0, etc.
 *
 * @param sequence Packet sequence number
 * @return Buffer index (0 to MAX_PENDING-1)
 */
int getBufferIndex(uint32_t sequence) {
    return sequence % MAX_PENDING;
}

/**
 * @brief Add a pending packet to the buffer
 *
 * Stores packet in slot calculated from sequence number.
 * May overwrite old packet if collision occurs (same slot).
 *
 * @param record RTT record to add
 */
void addPendingPacket(const RTTRecord* record) {
    int index = getBufferIndex(record->sequence);
    memcpy(&pendingPackets[index], record, sizeof(RTTRecord));
}

/**
 * @brief Mark a packet as received
 *
 * Updates recv_time_ms and sets received=true.
 * IMPORTANT: Only marks if sequence number matches the current packet in that slot.
 * This prevents marking wrong packet if collision occurred.
 *
 * @param sequence Packet sequence number
 * @param recv_time_ms Time packet was received (millis())
 */
void markPacketReceived(uint32_t sequence, uint32_t recv_time_ms) {
    int index = getBufferIndex(sequence);

    // Verify sequence number matches (collision check)
    if (pendingPackets[index].sequence == sequence) {
        pendingPackets[index].recv_time_ms = recv_time_ms;
        pendingPackets[index].received = true;
    }
    // If sequence doesn't match, packet was overwritten - do nothing
}

/**
 * @brief Check if a packet is pending in the buffer
 *
 * Returns true if:
 * - Slot has a packet with matching sequence number
 * - send_time_ms is non-zero (slot is not empty)
 *
 * @param sequence Packet sequence number
 * @return true if packet is pending
 */
bool isPacketPending(uint32_t sequence) {
    int index = getBufferIndex(sequence);
    return (pendingPackets[index].sequence == sequence &&
            pendingPackets[index].send_time_ms != 0);
}

/**
 * @brief Check if a packet has been received
 *
 * Returns true if:
 * - Packet is pending (sequence matches, slot not empty)
 * - received flag is true
 *
 * @param sequence Packet sequence number
 * @return true if packet was received
 */
bool isPacketReceived(uint32_t sequence) {
    int index = getBufferIndex(sequence);
    return (pendingPackets[index].sequence == sequence &&
            pendingPackets[index].send_time_ms != 0 &&
            pendingPackets[index].received);
}

#endif // RTT_BUFFER_H
