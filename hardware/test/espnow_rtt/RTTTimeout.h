/**
 * @file RTTTimeout.h
 * @brief Timeout handling for RTT packet tracking
 */

#ifndef RTT_TIMEOUT_H
#define RTT_TIMEOUT_H

#include <Arduino.h>
#include "RTTCommon.h"

/**
 * @brief Calculate elapsed time between two timestamps
 *
 * Handles millis() wraparound correctly using unsigned arithmetic.
 * Wraparound example:
 *   start = 0xFFFFFF00 (4294967040)
 *   current = 0x00000200 (512)
 *   elapsed = 0x00000200 - 0xFFFFFF00 = 0x300 = 768ms ✓
 *
 * @param start_time_ms Start timestamp (millis())
 * @param current_time_ms Current timestamp (millis())
 * @return Elapsed time in milliseconds
 */
uint32_t getElapsedTime(uint32_t start_time_ms, uint32_t current_time_ms) {
    // Unsigned subtraction handles wraparound automatically
    return current_time_ms - start_time_ms;
}

/**
 * @brief Check if a packet has timed out
 *
 * A packet times out if it's been waiting >= 500ms without receiving an echo.
 *
 * @param send_time_ms Time packet was sent (millis())
 * @param current_time_ms Current time (millis())
 * @return true if packet timed out (elapsed >= 500ms)
 */
bool isTimedOut(uint32_t send_time_ms, uint32_t current_time_ms) {
    uint32_t elapsed = getElapsedTime(send_time_ms, current_time_ms);
    return elapsed >= TIMEOUT_MS;
}

/**
 * @brief Check if a packet was recently sent
 *
 * Opposite of isTimedOut - returns true if packet is still within timeout window.
 *
 * @param send_time_ms Time packet was sent (millis())
 * @param current_time_ms Current time (millis())
 * @return true if packet sent < 500ms ago
 */
bool isRecentlySent(uint32_t send_time_ms, uint32_t current_time_ms) {
    uint32_t elapsed = getElapsedTime(send_time_ms, current_time_ms);
    return elapsed < TIMEOUT_MS;
}

/**
 * @brief Determine if an RTT record should be written to SD card
 *
 * Write policy:
 * - WRITE if received (got echo - write immediately)
 * - WRITE if timed out (>= 500ms without echo - mark as lost)
 * - DON'T WRITE if waiting (< 500ms, no echo yet - still pending)
 * - DON'T WRITE if empty slot (send_time_ms == 0)
 *
 * @param record RTT record to check
 * @param current_time_ms Current time (millis())
 * @return true if record should be written to CSV
 */
bool shouldWriteRecord(const RTTRecord* record, uint32_t current_time_ms) {
    // Empty slot - don't write
    if (record->send_time_ms == 0) {
        return false;
    }

    // Packet received - write immediately
    if (record->received) {
        return true;
    }

    // Packet not received - check if timed out
    return isTimedOut(record->send_time_ms, current_time_ms);
}

#endif // RTT_TIMEOUT_H
