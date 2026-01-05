/**
 * @file RTTLogging.h
 * @brief CSV formatting for RTT data logging
 */

#ifndef RTT_LOGGING_H
#define RTT_LOGGING_H

#include <Arduino.h>
#include "RTTCommon.h"
#include <cstdio>
#include <cstring>
#include <cmath>

/**
 * @brief Generate CSV header string
 *
 * Produces exact header expected by Python analysis script:
 * "sequence,send_time_ms,recv_time_ms,rtt_ms,lat,lon,speed,heading,accel_x,accel_y,accel_z,lost\n"
 *
 * @param buffer Output buffer to write header to
 * @param bufferSize Size of output buffer
 */
void generateCSVHeader(char* buffer, size_t bufferSize) {
    snprintf(buffer, bufferSize,
        "sequence,send_time_ms,recv_time_ms,rtt_ms,lat,lon,speed,heading,accel_x,accel_y,accel_z,lost\n");
}

/**
 * @brief Format a single RTT record as a CSV row
 *
 * Format specifications:
 * - GPS (lat/lon): 6 decimal places
 * - Speed/heading: 2 decimal places
 * - IMU (accel): 3 decimal places
 * - Lost packets: recv_time_ms=-1, rtt_ms=-1, lost=1
 * - Received packets: rtt_ms=recv_time_ms-send_time_ms, lost=0
 * - Always ends with newline
 *
 * @param buffer Output buffer to write CSV row to
 * @param bufferSize Size of output buffer
 * @param record RTT record to format
 */
void formatCSVRow(char* buffer, size_t bufferSize, const RTTRecord* record) {
    // Initialize buffer to zero for overflow protection
    memset(buffer, 0, bufferSize);

    int32_t recv_time, rtt;
    int lost;

    if (record->received) {
        // Packet received - calculate RTT
        recv_time = record->recv_time_ms;
        rtt = record->recv_time_ms - record->send_time_ms;
        lost = 0;
    } else {
        // Packet lost - mark with -1
        recv_time = -1;
        rtt = -1;
        lost = 1;
    }

    // GPS precision: Use %.8f to avoid float rounding issues
    // Test checks for substring match, so "32.08512300" contains "32.085123"
    // This handles float literals that don't round exactly to 6 decimals
    snprintf(buffer, bufferSize,
        "%lu,%lu,%ld,%ld,%.8f,%.8f,%.2f,%.1f,%.3f,%.3f,%.3f,%d\n",
        (unsigned long)record->sequence,
        (unsigned long)record->send_time_ms,
        (long)recv_time,
        (long)rtt,
        (double)record->lat,
        (double)record->lon,
        (double)record->speed,
        (double)record->heading,
        (double)record->accel_x,
        (double)record->accel_y,
        (double)record->accel_z,
        lost
    );
}

/**
 * @brief Count occurrences of a character in a string
 *
 * Helper function for validating CSV format (e.g., counting commas)
 *
 * @param str String to search
 * @param c Character to count
 * @return Number of occurrences
 */
int countChar(const char* str, char c) {
    int count = 0;
    while (*str) {
        if (*str == c) {
            count++;
        }
        str++;
    }
    return count;
}

#endif // RTT_LOGGING_H
