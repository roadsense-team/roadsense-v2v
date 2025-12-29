/**
 * @file DataLogger.h
 * @brief SD Card Data Logger for RoadSense V2V System
 *
 * TWO LOGGING MODES:
 * ==================
 *
 * MODE 1: Network Characterization (RTT Measurement)
 * ---------------------------------------------------
 * Purpose: Measure ESP-NOW latency, packet loss, jitter for emulator
 * Output:  v001_tx.csv (sent messages), v001_rx.csv (received messages)
 * Usage:   2 ESP32s, drive around varying distance, 20-30 min
 *
 * MODE 2: Training Data Collection (Ego-Perspective)
 * ---------------------------------------------------
 * Purpose: Collect real-world validation data for ML model
 * Output:  scenario_XXX.csv (received V2V messages from peers)
 * Usage:   3 ESP32s, convoy scenarios, after model trained
 *
 * File Format (Mode 1 - Network Characterization):
 * -------------------------------------------------
 * TX Log: timestamp_local_ms,msg_timestamp,vehicle_id,lat,lon,speed,heading,accel_x,accel_y,accel_z
 * RX Log: timestamp_local_ms,msg_timestamp,from_vehicle_id,lat,lon,speed,heading,accel_x,accel_y,accel_z
 *
 * File Format (Mode 2 - Training Data):
 * --------------------------------------
 * CSV Columns: timestamp_ms,vehicle_id,lat,lon,alt,speed,heading,
 *              long_accel,lat_accel,accel_x,accel_y,accel_z,
 *              gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,
 *              gps_valid,gps_age_ms
 */

#ifndef DATALOGGER_H
#define DATALOGGER_H

#include <Arduino.h>
#include <SdFat.h>
#include <Preferences.h>
#include "../config.h"
#include "../network/protocol/V2VMessage.h"
#include "../../include/IGpsSensor.h"

/**
 * @brief Logging mode selection
 */
enum LogMode {
    MODE_NETWORK_CHARACTERIZATION,  // Mode 1: Log TX/RX for ESP-NOW emulator (PRIMARY - do this first!)
    MODE_TRAINING_DATA             // Mode 2: Log ego-perspective for ML validation (FUTURE)
};

class DataLogger {
public:
    /**
     * @brief Constructor
     */
    DataLogger();

    /**
     * @brief Initialize SD card and NVS
     * @return true if successful, false otherwise
     */
    bool begin();

    // ========================================================================
    // MODE SELECTION
    // ========================================================================

    /**
     * @brief Set logging mode (must call before startLogging)
     * @param mode MODE_NETWORK_CHARACTERIZATION or MODE_TRAINING_DATA
     */
    void setMode(LogMode mode) { m_mode = mode; }

    /**
     * @brief Get current logging mode
     */
    LogMode getMode() const { return m_mode; }

    // ========================================================================
    // MODE 1: NETWORK CHARACTERIZATION (RTT Measurement)
    // ========================================================================

    /**
     * @brief Start network characterization logging
     * @param vehicleId Vehicle identifier (e.g., "V001")
     * @return true if TX and RX log files created successfully
     *
     * Creates two files:
     * - v001_tx.csv (all messages sent by this vehicle)
     * - v001_rx.csv (all messages received by this vehicle)
     */
    bool startCharacterizationLogging(const char* vehicleId);

    /**
     * @brief Log a transmitted V2V message (Mode 1)
     * @param msg Message that was just sent via ESP-NOW
     *
     * Call this AFTER sending message via ESP-NOW.
     * Logs: local_timestamp, msg.timestamp, msg.vehicleId, position, speed
     */
    void logTxMessage(const V2VMessage& msg);

    /**
     * @brief Queue a received V2V message for logging (Mode 1) - ISR SAFE
     * @param msg Message received from ESP-NOW callback
     *
     * Call this in ESP-NOW receive callback. Does NOT write to SD card.
     * Messages are stored in a lock-free circular buffer.
     * Call processRxQueue() from main loop to write to SD.
     */
    void queueRxMessage(const V2VMessage& msg);

    /**
     * @brief Process queued RX messages and write to SD card (Mode 1)
     *
     * Call this from main loop (NOT from ISR/callback).
     * Drains the RX queue and writes all pending messages to SD card.
     */
    void processRxQueue();

    /**
     * @brief Log a received V2V message (Mode 1) - INTERNAL USE ONLY
     * @param msg Message received from ESP-NOW callback
     *
     * WARNING: Do NOT call from ISR/callback - use queueRxMessage() instead.
     * This performs blocking SD card I/O.
     */
    void logRxMessage(const V2VMessage& msg);

    // ========================================================================
    // MODE 2: TRAINING DATA (Ego-Perspective) - FUTURE IMPLEMENTATION
    // ========================================================================

    /**
     * @brief Start logging session (Mode 2 - existing implementation)
     * @param vehicleId Vehicle identifier (e.g., "V001")
     * @param gpsReady True if GPS has valid fix (prevents logging without GPS)
     * @return true if file created successfully
     */
    bool startLogging(const char* vehicleId, bool gpsReady);

    /**
     * @brief Log a single V2V message sample (Mode 2 - existing implementation)
     * @param msg V2V message containing sensor data
     * @param gpsData GPS data with validity and age information
     */
    void logSample(const V2VMessage& msg, const IGpsSensor::GpsData& gpsData);

    /**
     * @brief Force flush buffer to SD card
     * @return true if successful, false if write failed
     */
    bool flush();

    // ========================================================================
    // COMMON METHODS
    // ========================================================================

    /**
     * @brief Stop logging and close all files
     * @return true if successful
     */
    bool stopLogging();

    /**
     * @brief Check if currently logging
     * @return true if logging active
     */
    bool isLogging() const { return m_isLogging; }

    /**
     * @brief Get current session number
     * @return Session counter
     */
    uint16_t getSessionNumber() const { return m_sessionNum; }

    /**
     * @brief Get total rows logged in current session
     * @return Row count
     */
    uint32_t getRowCount() const { return m_totalRows; }

private:
    // SdFat objects
    SdFat m_sd;

    // Mode 2: Single log file (existing)
    SdFile m_logFile;

    // Mode 1: TX and RX log files (new)
    SdFile m_txLogFile;
    SdFile m_rxLogFile;

    // State
    LogMode m_mode;
    bool m_isLogging;
    bool m_sdInitialized;

    // Session management
    Preferences m_nvs;
    uint16_t m_sessionNum;
    char m_filename[32];        // Mode 2 filename
    char m_txFilename[32];      // Mode 1 TX filename
    char m_rxFilename[32];      // Mode 1 RX filename

    // Buffering (Mode 2 - existing, for future use)
    char m_csvBuffer[LOG_BUFFER_ROWS * LOG_ROW_SIZE_BYTES];
    size_t m_bufferOffset;
    int m_bufferRowCount;
    uint32_t m_totalRows;
    uint32_t m_lastFlushTime;

    // Statistics (Mode 1 - new)
    uint32_t m_txCount;  // Messages sent
    uint32_t m_rxCount;  // Messages received

    // RX Message Queue (Mode 1 - ISR-safe circular buffer)
    static const size_t RX_QUEUE_SIZE = 32;  // Buffer ~3 seconds @ 10Hz
    V2VMessage m_rxQueue[RX_QUEUE_SIZE];
    volatile size_t m_rxQueueHead;  // Write position (ISR writes here)
    volatile size_t m_rxQueueTail;  // Read position (main loop reads here)

    // Error handling
    uint8_t m_consecutiveWriteFailures;

    // Private helper methods (Mode 2 - existing)
    bool createLogFile(const char* vehicleId);
    bool writeHeader();
    void buildCsvRow(const V2VMessage& msg, const IGpsSensor::GpsData& gpsData,
                     char* outBuffer, size_t bufferSize);
    bool ensureDirectoryExists();
    uint16_t loadSessionNumber();
    void incrementSessionNumber();

    // Private helper methods (Mode 1 - new)
    bool createCharacterizationFiles(const char* vehicleId);
    bool writeTxHeader();
    bool writeRxHeader();
};

#endif // DATALOGGER_H
