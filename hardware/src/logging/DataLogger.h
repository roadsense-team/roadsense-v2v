/**
 * @file DataLogger.h
 * @brief SD Card Data Logger for RoadSense V2V System
 *
 * Features:
 * - CSV format logging (compatible with Python pandas)
 * - Row buffering (10 rows = 1 second at 10Hz)
 * - Session counter (persistent in NVS)
 * - GPS age tracking (gps_valid, gps_age_ms columns)
 * - Magnetometer placeholder (zeros until QMC5883L integrated)
 *
 * File Format: V001_session_001.csv
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

    /**
     * @brief Start logging session
     * @param vehicleId Vehicle identifier (e.g., "V001")
     * @param gpsReady True if GPS has valid fix (prevents logging without GPS)
     * @return true if file created successfully
     */
    bool startLogging(const char* vehicleId, bool gpsReady);

    /**
     * @brief Log a single V2V message sample
     * @param msg V2V message containing sensor data
     * @param gpsData GPS data with validity and age information
     */
    void logSample(const V2VMessage& msg, const IGpsSensor::GpsData& gpsData);

    /**
     * @brief Force flush buffer to SD card
     * @return true if successful, false if write failed
     */
    bool flush();

    /**
     * @brief Stop logging and close file
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
    SdFile m_logFile;

    // State
    bool m_isLogging;
    bool m_sdInitialized;

    // Session management
    Preferences m_nvs;
    uint16_t m_sessionNum;
    char m_filename[32];

    // Buffering (fixed size - no heap allocation)
    char m_csvBuffer[LOG_BUFFER_ROWS * LOG_ROW_SIZE_BYTES];
    size_t m_bufferOffset;      // Current write position in buffer
    int m_bufferRowCount;
    uint32_t m_totalRows;
    uint32_t m_lastFlushTime;

    // Error handling
    uint8_t m_consecutiveWriteFailures;

    // Private helper methods
    bool createLogFile(const char* vehicleId);
    bool writeHeader();
    void buildCsvRow(const V2VMessage& msg, const IGpsSensor::GpsData& gpsData,
                     char* outBuffer, size_t bufferSize);
    bool ensureDirectoryExists();
    uint16_t loadSessionNumber();
    void incrementSessionNumber();
};

#endif // DATALOGGER_H
