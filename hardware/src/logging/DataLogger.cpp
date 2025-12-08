/**
 * @file DataLogger.cpp
 * @brief SD Card Data Logger Implementation
 */

#include "DataLogger.h"
#include "../utils/Logger.h"
#include <SPI.h>

// Constructor
DataLogger::DataLogger()
    : m_isLogging(false),
      m_sdInitialized(false),
      m_sessionNum(0),
      m_bufferOffset(0),
      m_bufferRowCount(0),
      m_totalRows(0),
      m_lastFlushTime(0),
      m_consecutiveWriteFailures(0) {
    memset(m_filename, 0, sizeof(m_filename));
    memset(m_csvBuffer, 0, sizeof(m_csvBuffer));
}

// Initialize SD card and NVS
bool DataLogger::begin() {
    Logger& log = Logger::getInstance();

    // 1. Initialize NVS for session counter
    if (!m_nvs.begin(NVS_NAMESPACE, false)) {
        log.error("DataLogger", "Failed to open NVS namespace");
        return false;
    }
    m_sessionNum = loadSessionNumber();
    log.info("DataLogger", "Session counter loaded: " + String(m_sessionNum));

    // 2. Initialize SPI for SD card
    SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);

    // 3. Initialize SD card
    // NOTE: Using conservative 10 MHz instead of config's 25 MHz
    // SdFat v2 is strict about initialization speed - many SD cards fail at 25 MHz
    log.info("DataLogger", "Attempting SD card init with CS=" + String(SD_CS_PIN) + ", SPI=10 MHz");

    if (!m_sd.begin(SD_CS_PIN, SD_SCK_MHZ(10))) {
        log.error("DataLogger", "SD card initialization failed");
        log.error("DataLogger", "Check: 1) Card inserted, 2) Formatted FAT32, 3) Wiring");
        log.error("DataLogger", "SdFat error code: " + String(m_sd.sdErrorCode()));
        log.error("DataLogger", "SdFat error data: " + String(m_sd.sdErrorData()));

        // Try one more time with even slower speed (4 MHz)
        delay(100);
        log.info("DataLogger", "Retrying with ultra-safe 4 MHz...");
        if (!m_sd.begin(SD_CS_PIN, SD_SCK_MHZ(4))) {
            log.error("DataLogger", "SD card init failed even at 4 MHz - hardware issue?");
            return false;
        }
        log.info("DataLogger", "✅ Success at 4 MHz (slower but stable)");
    }

    m_sdInitialized = true;

    // 3. Get card info
    uint32_t cardSizeMB = m_sd.card()->sectorCount() / 2048;
    log.info("DataLogger", "✅ SD card OK: " + String(cardSizeMB) + " MB");

    // 4. Ensure directory exists
    if (!ensureDirectoryExists()) {
        log.error("DataLogger", "Failed to create log directory");
        return false;
    }

    return true;
}

// Start logging session
bool DataLogger::startLogging(const char* vehicleId, bool gpsReady) {
    Logger& log = Logger::getInstance();

    if (!m_sdInitialized) {
        log.error("DataLogger", "SD card not initialized");
        return false;
    }

    if (m_isLogging) {
        log.warning("DataLogger", "Already logging");
        return false;
    }

    if (!gpsReady) {
        log.warning("DataLogger", "GPS not ready - waiting for fix");
        return false;
    }

    // Increment session number
    incrementSessionNumber();

    // Create log file
    if (!createLogFile(vehicleId)) {
        return false;
    }

    // Write CSV header
    if (!writeHeader()) {
        m_logFile.close();
        return false;
    }

    // Reset state
    memset(m_csvBuffer, 0, sizeof(m_csvBuffer)); // Clear buffer
    m_bufferOffset = 0;
    m_bufferRowCount = 0;
    m_totalRows = 0;
    m_lastFlushTime = millis();
    m_consecutiveWriteFailures = 0;
    m_isLogging = true;

    log.info("DataLogger", "✅ Logging started: " + String(m_filename));
    return true;
}

// Log a single sample
void DataLogger::logSample(const V2VMessage& msg, const IGpsSensor::GpsData& gpsData) {
    if (!m_isLogging) {
        return;
    }

    // Build CSV row (into temporary buffer)
    char rowBuffer[LOG_ROW_SIZE_BYTES];
    buildCsvRow(msg, gpsData, rowBuffer, sizeof(rowBuffer));

    // Append to main buffer using memcpy (zero heap allocation)
    size_t rowLen = strlen(rowBuffer);
    if (m_bufferOffset + rowLen < sizeof(m_csvBuffer)) {
        memcpy(m_csvBuffer + m_bufferOffset, rowBuffer, rowLen);
        m_bufferOffset += rowLen;
        m_bufferRowCount++;
        m_totalRows++;
    } else {
        // Buffer overflow protection (should never happen)
        Logger::getInstance().error("DataLogger", "Buffer overflow! Forcing flush.");
        flush();
        return;
    }

    // Auto-flush if buffer full OR 1 second elapsed
    if (m_bufferRowCount >= LOG_BUFFER_ROWS ||
        (millis() - m_lastFlushTime >= LOG_FLUSH_INTERVAL_MS)) {
        flush();
    }
}

// Flush buffer to SD card
bool DataLogger::flush() {
    if (!m_isLogging || m_bufferRowCount == 0) {
        return true; // Nothing to flush
    }

    // Write buffer to file
    size_t written = m_logFile.write(m_csvBuffer, m_bufferOffset);

    if (written != m_bufferOffset) {
        m_consecutiveWriteFailures++;
        Logger::getInstance().error("DataLogger",
            "Write failed: " + String(written) + "/" + String(m_bufferOffset) +
            " (Failure " + String(m_consecutiveWriteFailures) + "/" + String(MAX_WRITE_FAILURES) + ")");

        // Auto-stop after multiple failures (SD card full or hardware error)
        if (m_consecutiveWriteFailures >= MAX_WRITE_FAILURES) {
            Logger::getInstance().error("DataLogger", "❌ SD CARD FULL or WRITE ERROR - Logging stopped automatically");
            stopLogging();
        }
        return false;
    }

    // Sync to SD card (force physical write)
    if (!m_logFile.sync()) {
        Logger::getInstance().warning("DataLogger", "Sync failed (data may be cached)");
    }

    // Reset buffer and error counter
    memset(m_csvBuffer, 0, sizeof(m_csvBuffer));
    m_bufferOffset = 0;
    m_bufferRowCount = 0;
    m_lastFlushTime = millis();
    m_consecutiveWriteFailures = 0;  // Reset on success

    return true;
}

// Stop logging
bool DataLogger::stopLogging() {
    if (!m_isLogging) {
        return true;
    }

    // Flush remaining buffer
    flush();

    // Close file
    m_logFile.close();
    m_isLogging = false;

    Logger::getInstance().info("DataLogger", "✅ Logging stopped: " + String(m_totalRows) + " rows");
    return true;
}

// ============================================================================
// PRIVATE HELPER METHODS
// ============================================================================

// Create log file
bool DataLogger::createLogFile(const char* vehicleId) {
    // Generate filename: V001_session_001.csv
    snprintf(m_filename, sizeof(m_filename), "%s/%s_session_%03d.csv",
             LOG_DIRECTORY, vehicleId, m_sessionNum);

    // Open file for writing (create if doesn't exist)
    if (!m_logFile.open(m_filename, O_WRONLY | O_CREAT | O_TRUNC)) {
        Logger::getInstance().error("DataLogger", "Failed to create file: " + String(m_filename));
        return false;
    }

    return true;
}

// Write CSV header
bool DataLogger::writeHeader() {
    const char* header =
        "timestamp_ms,vehicle_id,lat,lon,alt,speed,heading,"
        "long_accel,lat_accel,"
        "accel_x,accel_y,accel_z,"
        "gyro_x,gyro_y,gyro_z,"
        "mag_x,mag_y,mag_z,"
        "gps_valid,gps_age_ms\n";

    size_t written = m_logFile.write(header, strlen(header));

    if (written != strlen(header)) {
        Logger::getInstance().error("DataLogger", "Failed to write header");
        return false;
    }

    m_logFile.sync();
    return true;
}

// Build CSV row (writes directly to output buffer - zero heap allocation)
void DataLogger::buildCsvRow(const V2VMessage& msg, const IGpsSensor::GpsData& gpsData,
                              char* outBuffer, size_t bufferSize) {
    // Calculate GPS age (millis() - last GPS update time)
    // Always report real age, regardless of cached flag
    uint32_t gps_age_ms = millis() - gpsData.timestamp;

    snprintf(outBuffer, bufferSize,
        "%lu,%s,%.6f,%.6f,%.1f,%.2f,%.1f,"   // timestamp, vehicleId, lat, lon, alt, speed, heading
        "%.2f,%.2f,"                          // long_accel, lat_accel
        "%.2f,%.2f,%.2f,"                     // accel_x, accel_y, accel_z
        "%.3f,%.3f,%.3f,"                     // gyro_x, gyro_y, gyro_z
        "%.2f,%.2f,%.2f,"                     // mag_x, mag_y, mag_z (zeros)
        "%d,%lu\n",                           // gps_valid, gps_age_ms

        // Timestamp & ID
        msg.timestamp,
        msg.vehicleId,

        // Position
        msg.position.lat,
        msg.position.lon,
        msg.position.alt,

        // Dynamics
        msg.dynamics.speed,
        msg.dynamics.heading,
        msg.dynamics.longAccel,
        msg.dynamics.latAccel,

        // Raw Sensors
        msg.sensors.accel[0], msg.sensors.accel[1], msg.sensors.accel[2],
        msg.sensors.gyro[0], msg.sensors.gyro[1], msg.sensors.gyro[2],
        msg.sensors.mag[0], msg.sensors.mag[1], msg.sensors.mag[2],  // Zeros (no magnetometer)

        // GPS Metadata
        gpsData.valid ? 1 : 0,
        gps_age_ms
    );

    // Safety check for buffer overflow
    if (strlen(outBuffer) >= bufferSize - 1) {
        Logger::getInstance().error("DataLogger", "CSV row overflow! Increase LOG_ROW_SIZE_BYTES");
    }
}

// Ensure log directory exists
bool DataLogger::ensureDirectoryExists() {
    if (!m_sd.exists(LOG_DIRECTORY)) {
        if (!m_sd.mkdir(LOG_DIRECTORY)) {
            Logger::getInstance().error("DataLogger", "Failed to create directory: " + String(LOG_DIRECTORY));
            return false;
        }
        Logger::getInstance().info("DataLogger", "Created directory: " + String(LOG_DIRECTORY));
    }
    return true;
}

// Load session number from NVS
uint16_t DataLogger::loadSessionNumber() {
    return m_nvs.getUShort(NVS_SESSION_KEY, 0);
}

// Increment and save session number
void DataLogger::incrementSessionNumber() {
    m_sessionNum++;
    m_nvs.putUShort(NVS_SESSION_KEY, m_sessionNum);
    Logger::getInstance().info("DataLogger", "Session number incremented: " + String(m_sessionNum));
}
