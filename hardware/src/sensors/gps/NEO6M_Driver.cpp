/**
 * @file NEO6M_Driver.cpp
 * @brief NEO-6M GPS module driver implementation
 *
 * Based on legacy RoadSense NEO6M_GPS_Service with improvements:
 * - Non-blocking operation (removed delay() and timeout loops)
 * - GPS heading extraction (replaces magnetometer - MPU6500 has no compass)
 * - Enhanced quality indicators (satellites, HDOP)
 * - 4-state cache logic (fresh/cached/stale/none)
 * - Direct Logger singleton usage (removed middleware)
 */

#include "NEO6M_Driver.h"

/**
 * Constructor
 *
 * ✅ USER REFINEMENT: Use GPS_SERIAL_PORT constant (value = 1)
 * This initializes Serial1 (UART1), which will be remapped to
 * GPIO 16/17 when begin() is called.
 */
NEO6M_Driver::NEO6M_Driver()
    : gpsSerial(GPS_SERIAL_PORT)  // ✅ Value = 1 (Serial1, not hardcoded)
    , initialized(false)
    , everHadFix(false)
    , lastValidReadingTime(0)
{
    // Initialize cache to zeros
    lastValidReading = GpsData();
}

/**
 * Initialize GPS module
 *
 * ✅ USER REFINEMENT: Non-blocking (returns immediately)
 * ✅ Legacy used GPS_SERIAL_PORT = 1 (Serial1)
 */
bool NEO6M_Driver::begin() {
    Logger& log = Logger::getInstance();

    log.info("NEO6M", "Initializing GPS module...");

    // Initialize Serial1 (UART1) remapped to GPIO 16/17
    // ✅ Pins from config.h: GPS_RX_PIN=16, GPS_TX_PIN=17, GPS_BAUD_RATE=9600
    gpsSerial.begin(GPS_BAUD_RATE, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

    // Check if serial port opened successfully
    if (!gpsSerial) {
        log.error("NEO6M", "Failed to initialize GPS serial");
        return false;
    }

    initialized = true;
    everHadFix = false;
    lastValidReadingTime = 0;

    log.info("NEO6M", "GPS initialized (9600 baud, Serial1 @ GPIO 16/17)");
    log.warning("NEO6M", "⚠️  Cold start: 30-60s to first fix");

    // ✅ KEY CHANGE: Don't wait for fix - let update() handle it asynchronously
    return true;
}

/**
 * Poll UART and update GPS parser
 *
 * ✅ USER REFINEMENT: Non-blocking (no delay(), no timeout loop)
 * CRITICAL: Call every loop iteration (~100Hz) to drain UART buffer
 */
bool NEO6M_Driver::update() {
    if (!initialized) {
        return false;
    }

    bool newData = false;

    // Drain UART buffer (non-blocking)
    // NEO-6M sends ~960 bytes/sec @ 9600 baud
    // At 100Hz loop rate, we read ~10 bytes/iteration
    while (gpsSerial.available() > 0) {
        char c = gpsSerial.read();
        if (gps.encode(c)) {
            // TinyGPS++ parsed a complete NMEA sentence
            newData = true;
        }
    }

    // Update cache if new valid location
    if (newData && gps.location.isValid()) {
        updateCache();
    }

    return newData;
}

/**
 * Update internal cache when new valid GPS fix arrives
 *
 * ✅ USER REFINEMENT: Set cached=false, timestamp=millis()
 * ✅ USER REFINEMENT: Don't overwrite heading if invalid (keep previous)
 */
void NEO6M_Driver::updateCache() {
    // Extract all fields from TinyGPS++
    lastValidReading.latitude = gps.location.lat();
    lastValidReading.longitude = gps.location.lng();
    lastValidReading.altitude = gps.altitude.meters();

    // Speed (always update if location valid)
    lastValidReading.speed = gps.speed.mps();  // m/s

    // ✅ Heading validation (3 checks per user refinement)
    if (gps.course.isValid() &&                          // Checksum valid
        gps.course.age() < 2000 &&                       // Updated recently
        lastValidReading.speed >= GPS_HEADING_MIN_SPEED_MPS) {  // Moving fast enough

        lastValidReading.heading = gps.course.deg();  // Update heading
    }
    // ✅ If heading invalid: Keep previous value (don't overwrite)

    // Quality indicators
    // ✅ Clamp satellites to uint8_t range (0-255)
    lastValidReading.satellites = (uint8_t)min((uint32_t)255, gps.satellites.value());
    lastValidReading.hdop = gps.hdop.hdop();

    // Metadata
    lastValidReading.valid = true;
    lastValidReading.cached = false;  // ✅ Fresh data, not from cache
    lastValidReading.timestamp = millis();
    lastValidReadingTime = millis();

    // Track first fix
    if (!everHadFix) {
        everHadFix = true;
        Logger::getInstance().info("NEO6M", "✅ First GPS fix acquired!");
    }
}

/**
 * Read cached GPS data (always non-blocking)
 *
 * ✅ USER REFINEMENT: Cache state semantics
 * - <2s: valid=true, cached=false (fresh)
 * - 2-30s: valid=true, cached=true (cached but valid)
 * - >30s: valid=false, cached=true (stale/expired)
 * - Never had fix: valid=false, cached=false (zeros)
 */
IGpsSensor::GpsData NEO6M_Driver::read() {
    if (!initialized) {
        Logger::getInstance().error("NEO6M", "read() called before begin()!");
        return GpsData();  // ✅ Zeros with valid=false, cached=false
    }

    GpsData data = lastValidReading;  // Start with cached data
    uint32_t age = millis() - lastValidReadingTime;

    // ✅ Handle different cache states
    if (lastValidReadingTime == 0) {
        // Never had fix: valid=false, cached=false
        data.valid = false;
        data.cached = false;
    } else if (age > GPS_CACHE_TIMEOUT_MS) {
        // ✅ Expired (>30s): valid=false, cached=true
        data.valid = false;
        data.cached = true;

        static uint32_t lastWarning = 0;
        if (millis() - lastWarning > 10000) {
            Logger::getInstance().warning("NEO6M",
                "GPS cache expired (age: " + String(age / 1000) + "s)");
            lastWarning = millis();
        }
    } else if (age > 2000) {
        // ✅ Old but valid (2s-30s): valid=true, cached=true
        data.valid = true;
        data.cached = true;
    }
    // ✅ Fresh (<2s): valid=true, cached=false (from updateCache())

    return data;
}

/**
 * Check if GPS has ever achieved a fix
 */
bool NEO6M_Driver::hasEverHadFix() const {
    return everHadFix;
}
