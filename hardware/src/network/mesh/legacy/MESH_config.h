#ifndef MESH_CONFIG_H
#define MESH_CONFIG_H

#include <Arduino.h>
#include "../Constants/MESH_Constants.h"

// Network configuration structure
struct NetworkConfig {
    char ssid[32];         // Mesh network's SSID
    char password[64];     // Mesh network's password
    uint8_t channel;       // Mesh network's operating channel
    uint8_t maxLayer;      // Maximum allowed layer (hop count)
    uint8_t maxConnections; // Maximum number of connections
    
    // Constructor with defaults
    NetworkConfig() : 
        channel(MESH_CHANNEL),
        maxLayer(MESH_MAX_LAYER),
        maxConnections(MESH_MAX_CONNECTIONS) {
        strncpy(ssid, MESH_SSID, sizeof(ssid) - 1);
        ssid[sizeof(ssid) - 1] = '\0';
        strncpy(password, MESH_PASSWORD, sizeof(password) - 1);
        password[sizeof(password) - 1] = '\0';
    }
};

// Delay configuration structure
struct DelayConfig {
    unsigned long defaultDelay;  // Default delay value
    unsigned long shortDelay;    // Short delay value
    unsigned long midDelay;      // Medium delay value 
    unsigned long longDelay;     // Long delay value
    
    // Constructor with defaults
    DelayConfig() :
        defaultDelay(DEFAULT_DELAY),
        shortDelay(DEFAULT_SHORT_DELAY),
        midDelay(MID_DELAY),
        longDelay(LONGEST_DELAY) {}
};

// Interval configuration structure
struct IntervalConfig {
    unsigned long sendInterval;       // Interval between sending data
    unsigned long cleanupInterval;    // Interval between cleanup operations
    unsigned long processInterval;    // Interval between processing packages
    unsigned long meshStatusInterval; // Interval between sending mesh status
    unsigned long networkSnapshotInterval; // Interval for network snapshots
    unsigned long broadcastInterval;  // Interval for broadcasting
    
    // Constructor with defaults
    IntervalConfig() :
        sendInterval(DEFAULT_SEND_INTERVAL),
        cleanupInterval(DEFAULT_CLEANUP_INTERVAL),
        processInterval(DEFAULT_PROCESS_INTERVAL),
        meshStatusInterval(DEFAULT_MESH_STATUS_INTERVAL),
        networkSnapshotInterval(NETWORK_SNAPSHOT_INTERVAL),
        broadcastInterval(30000) {}   // 30 seconds default
};

// Timeout configuration structure
struct TimeoutConfig {
    unsigned long peerTimeout;     // Timeout for considering a peer inactive
    unsigned long packageTimeout;  // Timeout for considering a package expired
    unsigned long gpsReadTimeout;  // Timeout for GPS read operations
    unsigned long gpsCacheTimeout; // Timeout for GPS cache validity
    unsigned long nodeJoinWindow;  // Window for considering a node join as recent
    uint8_t maxHopCount;           // Maximum hop count for a package
    
    // Constructor with defaults
    TimeoutConfig() :
        peerTimeout(DEFAULT_PEER_TIMEOUT),
        packageTimeout(DEFAULT_PACKAGE_TIMEOUT),
        gpsReadTimeout(GPS_READ_TIMEOUT),
        gpsCacheTimeout(GPS_CACHE_TIMEOUT),
        nodeJoinWindow(10000),     // 10 seconds default
        maxHopCount(DEFAULT_MAX_HOP_COUNT) {}
};

// GPS configuration structure
struct GPSConfig {
    uint8_t rxPin;           // RX pin for GPS module
    uint8_t txPin;           // TX pin for GPS module
    unsigned long baudRate;  // Baud rate for GPS module
    uint8_t serialPort;      // Serial port for GPS module
    unsigned long shortDelay; // Short delay for GPS operations
    
    // Constructor with defaults
    GPSConfig() :
        rxPin(GPS_RX_PIN),
        txPin(GPS_TX_PIN),
        baudRate(NEO6M_BAUD_RATE),
        serialPort(GPS_SERIAL_PORT),
        shortDelay(SHORT_GPS_DELAY) {}
};

// Accelerometer configuration structure
struct AccelerometerConfig {
    uint8_t sdaPin;          // SDA pin for MPU6050
    uint8_t sclPin;          // SCL pin for MPU6050
    unsigned long shortDelay; // Short delay for accelerometer operations
    
    // Constructor with defaults
    AccelerometerConfig() :
        sdaPin(MPU6050_SDA_PIN),
        sclPin(MPU6050_SCL_PIN),
        shortDelay(ACC_SHORT_DELAY) {}
};

// Communication configuration structure
struct CommunicationConfig {
    unsigned long serialBaudRate;  // Serial baud rate
    uint8_t macAddressLength;      // MAC address length
    uint8_t messageTypeOriginal;   // Original message type
    uint8_t messageTypeRetransmission; // Retransmission message type
    uint8_t telemetryDataType;     // Telemetry data type identifier
    
    // Constructor with defaults
    CommunicationConfig() :
        serialBaudRate(SERIAL_BAUD_RATE),
        macAddressLength(MAC_ADDRESS_LENGTH),
        messageTypeOriginal(MESSAGE_TYPE_ORIGINAL),
        messageTypeRetransmission(MESSAGE_TYPE_RETRANSMISSION),
        telemetryDataType(MASSAGE_TYPE_TELEMETRY) {}
};

// Feature configuration structure
struct FeatureConfig {
    bool useRealSensors;                  // Whether to use real sensors
    bool logToFile;                       // Whether to log to file
    bool sendIndividualUpdates;           // Whether to send individual updates
    bool criticalEventsCanBypassConsolidation;  // Whether critical events can bypass consolidation
    size_t defaultBufferSize;             // Default buffer size
    size_t packageSize;                   // Package size
    unsigned int maxPeerDistance;         // Maximum distance for peers
    
    // Constructor with defaults
    FeatureConfig() :
        useRealSensors(USE_REAL_SENSORS),
        logToFile(LOG_TO_FILE),
        sendIndividualUpdates(SEND_INDIVIDUAL_UPDATES),
        criticalEventsCanBypassConsolidation(CRITICAL_EVENTS_BYPASS_CONSOLIDATION),
        defaultBufferSize(DEFAULT_BUFFER_SIZE),
        packageSize(PACKAGE_SIZE),
        maxPeerDistance(MAX_PEER_DISTANCE) {}
};

class MESH_Config {
public:

    static MESH_Config& getInstance() {
        Serial.println("MESH_Config singleton instance created");
        static MESH_Config instance;  // Guaranteed to be initialized only once
        return instance;
    }
    MESH_Config(const MESH_Config&) = delete;
    MESH_Config& operator=(const MESH_Config&) = delete;

    
    bool configure();
    
    // Network configuration getters/setters
    const char* getMeshSSID() const;
    const char* getMeshPassword() const;
    uint8_t getMeshChannel() const;
    uint8_t getMaxLayer() const;
    uint8_t getMaxConnections() const;
    void setMeshSSID(const char* ssid);
    void setMeshPassword(const char* password);
    void setMeshChannel(uint8_t channel);
    void setMaxLayer(uint8_t maxLayer);
    void setMaxConnections(uint8_t maxConnections);
    
    // Interval configuration getters/setters
    unsigned long getSendInterval() const;
    unsigned long getCleanupInterval() const;
    unsigned long getProcessInterval() const;
    unsigned long getMeshStatusInterval() const;
    unsigned long getNetworkSnapshotInterval() const;
    unsigned long getBroadcastInterval() const;
    void setSendInterval(unsigned long interval);
    void setCleanupInterval(unsigned long interval);
    void setProcessInterval(unsigned long interval);
    void setMeshStatusInterval(unsigned long interval);
    void setNetworkSnapshotInterval(unsigned long interval);
    void setBroadcastInterval(unsigned long interval);
    
    // Delay configuration getters/setters
    unsigned long getDefaultDelay() const;
    unsigned long getShortDelay() const;
    unsigned long getMidDelay() const;
    unsigned long getLongDelay() const;
    void setDefaultDelay(unsigned long delay);
    void setShortDelay(unsigned long delay);
    void setMidDelay(unsigned long delay);
    void setLongDelay(unsigned long delay);
    
    // Timeout configuration getters/setters
    unsigned long getPeerTimeout() const;
    unsigned long getPackageTimeout() const;
    unsigned long getGpsReadTimeout() const;
    unsigned long getGpsCacheTimeout() const;
    unsigned long getNodeJoinWindow() const;
    uint8_t getMaxHopCount() const;
    void setPeerTimeout(unsigned long timeout);
    void setPackageTimeout(unsigned long timeout);
    void setGpsReadTimeout(unsigned long timeout);
    void setGpsCacheTimeout(unsigned long timeout);
    void setNodeJoinWindow(unsigned long window);
    void setMaxHopCount(uint8_t hopCount);
    
    // GPS configuration getters/setters
    uint8_t getGpsRxPin() const;
    uint8_t getGpsTxPin() const;
    unsigned long getGpsBaudRate() const;
    uint8_t getGpsSerialPort() const;
    unsigned long getGpsShortDelay() const;
    void setGpsRxPin(uint8_t pin);
    void setGpsTxPin(uint8_t pin);
    void setGpsBaudRate(unsigned long baudRate);
    void setGpsSerialPort(uint8_t port);
    void setGpsShortDelay(unsigned long delay);
    
    // Accelerometer configuration getters/setters
    uint8_t getAccSdaPin() const;
    uint8_t getAccSclPin() const;
    unsigned long getAccShortDelay() const;
    void setAccSdaPin(uint8_t pin);
    void setAccSclPin(uint8_t pin);
    void setAccShortDelay(unsigned long delay);
    
    // Communication configuration getters/setters
    unsigned long getSerialBaudRate() const;
    uint8_t getMacAddressLength() const;
    uint8_t getMessageTypeOriginal() const;
    uint8_t getMessageTypeRetransmission() const;
    uint8_t getTelemetryDataType() const;
    void setSerialBaudRate(unsigned long baudRate);
    void setMacAddressLength(uint8_t length);
    void setMessageTypeOriginal(uint8_t type);
    void setMessageTypeRetransmission(uint8_t type);
    void setTelemetryDataType(uint8_t type);
    
    // Feature configuration getters/setters
    bool getUseRealSensors() const;
    bool getLogToFile() const;
    bool getSendIndividualUpdates() const;
    bool getCriticalEventsCanBypassConsolidation() const;
    size_t getDefaultBufferSize() const;
    size_t getPackageSize() const;
    unsigned int getMaxPeerDistance() const;
    void setUseRealSensors(bool use);
    void setLogToFile(bool log);
    void setSendIndividualUpdates(bool send);
    void setCriticalEventsCanBypassConsolidation(bool bypass);
    void setDefaultBufferSize(size_t size);
    void setPackageSize(size_t size);
    void setMaxPeerDistance(unsigned int distance);
    
    // Helper method to check if a node join is recent
    bool isNodeJoinRecent(unsigned long joinTime) const {
        return (millis() - joinTime < timeoutConfig.nodeJoinWindow);
    }
    
    // Get complete config objects
    const NetworkConfig& getNetworkConfig() const { return networkConfig; }
    const DelayConfig& getDelayConfig() const { return delayConfig; }
    const IntervalConfig& getIntervalConfig() const { return intervalConfig; }
    const TimeoutConfig& getTimeoutConfig() const { return timeoutConfig; }
    const GPSConfig& getGpsConfig() const { return gpsConfig; }
    const AccelerometerConfig& getAccConfig() const { return accConfig; }
    const CommunicationConfig& getCommunicationConfig() const { return commConfig; }
    const FeatureConfig& getFeatureConfig() const { return featureConfig; }
    
private:
    MESH_Config();

    NetworkConfig networkConfig;
    DelayConfig delayConfig;
    IntervalConfig intervalConfig;
    TimeoutConfig timeoutConfig;
    GPSConfig gpsConfig;
    AccelerometerConfig accConfig;
    CommunicationConfig commConfig;
    FeatureConfig featureConfig;
};

#endif // MESH_CONFIG_H