#ifndef DATA_H
#define DATA_H

#include "../Interfaces/MockData/IGPS_mock_generator_service.h"
#include "../Interfaces/MockData/IACC_mock_generator_service.h"
#include "../Interfaces/Helpers/MACHelper.h"
#include "../Factorys/GPSMockGeneratorFactory.h"
#include "../Interfaces/Middleware/IConsoleMiddleware.h"
#include "../Interfaces/RealData/IAngleService.h"
#include "../Interfaces/RealData/ISpeedService.h"
#include <cstdint>
#include "../Config/MESH_config.h"

// Forward declarations to prevent circular dependencies
class IGPSSensor;
class IAccelerometerSensor;
class ISpeedService;
class IAngleService;

// Complete data structure to be sent
struct Data {
    GPSData                 gps;
    AccelerometerData       accel;
    unsigned long           timestamp;
    uint8_t                 sourceMAC[MAC_ADDRESS_LENGTH];    // Original source MAC
    uint8_t                 lastForwardingNode[MAC_ADDRESS_LENGTH]; // Physical sender's MAC
    uint8_t                 messageType;
    uint8_t                 hopCount;
    bool                    isRealSensorData;   // flage to indicate if data is from real sensors or mock
    unsigned long           angle;
    unsigned int            speed;

    // MAC address methods
    String getSourceMACString() const;
    void setSourceMAC(const uint8_t* mac);
    bool isSameSource(const uint8_t* mac) const;
    bool isSameSource(const Data& other) const;
    
    // Conversion to string
    String toString() const;
    
    // Getters and setters for angle and speed
    unsigned long getAngle() const { return angle; }
    void setAngle(unsigned long newAngle) { angle = newAngle; }
    
    unsigned int getSpeed() const { return speed; }
    void setSpeed(unsigned int newSpeed) { speed = newSpeed; }
    
    // Constructor
    Data() : messageType(MESSAGE_TYPE_ORIGINAL), hopCount(0), isRealSensorData(false), angle(0), speed(0) {
        memset(sourceMAC, 0, MAC_ADDRESS_LENGTH);
        memset(lastForwardingNode, 0, MAC_ADDRESS_LENGTH);
    }
};

// Data generator that composes specialized generators
class DataGenerator {
private:
    // Mock data generators
    IGPS_mock_generator_service* gpsGenerator;
    IACC_mock_generator_service* accelGenerator;
    
    // Real sensors
    IGPSSensor* gpsRealSensor;
    IAccelerometerSensor* accelerometerRealSensor;
    
    // Angle and speed services
    IAngleService* angleService;
    ISpeedService* speedService;
    
    // Previous GPS data for calculating speed and angle
    GPSData previousGPS;
    unsigned long previousTimestamp;
    
    bool ownGenerators;
    bool useRealSensors;  // Flag to indicate if we're using real sensors
    
    // Add console middleware
    IConsoleMiddleware* console;
    static IConsoleMiddleware* defaultConsole;
    const String MODULE_NAME = "DataGenerator";

public:
    // Default constructor - defaults to mock generators
    DataGenerator(IConsoleMiddleware* consoleMiddleware = nullptr);
    
    // Constructor with useRealSensors flag - automatically creates appropriate generators
    DataGenerator(bool useRealSensors, IConsoleMiddleware* consoleMiddleware = nullptr);
    
    // Overloaded constructor that uses a specific GPS generator type
    DataGenerator(GPSGeneratorType type, IConsoleMiddleware* consoleMiddleware = nullptr);
    
    // Constructor for real sensors
    DataGenerator(IGPSSensor* gpsSensor, IAccelerometerSensor* accelSensor,
                 IConsoleMiddleware* consoleMiddleware = nullptr);

    // Constructor that uses external generator instances
    DataGenerator(IGPS_mock_generator_service* gpsGen, IACC_mock_generator_service* accelGen,
                 IConsoleMiddleware* consoleMiddleware = nullptr);

    // Destructor
    ~DataGenerator();
    
    // Static method to set default console
    static void setDefaultConsole(IConsoleMiddleware* consoleMiddleware);

    // Generate complete data packet
    Data generateData();
    
    // Check if using real sensors
    void setUseRealSensors(bool useReal);
};

#endif // DATA_H