#ifndef MESH_SENDER_H
#define MESH_SENDER_H

#include <Arduino.h>
#include ".../../Interfaces/WiFi/IMESH_sender_service.h"
#include "../../Interfaces/WiFi/ISender_service.h"
#include "../../Interfaces/MockData/IGPS_mock_generator_service.h"
#include "../../Interfaces/MockData/IACC_mock_generator_service.h"
#include "../../Interfaces/Middleware/IConsoleMiddleware.h"
#include "../../Interfaces/Middleware/IDelayMiddleware.h"
#include "../../Data/Data.h"
#include "../../Config/MESH_config.h"
#include "../../Managers/PackageManager.h"
#include "../../Managers/PeerManager.h"
// Define esp_err_t to avoid dependency on esp_mesh.h
typedef int esp_err_t;

// Message type definitions
#define MESSAGE_TYPE_ORIGINAL 0
#define MESSAGE_TYPE_RETRANSMISSION 1

// MESH sender - Implements both general sender interface and MESH-specific interface
class MESH_Sender : public ISender_service, public IMESH_sender_service {
public:

    // NOT IN THE USE RIGHT NOW
    // Constructor with configuration, data generators, and custom console
    // MESH_Sender(MESH_Config* meshConfig, IGPS_mock_generator_service* gpsGen, 
    //            IACC_mock_generator_service* accelGen, IConsoleMiddleware* consoleMiddleware = nullptr);
    MESH_Sender(MESH_Config* meshConfig, IConsoleMiddleware* consoleMiddleware);
    // IN USE RIGHT NOW
    MESH_Sender(MESH_Config& meshConfig,IConsoleMiddleware* consoleMiddleware,IDelayMiddleware* IdelayMiddleware,
        PackageManager* pkgManager,
        PeerManager* peerManager);
    // *************************************************************
    // USE IT NOW FOR SINGLETON
    MESH_Sender(MESH_Config& meshConfig, IConsoleMiddleware* consoleMiddleware = nullptr);

    // Destructor
    ~MESH_Sender();
    
    static void setDefaultConsole(IConsoleMiddleware* consoleMiddleware);
    bool begin() override;
    bool sendData(const Data& data) override;
    bool generateAndSendData() override;
    DataGenerator* getDataGenerator() const;
    static void sentCallback(const uint8_t *mac_addr, esp_err_t status);
    void broadcastAllStoredPackages();
    void forwardStoredPackages(int lastNodeJoinTime);

private:
    // Configuration
    MESH_Config& meshConfig;
    IDelayMiddleware* dealayMiddleware;

    // Data generators
    IGPS_mock_generator_service* gpsGenerator;
    IACC_mock_generator_service* accelGenerator;
    DataGenerator* dataGenerator;
    bool ownGenerators;
    
    // Console middleware
    IConsoleMiddleware* console;
    
    // Mesh network parameters
    uint8_t broadcastAddress[MAC_ADDRESS_LENGTH] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    bool initialized;
    
    // Static reference for callbacks
    static MESH_Sender* instance;
    
    // Static default console
    static IConsoleMiddleware* defaultConsole;
    
    // Callback implementation
    void onDataSent(const uint8_t* mac_addr, esp_err_t status);

};

#endif // MESH_SENDER_H