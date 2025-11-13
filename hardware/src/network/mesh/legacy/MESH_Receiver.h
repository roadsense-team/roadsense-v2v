#ifndef MESH_RECEIVER_H
#define MESH_RECEIVER_H

#include <Arduino.h>
#include "../../Interfaces/WiFi/IMESH_receiver_service.h"
#include "../../Interfaces/Middleware/IConsoleMiddleware.h"
#include "../../Data/Data.h"
#include "../../Config/MESH_config.h"
#include "../Senders/MESH_Sender.h"
#include "../../Managers/PackageManager.h"  // Add this include
#include "../../Managers/PeerManager.h"     // Add this include
#include "../../WiFi/Senders/SerialLayer2Sender.h"
// Define esp_err_t if not already defined elsewhere
typedef int esp_err_t;
class MESH_Receiver : public IMESH_receiver_service {
public:
    // Constructor with configuration and optional console\
    // NOT IN USE RIGHT NOW
    // MESH_Receiver(MESH_Config* meshConfig, IConsoleMiddleware* consoleMiddleware = nullptr);
    // Constructor with configuration only (uses default console if set)
    MESH_Receiver(MESH_Config& meshConfig, 
        PackageManager* pkgManager,
        PeerManager* peerMgr,
        ILayer2Sender* l2Sender,
        IMESH_sender_service* meshSenderService,
        IConsoleMiddleware* consoleMiddleware = nullptr);

    // Destructor
    ~MESH_Receiver();
    
    // Set the default console for all instances that don't specify one
    static void setDefaultConsole(IConsoleMiddleware* consoleMiddleware);
    
    // IMESH_receiver_service implementation
    bool begin() override;
    
    // this function is not used right now, but it is here for future use
    // i will use it for the logger
    void processReceivedData(const Data& data, const uint8_t* sender_mac) override;
    
    // Static callback for received data
    static void receivedCallback(const uint8_t *mac_addr, const uint8_t *data, int data_len);
    
private:
    // Module name for logging
    const String MODULE_NAME = "MESH_Receiver";
    
    // Configuration
    MESH_Config& meshConfig;
    
    // Console middleware
    IConsoleMiddleware* console;
    
    // Static default console
    static IConsoleMiddleware* defaultConsole;
    
    // Flag to track initialization status
    bool initialized;
    
    // Static instance for callback
    static MESH_Receiver* instance;
    
    // Internal method to handle received data
    void onDataReceived(const uint8_t* mac_addr, const uint8_t* data, int data_len);
};

#endif // MESH_RECEIVER_H