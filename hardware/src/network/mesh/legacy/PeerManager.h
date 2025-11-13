#ifndef PEER_MANAGER_H
#define PEER_MANAGER_H

#include <Arduino.h>
#include <vector>
#include "../Data/Data.h"
#include "../Interfaces/Middleware/IConsoleMiddleware.h"
#include "../Helpers/Helpers.h"
#include "../Config/MESH_config.h"
// Define maximum distance threshold (if you're calculating distances)
class Helper;
// Struct to track peer information
struct PeerInfo {
    uint8_t macAddress[MAC_ADDRESS_LENGTH];       // MAC address of the peer
    unsigned long lastSeen;      // Timestamp of last message from this peer
    Data lastData;               // Most recent data received from this peer
    uint8_t hopCount;            // Current hop count to reach this peer
    float distance;              // Estimated distance to peer (if implemented)
    bool isActive;               // Flag to indicate if peer is currently active
};
struct PeerConnection {
    String fromMac;
    String toMac;
    bool isDirectPeer;
    int hopCount;
    int lastSeenSeconds;
};
struct TrackedPeer {
    bool wasPresent;
    unsigned long lastSeenTime;
};

class PeerManager {
public:
    // Then add these method declarations to the public section of the PeerManager class:
    // Network statistics methods
    int getDirectPeerCount() const;
    int getIndirectPeerCount() const;
    int getTotalMessageCount() const;
    int getNetworkDiameter() const;
    std::vector<PeerConnection> getAllPeerConnections() const;
    // Constructor with optional console middleware
    PeerManager(IConsoleMiddleware* consoleMiddleware = nullptr);
    
    // Set the default console for all instances that don't specify one
    static void setDefaultConsole(IConsoleMiddleware* consoleMiddleware);
    
    // Add or update a peer in the network
    void updatePeer(const uint8_t* macAddress, const Data& data, uint8_t hopCount);
    
    // Remove inactive peers that haven't been seen for a while
    void cleanupInactivePeers();
    
    // Get a specific peer by MAC address
    PeerInfo* getPeer(const uint8_t* macAddress);
    
    // Get all active peers
    const std::vector<PeerInfo>& getActivePeers();
    
    // Should we forward a message from this peer?
    bool shouldForwardMessage(const uint8_t* macAddress, uint8_t hopCount);
    
    // Get number of active peers
    size_t getActivePeerCount() const; // Add 'const' keyword here    
    // Print peer list to console
    void printPeerList();
    
    // Get total peer count
    int getPeerCount();
    // Get total peer count including inactive peers
    bool checkForNewPeers(int lastNodeJoinTime);
private:
    // Module name for logging
    const String MODULE_NAME = "PeerManager";
    
    MESH_Config& meshConfig = MESH_Config::getInstance();

    static std::map<String, TrackedPeer> knownPeers;

    // Console middleware
    IConsoleMiddleware* console;
    
    Helper& helper = Helper::getInstance();

    // Static default console
    static IConsoleMiddleware* defaultConsole;
    
    std::vector<PeerInfo> peers;
    
    // Calculate distance between peers (if GPS data is available)
    float calculateDistance(const GPSData& gps1, const GPSData& gps2);
    
    // Removed macToString and compareMac methods as we'll use MACHelper instead
};

#endif // PEER_MANAGER_H