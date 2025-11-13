#ifndef CLEANUP_MANAGER_H
#define CLEANUP_MANAGER_H

#include <Arduino.h>
#include <set>
#include "PeerManager.h"
#include "PackageManager.h"
#include "../Interfaces/Middleware/IConsoleMiddleware.h"
#include "../Interfaces/Helpers/MACHelper.h" 

class CleanupManager {
public:
    // Constructor with required managers and optional console middleware
    CleanupManager(PeerManager* peerManager, PackageManager* packageManager, 
                  IConsoleMiddleware* consoleMiddleware = nullptr);
    
    // Set the default console for all instances that don't specify one
    static void setDefaultConsole(IConsoleMiddleware* consoleMiddleware);
    
    // Set timeout values
    void setPeerTimeout(unsigned long timeout);
    void setPackageTimeout(unsigned long timeout);
    
    // Perform synchronized cleanup across all managers
    void performCleanup();
    
    // Print cleanup statistics
    void printCleanupStats();
    
private:
    // Module name for logging
    const String MODULE_NAME = "CleanupManager";
    
    // Required managers
    PeerManager* peerManager;
    PackageManager* packageManager;
    
    // Console middleware
    IConsoleMiddleware* console;
    
    // Static default console
    static IConsoleMiddleware* defaultConsole;
    
    // Timeout settings
    unsigned long peerTimeout;       // Timeout for inactive peers
    unsigned long packageTimeout;    // Timeout for old packages
    
    // Statistics
    unsigned long lastCleanupTime;
    int lastPeersRemoved;
    int lastPackagesRemoved;
    
};

#endif // CLEANUP_MANAGER_H