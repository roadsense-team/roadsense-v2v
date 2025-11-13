#ifndef PACKAGE_MANAGER_H
#define PACKAGE_MANAGER_H

#include <Arduino.h>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include "../Data/Data.h"
#include "../Interfaces/Middleware/IConsoleMiddleware.h"
#include "../Interfaces/Helpers/MACHelper.h"
#include <WiFi.h>

// Structure to store package data with metadata
struct PackageData {
    Data data;                   // The actual data content
    unsigned long receivedTime;  // When this package was received
    uint8_t hopCount;            // Hop count at time of receipt
    bool processed;              // Flag to indicate if this data has been processed
};

class PackageManager {
public:
    std::vector<PackageData> getAllPackagesForMAC(const uint8_t* macAddress);
    // Define the callback type for processing packages
    using ProcessPackageCallback = std::function<void(const String&, const PackageData&)>;
    void resetProcessedStatus(unsigned long maxAge);

    // Constructor with optional console middleware
    PackageManager(IConsoleMiddleware* consoleMiddleware = nullptr);
    
    // Set a callback to be called when processing packages
    void setProcessPackageCallback(ProcessPackageCallback callback) {
        onProcessPackage = callback;
    }
    
    // Process all pending packages
    void processAllPendingPackages();
    
    // Set the default console for all instances that don't specify one
    static void setDefaultConsole(IConsoleMiddleware* consoleMiddleware);
    
    // Add a new package associated with a MAC address
    void addPackage(const uint8_t* macAddress, const Data& data, uint8_t hopCount);
    
    // Get the latest data for a specific MAC address
    PackageData* getLatestPackage(const uint8_t* macAddress);
    
    // Get all packages for analysis
    const std::map<String, std::vector<PackageData>>& getAllPackages() const;
    
    // Clean up old packages to prevent memory issues
    void cleanupOldPackages(unsigned long maxAge = MAX_AGE_PACKAG); // Default 60 seconds
    
    // Mark a package as processed
    void markProcessed(const uint8_t* macAddress);
    
    // Cleanup packages based on active MACs list
    int cleanupWithActiveList(const std::set<String>& activeMacs, unsigned long maxAge = MAX_AGE_PACKAG);
    
    // Get unprocessed packages for a specific MAC
    std::vector<PackageData> getUnprocessedPackages(const uint8_t* macAddress);
    
    // Get all unprocessed packages
    std::vector<std::pair<String, PackageData>> getAllUnprocessedPackages();
    
    // Print package statistics for debugging
    void printPackageStats();
    
    // Add a package for the local device
    void addLocalPackage(const Data& data);
    
    std::map<String, std::vector<PackageData>> getMyDataPackages();

private:
    // Module name for logging
    const String MODULE_NAME = "PackageManager";
    
    // Console middleware
    IConsoleMiddleware* console;
    
    // Static default console
    static IConsoleMiddleware* defaultConsole;
    
    // Callback for processing packages
    ProcessPackageCallback onProcessPackage = nullptr;
    
    // Using a map with MAC address string as key and a vector of packages as value
    std::map<String, std::vector<PackageData>> packages;
    
    // Helper to get unprocessed count for a package vector
    int getUnprocessedCount(const std::vector<PackageData>& pkgVector);
};

#endif // PACKAGE_MANAGER_H