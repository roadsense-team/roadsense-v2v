/**
 * @file PackageManager.cpp
 * @brief Implementation of PackageManager for V2V message deduplication
 *
 * Ported from legacy RoadSense system with adaptations for:
 * - V2VMessage format (BSM-compatible)
 * - Logger class (replaces IConsoleMiddleware)
 * - Removed WiFi dependencies (MAC passed as parameter)
 */

#include "PackageManager.h"

// ============================================================================
// CONSTRUCTOR
// ============================================================================

PackageManager::PackageManager() {
    // No initialization needed - Logger is a singleton
}

// ============================================================================
// PACKAGE ADDITION
// ============================================================================

void PackageManager::addPackage(const uint8_t* macAddress, const V2VMessage& msg, uint8_t hopCount) {
    // CRITICAL: Use sourceMAC from message, NOT immediate sender's MAC
    // This ensures deduplication works across relay hops
    String sourceMacKey = MACHelper::macToString(msg.sourceMAC);
    String senderMacKey = MACHelper::macToString(macAddress);

    unsigned long currentTime = millis();

    // ========================================================================
    // MAC LIMIT ENFORCEMENT (Fix for DoS vulnerability)
    // ========================================================================
    // If this is a NEW MAC and we're at capacity, evict the oldest MAC (LRU)
    if (packages.find(sourceMacKey) == packages.end() &&  // New MAC (not currently tracked)
        packages.size() >= MAX_TRACKED_MACS) {             // At capacity

        // Find MAC with the oldest (earliest) receivedTime
        auto oldestMAC = packages.end();
        unsigned long oldestTime = ULONG_MAX;

        for (auto it = packages.begin(); it != packages.end(); ++it) {
            if (!it->second.empty()) {
                // Get timestamp of oldest package for this MAC (front = oldest due to FIFO)
                unsigned long macOldestTime = it->second.front().receivedTime;
                if (macOldestTime < oldestTime) {
                    oldestTime = macOldestTime;
                    oldestMAC = it;
                }
            }
        }

        // Evict the oldest MAC
        if (oldestMAC != packages.end()) {
            Logger::getInstance().warning(MODULE_NAME,
                "MAC limit reached (" + String(MAX_TRACKED_MACS) +
                "), evicting oldest: " + oldestMAC->first +
                " (age: " + String(currentTime - oldestTime) + "ms)");
            packages.erase(oldestMAC);
        }
    }

    // Create new package with metadata
    PackageData newPackage;
    newPackage.message = msg;
    newPackage.receivedTime = currentTime;
    newPackage.hopCount = hopCount;
    newPackage.processed = false;

    // Get or create vector for this source MAC (now safe - we enforced limit above)
    auto& packageVector = packages[sourceMacKey];

    // DEDUPLICATION: Check for duplicate based on timestamp
    // Same sourceMAC + same timestamp = duplicate message (relayed via different path)
    bool isDuplicate = false;
    for (const auto& pkg : packageVector) {
        if (pkg.message.timestamp == msg.timestamp) {
            isDuplicate = true;
            Logger::getInstance().debug(MODULE_NAME,
                "Duplicate message from " + sourceMacKey +
                " (timestamp: " + String(msg.timestamp) + ", via " + senderMacKey + ")");
            break;
        }
    }

    if (isDuplicate) {
        return;  // Reject duplicate
    }

    // Add package to storage
    packageVector.push_back(newPackage);

    // MEMORY MANAGEMENT: Limit packages per source
    // Remove oldest if exceeding limit (FIFO eviction)
    if (packageVector.size() > MAX_PACKAGES_PER_SOURCE) {
        packageVector.erase(packageVector.begin());
        Logger::getInstance().debug(MODULE_NAME,
            "Evicted oldest package from " + sourceMacKey + " (limit: " + String(MAX_PACKAGES_PER_SOURCE) + ")");
    }

    Logger::getInstance().debug(MODULE_NAME,
        "Added package from " + sourceMacKey +
        " (vehicle: " + String(msg.vehicleId) +
        ", hop: " + String(hopCount) +
        ", total: " + String(packageVector.size()) +
        ", tracked MACs: " + String(packages.size()) + ")");
}

void PackageManager::addLocalPackage(const V2VMessage& msg, const uint8_t* ownMac) {
    // Add locally-generated message with hop count = 0
    addPackage(ownMac, msg, 0);
}

// ============================================================================
// PACKAGE RETRIEVAL
// ============================================================================

PackageData* PackageManager::getLatestPackage(const uint8_t* macAddress) {
    String macKey = MACHelper::macToString(macAddress);

    auto it = packages.find(macKey);
    if (it == packages.end() || it->second.empty()) {
        return nullptr;
    }

    // Return pointer to last element (most recent)
    return &(it->second.back());
}

std::vector<PackageData> PackageManager::getAllPackagesForMAC(const uint8_t* macAddress) {
    String macKey = MACHelper::macToString(macAddress);

    auto it = packages.find(macKey);
    if (it == packages.end()) {
        return std::vector<PackageData>();  // Empty vector
    }

    return it->second;
}

std::vector<PackageData> PackageManager::getUnprocessedPackages(const uint8_t* macAddress) {
    String macKey = MACHelper::macToString(macAddress);
    std::vector<PackageData> unprocessed;

    auto it = packages.find(macKey);
    if (it == packages.end()) {
        return unprocessed;  // Empty
    }

    // Filter for unprocessed packages
    for (const auto& pkg : it->second) {
        if (!pkg.processed) {
            unprocessed.push_back(pkg);
        }
    }

    return unprocessed;
}

std::vector<std::pair<String, PackageData>> PackageManager::getAllUnprocessedPackages() {
    std::vector<std::pair<String, PackageData>> unprocessed;

    for (const auto& entry : packages) {
        const String& macKey = entry.first;
        const std::vector<PackageData>& pkgVector = entry.second;

        for (const auto& pkg : pkgVector) {
            if (!pkg.processed) {
                unprocessed.push_back({macKey, pkg});
            }
        }
    }

    return unprocessed;
}

std::map<String, std::vector<PackageData>> PackageManager::getMyDataPackages(const uint8_t* ownMac) {
    std::map<String, std::vector<PackageData>> myDataPackages;
    String localMacKey = MACHelper::macToString(ownMac);

    // Return packages where sourceMAC matches ownMac
    if (packages.find(localMacKey) != packages.end()) {
        myDataPackages[localMacKey] = packages[localMacKey];
    }

    return myDataPackages;
}

const std::map<String, std::vector<PackageData>>& PackageManager::getAllPackages() const {
    return packages;
}

// ============================================================================
// CLEANUP
// ============================================================================

void PackageManager::cleanupOldPackages(unsigned long maxAge) {
    unsigned long currentTime = millis();
    int totalRemoved = 0;

    // Iterate through all MAC addresses
    for (auto it = packages.begin(); it != packages.end(); ) {
        auto& packageVector = it->second;
        size_t initialSize = packageVector.size();

        // Remove packages older than maxAge using erase-remove idiom
        packageVector.erase(
            std::remove_if(
                packageVector.begin(),
                packageVector.end(),
                [currentTime, maxAge](const PackageData& pkg) {
                    return (currentTime - pkg.receivedTime) > maxAge;
                }
            ),
            packageVector.end()
        );

        size_t removed = initialSize - packageVector.size();
        totalRemoved += removed;

        if (removed > 0) {
            Logger::getInstance().debug(MODULE_NAME,
                "Cleaned " + String(removed) + " old packages from " + it->first);
        }

        // Remove MAC entry if no packages remain
        if (packageVector.empty()) {
            Logger::getInstance().debug(MODULE_NAME,
                "Removed empty MAC entry: " + it->first);
            it = packages.erase(it);
        } else {
            ++it;
        }
    }

    if (totalRemoved > 0) {
        Logger::getInstance().info(MODULE_NAME,
            "Cleanup complete: removed " + String(totalRemoved) + " packages (age > " + String(maxAge) + "ms)");
    }
}

int PackageManager::cleanupWithActiveList(const std::set<String>& activeMacs, unsigned long maxAge) {
    unsigned long currentTime = millis();
    int totalRemoved = 0;

    Logger::getInstance().debug(MODULE_NAME,
        "Cleanup with active list (" + String(activeMacs.size()) + " active peers)");

    for (auto it = packages.begin(); it != packages.end(); ) {
        const String& macKey = it->first;
        auto& packageVector = it->second;
        size_t initialSize = packageVector.size();

        bool isActive = (activeMacs.find(macKey) != activeMacs.end());

        if (!isActive) {
            // INACTIVE peer: Remove ALL packages immediately
            totalRemoved += packageVector.size();
            Logger::getInstance().debug(MODULE_NAME,
                "Removed " + String(packageVector.size()) + " packages from INACTIVE peer: " + macKey);
            it = packages.erase(it);
            continue;
        } else {
            // ACTIVE peer: Remove only OLD packages
            packageVector.erase(
                std::remove_if(
                    packageVector.begin(),
                    packageVector.end(),
                    [currentTime, maxAge](const PackageData& pkg) {
                        return (currentTime - pkg.receivedTime) > maxAge;
                    }
                ),
                packageVector.end()
            );

            size_t removed = initialSize - packageVector.size();
            totalRemoved += removed;

            if (removed > 0) {
                Logger::getInstance().debug(MODULE_NAME,
                    "Cleaned " + String(removed) + " old packages from ACTIVE peer: " + macKey);
            }

            // Remove if empty after cleanup
            if (packageVector.empty()) {
                Logger::getInstance().debug(MODULE_NAME,
                    "Removed empty MAC entry (active): " + macKey);
                it = packages.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (totalRemoved > 0) {
        Logger::getInstance().info(MODULE_NAME,
            "Active-list cleanup complete: removed " + String(totalRemoved) + " packages");
    }

    return totalRemoved;
}

// ============================================================================
// PROCESSING
// ============================================================================

void PackageManager::markProcessed(const uint8_t* macAddress) {
    String macKey = MACHelper::macToString(macAddress);

    auto it = packages.find(macKey);
    if (it == packages.end()) {
        return;
    }

    int marked = 0;
    for (auto& pkg : it->second) {
        if (!pkg.processed) {
            pkg.processed = true;
            marked++;
        }
    }

    if (marked > 0) {
        Logger::getInstance().debug(MODULE_NAME,
            "Marked " + String(marked) + " packages as processed for " + macKey);
    }
}

void PackageManager::resetProcessedStatus(unsigned long maxAge) {
    unsigned long currentTime = millis();
    int resetCount = 0;

    for (auto& entry : packages) {
        for (auto& pkg : entry.second) {
            if (pkg.processed && (currentTime - pkg.receivedTime) > maxAge) {
                pkg.processed = false;
                resetCount++;
            }
        }
    }

    if (resetCount > 0) {
        Logger::getInstance().debug(MODULE_NAME,
            "Reset processed status for " + String(resetCount) + " old packages");
    }
}

void PackageManager::setProcessPackageCallback(ProcessPackageCallback callback) {
    onProcessPackage = callback;
    Logger::getInstance().debug(MODULE_NAME, "Processing callback registered");
}

void PackageManager::processAllPendingPackages() {
    if (onProcessPackage == nullptr) {
        Logger::getInstance().debug(MODULE_NAME,
            "No callback registered - skipping processing");
        return;
    }

    int processed = 0;

    for (auto& entry : packages) {
        const String& macKey = entry.first;
        auto& packageVector = entry.second;

        for (auto& pkg : packageVector) {
            if (!pkg.processed) {
                // Invoke callback
                onProcessPackage(macKey, pkg);

                // Mark as processed
                pkg.processed = true;
                processed++;
            }
        }
    }

    if (processed > 0) {
        Logger::getInstance().info(MODULE_NAME,
            "Processed " + String(processed) + " pending packages");
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

void PackageManager::printPackageStats() {
    Logger::getInstance().info(MODULE_NAME, "=== Package Statistics ===");
    Logger::getInstance().info(MODULE_NAME,
        "Total MACs tracked: " + String(packages.size()) + " / " + String(MAX_TRACKED_MACS));

    int totalPackages = 0;
    int totalUnprocessed = 0;

    for (const auto& entry : packages) {
        const String& macKey = entry.first;
        const std::vector<PackageData>& pkgVector = entry.second;

        int unprocessed = getUnprocessedCount(pkgVector);
        totalPackages += pkgVector.size();
        totalUnprocessed += unprocessed;

        Logger::getInstance().info(MODULE_NAME,
            "  " + macKey + ": " + String(pkgVector.size()) +
            " packages (" + String(unprocessed) + " unprocessed)");
    }

    Logger::getInstance().info(MODULE_NAME,
        "Total packages: " + String(totalPackages) +
        " (" + String(totalUnprocessed) + " unprocessed)");

    // Memory estimate
    size_t memoryUsage = totalPackages * sizeof(PackageData);
    Logger::getInstance().info(MODULE_NAME,
        "Estimated memory: " + String(memoryUsage) + " bytes");

    Logger::getInstance().info(MODULE_NAME, "=========================");
}

uint32_t PackageManager::getPackageCount() const {
    uint32_t totalPackages = 0;

    for (const auto& entry : packages) {
        totalPackages += entry.second.size();
    }

    return totalPackages;
}

int PackageManager::getUnprocessedCount(const std::vector<PackageData>& pkgVector) {
    int count = 0;
    for (const auto& pkg : pkgVector) {
        if (!pkg.processed) {
            count++;
        }
    }
    return count;
}
