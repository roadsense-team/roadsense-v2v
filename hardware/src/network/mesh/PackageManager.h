/**
 * @file PackageManager.h
 * @brief Package management for V2V mesh networking
 *
 * Manages incoming V2V messages with deduplication, storage, and cleanup.
 * Ported from legacy RoadSense system with adaptations for V2VMessage format.
 *
 * Key Features:
 * - Deduplication based on sourceMAC + timestamp
 * - Bounded memory (max 3 packages per source)
 * - Automatic cleanup of stale packages (60 seconds)
 * - Processing callbacks for application logic
 *
 * @note Uses V2VMessage (BSM-compatible) instead of legacy Data struct
 */

#ifndef PACKAGE_MANAGER_H
#define PACKAGE_MANAGER_H

#include <Arduino.h>
#include <vector>
#include <map>
#include <set>
#include <functional>

#include "../protocol/V2VMessage.h"
#include "../../utils/Logger.h"
#include "../../utils/MACHelper.h"
#include "../../config.h"

/**
 * @brief Structure to store package data with metadata
 *
 * Wraps a V2VMessage with reception metadata (time, hop count, processing status).
 * Used internally by PackageManager for tracking.
 */
struct PackageData {
    V2VMessage message;          ///< The actual V2V message content
    unsigned long receivedTime;  ///< millis() when package was received
    uint8_t hopCount;            ///< Hop count at time of receipt
    bool processed;              ///< Processing status flag
};

/**
 * @brief Manages V2V message deduplication and storage
 *
 * PackageManager tracks received V2V messages, eliminates duplicates
 * (based on sourceMAC + timestamp), and provides cleanup of stale data.
 *
 * Deduplication Strategy:
 * - Key: sourceMAC (from V2VMessage.sourceMAC)
 * - Duplicate check: timestamp comparison (millis() from original sender)
 * - Storage limit: MAX_PACKAGES_PER_SOURCE (3) per MAC address
 * - Cleanup: Remove packages older than PACKAGE_TIMEOUT_MS (60 seconds)
 *
 * Usage Example:
 * @code
 * PackageManager pm;
 *
 * // When receiving a message via ESP-NOW:
 * pm.addPackage(senderMAC, v2vMessage, hopCount);
 *
 * // Process all unprocessed messages:
 * pm.setProcessPackageCallback([](const String& mac, const PackageData& pkg) {
 *     Serial.printf("Processing message from %s\n", mac.c_str());
 * });
 * pm.processAllPendingPackages();
 *
 * // Periodic cleanup (call in loop):
 * pm.cleanupOldPackages();
 * @endcode
 */
class PackageManager {
public:
    /**
     * @brief Callback function type for processing packages
     *
     * Called for each unprocessed package when processAllPendingPackages() is invoked.
     *
     * @param macAddress Source MAC address (as string)
     * @param package Package data with V2VMessage and metadata
     */
    using ProcessPackageCallback = std::function<void(const String&, const PackageData&)>;

    /**
     * @brief Construct a new PackageManager
     *
     * Initializes empty package storage. No external dependencies required.
     */
    PackageManager();

    /**
     * @brief Add a received package to storage
     *
     * Performs deduplication check based on sourceMAC + timestamp.
     * If duplicate detected, message is rejected. Otherwise, stored
     * with metadata and limited to MAX_PACKAGES_PER_SOURCE per MAC.
     *
     * @param macAddress Immediate sender's MAC address (6 bytes)
     * @param msg V2VMessage to store
     * @param hopCount Current hop count (from network layer)
     *
     * @note Uses msg.sourceMAC as the deduplication key, NOT macAddress
     * @note Duplicate = same sourceMAC + same timestamp
     */
    void addPackage(const uint8_t* macAddress, const V2VMessage& msg, uint8_t hopCount);

    /**
     * @brief Add a locally-generated package
     *
     * Used when this vehicle sends a message that should also be
     * stored locally (e.g., for logging, debugging, or HIL bridge).
     *
     * @param msg V2VMessage to store
     * @param ownMac This vehicle's MAC address (6 bytes)
     *
     * @note Sets hopCount = 0 (local origin)
     */
    void addLocalPackage(const V2VMessage& msg, const uint8_t* ownMac);

    /**
     * @brief Get the latest package from a specific source
     *
     * Returns the most recently received package from the given MAC address.
     *
     * @param macAddress Source MAC address (6 bytes)
     * @return PackageData* Pointer to latest package, or nullptr if none found
     *
     * @warning Returned pointer is valid until next cleanup or package addition
     */
    PackageData* getLatestPackage(const uint8_t* macAddress);

    /**
     * @brief Get all packages for a specific source MAC
     *
     * Returns all stored packages from the given MAC address.
     *
     * @param macAddress Source MAC address (6 bytes)
     * @return std::vector<PackageData> Vector of packages (may be empty)
     */
    std::vector<PackageData> getAllPackagesForMAC(const uint8_t* macAddress);

    /**
     * @brief Get unprocessed packages from a specific source
     *
     * Returns packages where processed == false.
     *
     * @param macAddress Source MAC address (6 bytes)
     * @return std::vector<PackageData> Vector of unprocessed packages
     */
    std::vector<PackageData> getUnprocessedPackages(const uint8_t* macAddress);

    /**
     * @brief Get all unprocessed packages across all sources
     *
     * Returns all packages where processed == false, from all MAC addresses.
     *
     * @return std::vector<std::pair<String, PackageData>> Vector of (MAC, PackageData) pairs
     */
    std::vector<std::pair<String, PackageData>> getAllUnprocessedPackages();

    /**
     * @brief Get packages generated by this vehicle
     *
     * Returns all packages where sourceMAC matches ownMac.
     *
     * @param ownMac This vehicle's MAC address (6 bytes)
     * @return std::map<String, std::vector<PackageData>> Map of MAC → packages
     */
    std::map<String, std::vector<PackageData>> getMyDataPackages(const uint8_t* ownMac);

    /**
     * @brief Get all packages (all sources)
     *
     * Direct access to internal storage. Use with caution.
     *
     * @return const std::map<String, std::vector<PackageData>>& Reference to package map
     */
    const std::map<String, std::vector<PackageData>>& getAllPackages() const;

    /**
     * @brief Clean up old packages based on age
     *
     * Removes packages older than maxAge milliseconds.
     * Also removes empty MAC entries from storage.
     *
     * @param maxAge Maximum age in milliseconds (default: PACKAGE_TIMEOUT_MS = 60 seconds)
     *
     * @note Call this periodically in main loop (e.g., every 1 second)
     */
    void cleanupOldPackages(unsigned long maxAge = PACKAGE_TIMEOUT_MS);

    /**
     * @brief Clean up with active peer list integration
     *
     * More aggressive cleanup:
     * - For INACTIVE peers: delete ALL packages immediately
     * - For ACTIVE peers: delete only OLD packages (>maxAge)
     *
     * @param activeMacs Set of active MAC addresses (from PeerManager)
     * @param maxAge Maximum age for active peers (default: PACKAGE_TIMEOUT_MS)
     * @return int Number of packages removed
     *
     * @note Requires integration with PeerManager
     */
    int cleanupWithActiveList(const std::set<String>& activeMacs,
                              unsigned long maxAge = PACKAGE_TIMEOUT_MS);

    /**
     * @brief Mark all packages from a MAC as processed
     *
     * Sets processed = true for all packages from the given source.
     *
     * @param macAddress Source MAC address (6 bytes)
     */
    void markProcessed(const uint8_t* macAddress);

    /**
     * @brief Reset processed status for old packages
     *
     * Sets processed = false for packages older than maxAge.
     * Useful for reprocessing stale data.
     *
     * @param maxAge Age threshold in milliseconds
     */
    void resetProcessedStatus(unsigned long maxAge);

    /**
     * @brief Set callback for processing packages
     *
     * Registers a function to be called for each unprocessed package
     * when processAllPendingPackages() is invoked.
     *
     * @param callback Function to call (signature: void(const String&, const PackageData&))
     */
    void setProcessPackageCallback(ProcessPackageCallback callback);

    /**
     * @brief Process all pending (unprocessed) packages
     *
     * Iterates through all packages where processed == false and
     * invokes the registered callback for each one.
     *
     * @note Does nothing if no callback is registered
     * @note Automatically marks packages as processed after callback
     */
    void processAllPendingPackages();

    /**
     * @brief Print package statistics to serial
     *
     * Displays:
     * - Total MACs tracked
     * - Packages per MAC
     * - Unprocessed count
     * - Memory usage estimate
     *
     * @note Useful for debugging
     */
    void printPackageStats();

    /**
     * @brief Get total package count across all sources
     *
     * Counts all stored packages from all MAC addresses.
     *
     * @return uint32_t Total number of packages stored
     *
     * @note Useful for testing and monitoring memory usage
     */
    uint32_t getPackageCount() const;

private:
    const String MODULE_NAME = "PackageManager";  ///< Logger module identifier

    /// Callback for processing packages
    ProcessPackageCallback onProcessPackage = nullptr;

    /// Main storage: map<sourceMAC_string, vector<PackageData>>
    /// Key = sourceMAC (from V2VMessage.sourceMAC), NOT immediate sender
    std::map<String, std::vector<PackageData>> packages;

    /**
     * @brief Count unprocessed packages in a vector
     *
     * Helper function for statistics.
     *
     * @param pkgVector Vector of packages
     * @return int Count of packages where processed == false
     */
    int getUnprocessedCount(const std::vector<PackageData>& pkgVector);
};

#endif // PACKAGE_MANAGER_H
