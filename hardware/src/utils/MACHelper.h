#ifndef MAC_HELPER_H
#define MAC_HELPER_H

#include <Arduino.h>
#include "../config.h"
/**
     MACHelper
     Utility class for MAC address operations
    
     This class provides static helper methods for working with
     MAC addresses across the application.
 */
class MACHelper {
public:
    /**
      Convert a MAC address byte array to a formatted string
     
      mac Pointer to 6-byte MAC address array
      String MAC address in format "xx:xx:xx:xx:xx:xx"
     */
    static String macToString(const uint8_t* mac) {
        char macStr[MAC_STRING_LENGTH];
        snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        return String(macStr);
    }
    
    /**
     Convert a MAC address string to a byte array
     
      macStr String MAC address in format "xx:xx:xx:xx:xx:xx"
      mac Output array to store the 6-byte MAC address
      bool True if conversion successful, false otherwise
     */
    static bool stringToMAC(const String& macStr, uint8_t* mac) {
        if (macStr.length() < 17) { // xx:xx:xx:xx:xx:xx is 17 chars
            return false;
        }
        
        return sscanf(macStr.c_str(), "%2hhx:%2hhx:%2hhx:%2hhx:%2hhx:%2hhx", 
                     &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]) == 6;
    }
    
    /**
       Compare two MAC addresses for equality
     
       mac1 First MAC address
       mac2 Second MAC address
       bool True if MAC addresses are identical
     */
    static bool compareMACAddresses(const uint8_t* mac1, const uint8_t* mac2) {
        for (int i = 0; i < MAC_ADDRESS_LENGTH; i++) {
            if (mac1[i] != mac2[i]) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Check if a MAC address is a broadcast address
     
       mac MAC address to check
       bool True if address is broadcast (FF:FF:FF:FF:FF:FF)
     */
    static bool isBroadcastAddress(const uint8_t* mac) {
        for (int i = 0; i < MAC_ADDRESS_LENGTH; i++) {
            if (mac[i] != 0xFF) {
                return false;
            }
        }
        return true;
    }
    
    /**
     Copy a MAC address from one array to another
     
     dest Destination array (must be at least 6 bytes)
     src Source MAC address array
     */
    static void copyMAC(uint8_t* dest, const uint8_t* src) {
        memcpy(dest, src, MAC_ADDRESS_LENGTH);
    }
};

#endif // MAC_HELPER_H