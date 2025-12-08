/**
 * @file test_main.cpp
 * @brief I2C Scanner + MPU WHO_AM_I Diagnostic
 *
 * Scans I2C bus for devices and reads WHO_AM_I register from MPU sensor
 */

#include <unity.h>
#include <Arduino.h>
#include <Wire.h>
#include "config.h"

void setUp(void) {}
void tearDown(void) {}

void test_i2c_scan() {
    Serial.println("\n=== I2C Bus Scanner ===");
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_HZ);

    Serial.printf("Scanning I2C bus (SDA=%d, SCL=%d)...\n", I2C_SDA_PIN, I2C_SCL_PIN);

    int devicesFound = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        byte error = Wire.endTransmission();

        if (error == 0) {
            Serial.printf("✅ Device found at 0x%02X\n", addr);
            devicesFound++;
        }
    }

    Serial.printf("\nTotal devices found: %d\n", devicesFound);
    TEST_ASSERT_GREATER_THAN(0, devicesFound);
}

void test_mpu_whoami() {
    Serial.println("\n=== MPU WHO_AM_I Check ===");

    // Try both possible I2C addresses
    uint8_t addresses[] = {0x68, 0x69};
    bool foundMPU = false;

    for (int i = 0; i < 2; i++) {
        uint8_t addr = addresses[i];

        Wire.beginTransmission(addr);
        byte error = Wire.endTransmission();

        if (error == 0) {
            Serial.printf("\n📍 MPU device responds at 0x%02X\n", addr);

            // Read WHO_AM_I register (0x75)
            Wire.beginTransmission(addr);
            Wire.write(0x75);  // WHO_AM_I register
            Wire.endTransmission(false);
            Wire.requestFrom(addr, (uint8_t)1);

            if (Wire.available()) {
                uint8_t whoami = Wire.read();
                Serial.printf("WHO_AM_I register: 0x%02X\n", whoami);

                // Known MPU variants:
                if (whoami == 0x70) {
                    Serial.println("✅ Identified as: MPU6500");
                    foundMPU = true;
                } else if (whoami == 0x71 || whoami == 0x73) {
                    Serial.println("✅ Identified as: MPU9250");
                    foundMPU = true;
                } else if (whoami == 0x68) {
                    Serial.println("✅ Identified as: MPU6050");
                    foundMPU = true;
                } else if (whoami == 0x98) {
                    Serial.println("✅ Identified as: MPU6000");
                    foundMPU = true;
                } else {
                    Serial.printf("⚠️  Unknown MPU variant (WHO_AM_I: 0x%02X)\n", whoami);
                    Serial.println("This might be:");
                    Serial.println("  - MPU6555 (0x7C)");
                    Serial.println("  - MPU9255 (0x73)");
                    Serial.println("  - Clone/fake chip");
                    foundMPU = true;  // Still found something
                }
            }
        }
    }

    if (!foundMPU) {
        TEST_FAIL_MESSAGE("No MPU device found at 0x68 or 0x69!");
    }

    TEST_ASSERT_TRUE(foundMPU);
}

void setup() {
    delay(2000);
    Serial.begin(115200);

    UNITY_BEGIN();
    RUN_TEST(test_i2c_scan);
    RUN_TEST(test_mpu_whoami);
    UNITY_END();
}

void loop() {}
