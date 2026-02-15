/**
 * @file test_main.cpp
 * @brief QMC5883L Magnetometer Hardware Test
 *
 * Tests:
 * 1. I2C communication at 0x0D
 * 2. Chip ID verification
 * 3. Sensor initialization
 * 4. Read magnetometer data (X, Y, Z)
 *
 * Hardware connections:
 * - SDA: GPIO 21 (shared with MPU6500)
 * - SCL: GPIO 22 (shared with MPU6500)
 * - VCC: 3.3V
 * - GND: GND
 */

#include <unity.h>
#include <Arduino.h>
#include <Wire.h>
#include "config.h"

// QMC5883L I2C Address and Registers
#define QMC5883L_ADDR       0x0D
#define QMC5883L_REG_XOUT_L 0x00
#define QMC5883L_REG_XOUT_H 0x01
#define QMC5883L_REG_YOUT_L 0x02
#define QMC5883L_REG_YOUT_H 0x03
#define QMC5883L_REG_ZOUT_L 0x04
#define QMC5883L_REG_ZOUT_H 0x05
#define QMC5883L_REG_STATUS 0x06
#define QMC5883L_REG_TOUT_L 0x07
#define QMC5883L_REG_TOUT_H 0x08
#define QMC5883L_REG_CONTROL1 0x09
#define QMC5883L_REG_CONTROL2 0x0A
#define QMC5883L_REG_SET_RESET 0x0B
#define QMC5883L_REG_CHIP_ID  0x0D

// Control register values
#define QMC5883L_MODE_CONTINUOUS 0x01
#define QMC5883L_ODR_200HZ       0x0C  // Output Data Rate: 200Hz
#define QMC5883L_RNG_8G          0x10  // Range: ±8 Gauss
#define QMC5883L_OSR_512         0x00  // Over Sample Ratio: 512

void setUp(void) {}
void tearDown(void) {}

/**
 * @brief Test I2C communication with QMC5883L
 */
void test_qmc5883l_i2c_communication() {
    Serial.println("\n=== QMC5883L I2C Communication Test ===");

    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_HZ);

    Serial.printf("Attempting to communicate with QMC5883L at 0x%02X...\n", QMC5883L_ADDR);

    Wire.beginTransmission(QMC5883L_ADDR);
    byte error = Wire.endTransmission();

    if (error == 0) {
        Serial.println("✅ QMC5883L responded on I2C bus!");
    } else {
        Serial.printf("❌ No response from QMC5883L (error code: %d)\n", error);
        Serial.println("Check connections:");
        Serial.println("  - SDA (GPIO 21) connected?");
        Serial.println("  - SCL (GPIO 22) connected?");
        Serial.println("  - VCC (3.3V) connected?");
        Serial.println("  - GND connected?");
        TEST_FAIL_MESSAGE("QMC5883L not found on I2C bus!");
    }

    TEST_ASSERT_EQUAL(0, error);
}

/**
 * @brief Test chip ID register (should return 0xFF)
 */
void test_qmc5883l_chip_id() {
    Serial.println("\n=== QMC5883L Chip ID Test ===");

    // Read Chip ID register (0x0D)
    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(QMC5883L_REG_CHIP_ID);
    Wire.endTransmission(false);

    Wire.requestFrom(static_cast<int>(QMC5883L_ADDR), 1);

    if (Wire.available()) {
        uint8_t chipId = Wire.read();
        Serial.printf("Chip ID register (0x0D): 0x%02X\n", chipId);

        if (chipId == 0xFF) {
            Serial.println("✅ Chip ID matches QMC5883L (0xFF)");
        } else {
            Serial.printf("⚠️  Unexpected Chip ID: 0x%02X (expected 0xFF)\n", chipId);
            Serial.println("This might be:");
            Serial.println("  - Different QMC5883L variant");
            Serial.println("  - Communication issue");
            Serial.println("  - Wrong sensor model");
        }

        TEST_ASSERT_EQUAL_HEX8(0xFF, chipId);
    } else {
        TEST_FAIL_MESSAGE("No data received from Chip ID register!");
    }
}

/**
 * @brief Initialize QMC5883L sensor
 */
void test_qmc5883l_initialization() {
    Serial.println("\n=== QMC5883L Initialization Test ===");

    // Step 1: Reset sensor
    Serial.println("Resetting sensor...");
    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(QMC5883L_REG_SET_RESET);
    Wire.write(0x01);  // Set/Reset Period
    byte error = Wire.endTransmission();
    TEST_ASSERT_EQUAL_MESSAGE(0, error, "Failed to write Set/Reset register");
    delay(10);

    // Step 2: Configure Control Register 1
    // Mode: Continuous | ODR: 200Hz | Range: 8G | OSR: 512
    uint8_t control1 = QMC5883L_MODE_CONTINUOUS | QMC5883L_ODR_200HZ |
                       QMC5883L_RNG_8G | QMC5883L_OSR_512;

    Serial.printf("Configuring Control Register 1: 0x%02X\n", control1);
    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(QMC5883L_REG_CONTROL1);
    Wire.write(control1);
    error = Wire.endTransmission();
    TEST_ASSERT_EQUAL_MESSAGE(0, error, "Failed to configure sensor");

    delay(100);  // Wait for sensor to stabilize

    Serial.println("✅ QMC5883L initialized successfully!");
}

/**
 * @brief Read magnetometer data (X, Y, Z)
 */
void test_qmc5883l_read_magnetometer() {
    Serial.println("\n=== QMC5883L Magnetometer Reading Test ===");

    // Read 6 bytes starting from XOUT_L
    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(QMC5883L_REG_XOUT_L);
    Wire.endTransmission(false);

    Wire.requestFrom(static_cast<int>(QMC5883L_ADDR), 6);

    if (Wire.available() >= 6) {
        // Read LSB then MSB explicitly to avoid undefined evaluation order
        uint8_t xl = Wire.read(); uint8_t xh = Wire.read();
        uint8_t yl = Wire.read(); uint8_t yh = Wire.read();
        uint8_t zl = Wire.read(); uint8_t zh = Wire.read();
        int16_t x = static_cast<int16_t>(xl | (xh << 8));
        int16_t y = static_cast<int16_t>(yl | (yh << 8));
        int16_t z = static_cast<int16_t>(zl | (zh << 8));

        Serial.println("Raw magnetometer data (16-bit signed):");
        Serial.printf("  X: %6d\n", x);
        Serial.printf("  Y: %6d\n", y);
        Serial.printf("  Z: %6d\n", z);

        // Verify data is not all zeros (sensor is working)
        bool isNonZero = (x != 0) || (y != 0) || (z != 0);
        TEST_ASSERT_TRUE_MESSAGE(isNonZero, "Magnetometer data is all zeros!");

        // Calculate magnetic field magnitude
        float magnitude = sqrt(x*x + y*y + z*z);
        Serial.printf("\nMagnetic field magnitude: %.1f (raw units)\n", magnitude);
        Serial.println("✅ Magnetometer data looks valid!");

        // Sanity check: magnitude should be > 0
        TEST_ASSERT_GREATER_THAN(0, magnitude);

    } else {
        TEST_FAIL_MESSAGE("Failed to read magnetometer data!");
    }
}

/**
 * @brief Read status register
 */
void test_qmc5883l_status_register() {
    Serial.println("\n=== QMC5883L Status Register Test ===");

    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(QMC5883L_REG_STATUS);
    Wire.endTransmission(false);

    Wire.requestFrom(static_cast<int>(QMC5883L_ADDR), 1);

    if (Wire.available()) {
        uint8_t status = Wire.read();
        Serial.printf("Status register: 0x%02X (binary: %08b)\n", status, status);
        Serial.printf("  - DRDY (bit 0): %s\n", (status & 0x01) ? "Data Ready" : "Not Ready");
        Serial.printf("  - OVL (bit 1):  %s\n", (status & 0x02) ? "Overflow" : "Normal");
        Serial.printf("  - DOR (bit 2):  %s\n", (status & 0x04) ? "Data Skipped" : "Normal");

        // Status register should be readable
        TEST_ASSERT_TRUE_MESSAGE(true, "Status register read successfully");
    } else {
        TEST_FAIL_MESSAGE("Failed to read status register!");
    }
}

void setup() {
    delay(2000);  // Wait for board stability
    Serial.begin(115200);

    Serial.println("\n");
    Serial.println("╔════════════════════════════════════════╗");
    Serial.println("║   QMC5883L Magnetometer Test Suite    ║");
    Serial.println("╚════════════════════════════════════════╝");
    Serial.println();

    UNITY_BEGIN();

    RUN_TEST(test_qmc5883l_i2c_communication);
    RUN_TEST(test_qmc5883l_chip_id);
    RUN_TEST(test_qmc5883l_initialization);
    RUN_TEST(test_qmc5883l_read_magnetometer);
    RUN_TEST(test_qmc5883l_status_register);

    UNITY_END();
}

void loop() {
    // Tests run once in setup()
}
