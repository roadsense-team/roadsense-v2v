#include <unity.h>
#include <Arduino.h>
#include <cmath>
#include <Wire.h>
#include "config.h"
#include "sensors/mag/QMC5883LDriver.h"

void setUp(void) {}
void tearDown(void) {}

void test_read_before_begin_fails() {
    QMC5883LDriver driver;
    float x = 1.0f, y = 1.0f, z = 1.0f;
    int16_t rawX = 1, rawY = 1, rawZ = 1;

    TEST_ASSERT_FALSE(driver.read(x, y, z));
    TEST_ASSERT_FALSE(driver.readRaw(rawX, rawY, rawZ));
}

void test_init_failure_with_wrong_address() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_HZ);

    QMC5883LDriver wrongAddressDriver(0x0E);
    TEST_ASSERT_FALSE(wrongAddressDriver.begin(Wire));
}

void test_read_after_failed_init_returns_false() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_HZ);

    QMC5883LDriver wrongAddressDriver(0x0E);
    TEST_ASSERT_FALSE(wrongAddressDriver.begin(Wire));

    float x = 0.0f, y = 0.0f, z = 0.0f;
    TEST_ASSERT_FALSE(wrongAddressDriver.read(x, y, z));
}

void test_init_success_and_read_valid_data() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_HZ);

    QMC5883LDriver driver;
    if (!driver.begin(Wire)) {
        TEST_IGNORE_MESSAGE("QMC5883L not connected; skipping success/read test");
    }

    float x = 0.0f, y = 0.0f, z = 0.0f;
    TEST_ASSERT_TRUE(driver.read(x, y, z));

    // Expect at least one non-zero axis when sensor is active.
    TEST_ASSERT_TRUE((fabsf(x) > 0.001f) || (fabsf(y) > 0.001f) || (fabsf(z) > 0.001f));
}

void setup() {
    delay(2000);
    Serial.begin(115200);

    UNITY_BEGIN();
    RUN_TEST(test_read_before_begin_fails);
    RUN_TEST(test_init_failure_with_wrong_address);
    RUN_TEST(test_read_after_failed_init_returns_false);
    RUN_TEST(test_init_success_and_read_valid_data);
    UNITY_END();
}

void loop() {}
