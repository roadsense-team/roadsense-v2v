/**
 * @file QMC5883LDriver.h
 * @brief Driver for QMC5883L 3-axis magnetometer
 */

#ifndef QMC5883L_DRIVER_H
#define QMC5883L_DRIVER_H

#include <Arduino.h>
#include <Wire.h>
#include "../../config.h"

class QMC5883LDriver {
public:
    struct MagData {
        float x;
        float y;
        float z;
        uint32_t timestamp;
        bool valid;

        MagData() : x(0.0f), y(0.0f), z(0.0f), timestamp(0), valid(false) {}
    };

    explicit QMC5883LDriver(uint8_t i2cAddr = QMC5883L_I2C_ADDR);

    /**
     * @brief Initialize sensor using an existing I2C bus.
     * @note Does NOT call Wire.begin().
     */
    bool begin(TwoWire& wire);

    /**
     * @brief Read raw signed axis counts from sensor registers.
     */
    bool readRaw(int16_t& x, int16_t& y, int16_t& z);

    /**
     * @brief Read scaled magnetic field values in microtesla.
     */
    bool read(float& x, float& y, float& z);

    /**
     * @brief Read scaled values into a struct.
     */
    bool read(MagData& data);

    bool isInitialized() const { return m_initialized; }

private:
    TwoWire* m_wire;
    uint8_t m_i2cAddr;
    bool m_initialized;

    bool probeDevice();
    bool writeRegister(uint8_t reg, uint8_t value);
    bool readRegister(uint8_t reg, uint8_t& value);
};

#endif // QMC5883L_DRIVER_H
