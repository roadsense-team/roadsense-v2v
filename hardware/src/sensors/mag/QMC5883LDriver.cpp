/**
 * @file QMC5883LDriver.cpp
 * @brief Driver for QMC5883L 3-axis magnetometer
 */

#include "QMC5883LDriver.h"

namespace {
constexpr uint8_t QMC5883L_REG_XOUT_L = 0x00;
constexpr uint8_t QMC5883L_REG_CONTROL1 = 0x09;
constexpr uint8_t QMC5883L_REG_CONTROL2 = 0x0A;
constexpr uint8_t QMC5883L_REG_SET_RESET = 0x0B;
constexpr uint8_t QMC5883L_REG_CHIP_ID = 0x0D;

constexpr uint8_t QMC5883L_MODE_CONTINUOUS = 0x01;
constexpr uint8_t QMC5883L_ODR_200HZ = 0x0C;
constexpr uint8_t QMC5883L_RNG_8G = 0x10;
constexpr uint8_t QMC5883L_OSR_512 = 0x00;
constexpr uint8_t QMC5883L_CONTROL1_CONFIG =
    QMC5883L_MODE_CONTINUOUS | QMC5883L_ODR_200HZ | QMC5883L_RNG_8G | QMC5883L_OSR_512;

// ±8 gauss mode: 3000 LSB/gauss, 1 gauss = 100 microtesla.
constexpr float QMC5883L_UT_PER_LSB = 100.0f / 3000.0f;
} // namespace

QMC5883LDriver::QMC5883LDriver(uint8_t i2cAddr)
    : m_wire(nullptr)
    , m_i2cAddr(i2cAddr)
    , m_initialized(false) {}

bool QMC5883LDriver::begin(TwoWire& wire) {
    m_wire = &wire;
    m_initialized = false;

    if (!probeDevice()) {
        return false;
    }

    uint8_t chipId = 0;
    if (!readRegister(QMC5883L_REG_CHIP_ID, chipId) || chipId != 0xFF) {
        return false;
    }

    if (!writeRegister(QMC5883L_REG_SET_RESET, 0x01)) {
        return false;
    }

    if (!writeRegister(QMC5883L_REG_CONTROL1, QMC5883L_CONTROL1_CONFIG)) {
        return false;
    }

    if (!writeRegister(QMC5883L_REG_CONTROL2, 0x00)) {
        return false;
    }

    uint8_t configReadback = 0;
    if (!readRegister(QMC5883L_REG_CONTROL1, configReadback)) {
        return false;
    }

    // Bits [4:0] encode mode/ODR/range/OSR in this configuration.
    if ((configReadback & 0x1F) != (QMC5883L_CONTROL1_CONFIG & 0x1F)) {
        return false;
    }

    m_initialized = true;
    return true;
}

bool QMC5883LDriver::readRaw(int16_t& x, int16_t& y, int16_t& z) {
    x = 0;
    y = 0;
    z = 0;

    if (!m_initialized || m_wire == nullptr) {
        return false;
    }

    m_wire->beginTransmission(m_i2cAddr);
    m_wire->write(QMC5883L_REG_XOUT_L);
    if (m_wire->endTransmission(false) != 0) {
        return false;
    }

    const uint8_t expectedBytes = 6;
    const uint8_t received = m_wire->requestFrom(static_cast<int>(m_i2cAddr), static_cast<int>(expectedBytes));
    if (received != expectedBytes || m_wire->available() < expectedBytes) {
        return false;
    }

    // Read LSB then MSB explicitly to avoid undefined evaluation order
    uint8_t xl = m_wire->read(); uint8_t xh = m_wire->read();
    uint8_t yl = m_wire->read(); uint8_t yh = m_wire->read();
    uint8_t zl = m_wire->read(); uint8_t zh = m_wire->read();
    x = static_cast<int16_t>(xl | (xh << 8));
    y = static_cast<int16_t>(yl | (yh << 8));
    z = static_cast<int16_t>(zl | (zh << 8));
    return true;
}

bool QMC5883LDriver::read(float& x, float& y, float& z) {
    int16_t rawX = 0;
    int16_t rawY = 0;
    int16_t rawZ = 0;

    if (!readRaw(rawX, rawY, rawZ)) {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
        return false;
    }

    x = static_cast<float>(rawX) * QMC5883L_UT_PER_LSB;
    y = static_cast<float>(rawY) * QMC5883L_UT_PER_LSB;
    z = static_cast<float>(rawZ) * QMC5883L_UT_PER_LSB;
    return true;
}

bool QMC5883LDriver::read(MagData& data) {
    data.valid = read(data.x, data.y, data.z);
    data.timestamp = millis();
    return data.valid;
}

bool QMC5883LDriver::probeDevice() {
    if (m_wire == nullptr) {
        return false;
    }

    m_wire->beginTransmission(m_i2cAddr);
    return (m_wire->endTransmission() == 0);
}

bool QMC5883LDriver::writeRegister(uint8_t reg, uint8_t value) {
    if (m_wire == nullptr) {
        return false;
    }

    m_wire->beginTransmission(m_i2cAddr);
    m_wire->write(reg);
    m_wire->write(value);
    return (m_wire->endTransmission() == 0);
}

bool QMC5883LDriver::readRegister(uint8_t reg, uint8_t& value) {
    value = 0;

    if (m_wire == nullptr) {
        return false;
    }

    m_wire->beginTransmission(m_i2cAddr);
    m_wire->write(reg);
    if (m_wire->endTransmission(false) != 0) {
        return false;
    }

    if (m_wire->requestFrom(static_cast<int>(m_i2cAddr), 1) != 1 || m_wire->available() < 1) {
        return false;
    }

    value = m_wire->read();
    return true;
}
