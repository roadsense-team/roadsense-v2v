#include "ConeFilter.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
}

float ConeFilter::normalizeAngleDeg(float angleDeg) {
    float normalized = std::fmod(angleDeg + 180.0f, 360.0f);
    if (normalized < 0.0f) {
        normalized += 360.0f;
    }
    return normalized - 180.0f;
}

bool ConeFilter::isInCone(float egoHeadingDeg,
                          double egoLat,
                          double egoLon,
                          double peerLat,
                          double peerLon,
                          float halfAngleDeg) {
    const double dLat = peerLat - egoLat;
    const double dLon = peerLon - egoLon;

    if (std::fabs(dLat) < 1e-9 && std::fabs(dLon) < 1e-9) {
        return true;
    }

    // Project lat/lon delta into local EN plane.
    const double avgLatRad = (egoLat + peerLat) * 0.5 * (kPi / 180.0);
    const double east = dLon * std::cos(avgLatRad);
    const double north = dLat;

    // Bearing in degrees with 0 = North, clockwise positive.
    float bearingDeg = static_cast<float>(std::atan2(east, north) * (180.0 / kPi));
    if (bearingDeg < 0.0f) {
        bearingDeg += 360.0f;
    }

    const float deltaDeg = normalizeAngleDeg(bearingDeg - egoHeadingDeg);
    return std::fabs(deltaDeg) <= halfAngleDeg;
}
