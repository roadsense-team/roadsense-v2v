#ifndef CONE_FILTER_H
#define CONE_FILTER_H

#include <cmath>

class ConeFilter {
public:
    // Heading convention matches SUMO/GPS: 0 deg = North, clockwise positive.
    static bool isInCone(float egoHeadingDeg,
                         double egoLat,
                         double egoLon,
                         double peerLat,
                         double peerLon,
                         float halfAngleDeg = 45.0f);

private:
    static float normalizeAngleDeg(float angleDeg);
};

#endif // CONE_FILTER_H
