#include "MeshRelayPolicy.h"

namespace MeshRelayPolicy {

bool shouldRelayMessage(uint8_t incomingHopCount,
                        bool sourceIsSelf,
                        bool inFrontCone,
                        uint8_t maxHopCount) {
    if (sourceIsSelf) {
        return false;
    }
    if (!inFrontCone) {
        return false;
    }
    return incomingHopCount < maxHopCount;
}

uint8_t computeRelayedHopCount(uint8_t incomingHopCount, uint8_t maxHopCount) {
    if (incomingHopCount >= maxHopCount) {
        return maxHopCount;
    }
    return incomingHopCount + 1;
}

} // namespace MeshRelayPolicy
