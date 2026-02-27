#ifndef MESH_RELAY_POLICY_H
#define MESH_RELAY_POLICY_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace MeshRelayPolicy {

bool shouldRelayMessage(uint8_t incomingHopCount,
                        bool sourceIsSelf,
                        bool inFrontCone,
                        uint8_t maxHopCount);

uint8_t computeRelayedHopCount(uint8_t incomingHopCount, uint8_t maxHopCount);

template <typename MessageT, typename AddPackageFn>
bool parseAndStoreReceivedMessage(const uint8_t* senderMac,
                                  const uint8_t* data,
                                  size_t len,
                                  AddPackageFn&& addPackage,
                                  MessageT* parsedMessage = nullptr) {
    static_assert(std::is_trivially_copyable<MessageT>::value,
                  "MessageT must be trivially copyable");

    if (senderMac == nullptr || data == nullptr || len != sizeof(MessageT)) {
        return false;
    }

    MessageT msg{};
    std::memcpy(&msg, data, sizeof(MessageT));
    addPackage(senderMac, msg, msg.hopCount);

    if (parsedMessage != nullptr) {
        *parsedMessage = msg;
    }

    return true;
}

} // namespace MeshRelayPolicy

#endif // MESH_RELAY_POLICY_H
