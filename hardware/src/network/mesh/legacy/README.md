# Legacy Firmware Reference Files

**DO NOT MODIFY OR COMPILE THESE FILES**

This directory contains **read-only reference copies** of the original RoadSense firmware's ESP-NOW mesh networking implementation.

## Purpose

These files serve as reference for porting to the new architecture:
- **Study** the proven ESP-NOW initialization sequence
- **Reference** deduplication and relay logic
- **Understand** mesh network design patterns
- **Preserve** the original implementation for comparison

## Files

| File | Purpose | Status |
|------|---------|--------|
| `MESH_Sender.h/cpp.txt` | ESP-NOW broadcast sender | ✅ Ported to `EspNowTransport` |
| `MESH_Receiver.h/cpp.txt` | ESP-NOW receiver + relay | ⏳ To be ported in Session 2-3 |
| `PackageManager.h/cpp.txt` | Message deduplication | ⏳ Port in Session 2 |
| `PeerManager.h/cpp.txt` | Active peer tracking | ⏳ Port in Session 3 |
| `CleanupManager.h/cpp.txt` | Garbage collection | ⏳ Port in Session 3 |
| `Data.h/cpp.txt` | Legacy message format | ✅ Replaced by `V2VMessage.h` |
| `MESH_config.h/cpp.txt` | Configuration singleton | 📖 Reference only |

## Why `.cpp.txt`?

The `.cpp` files are renamed to `.cpp.txt` to prevent PlatformIO from compiling them. They have dependencies on legacy interfaces that don't exist in the new architecture.

## ⚠️ Temporary Files - Cleanup Plan

**IMPORTANT:** These files are **TEMPORARY REFERENCE ONLY** and should be removed after Phase 2 is complete and validated.

**Cleanup Timeline:**
1. **During Phase 2 (Sessions 1-4):** Keep all files - actively referencing during porting
2. **After Phase 2 Session 4 complete:** Validate all ported components work with hardware
3. **After hardware validation passes:** Remove this entire `legacy/` directory
4. **Optional:** Archive to `docs/legacy/reference_code/` if team wants to preserve for future reference

**Validation Checklist Before Removal:**
- [ ] Phase 2 Session 4 complete (all sessions done)
- [ ] Hardware Test Session 1 passed (2-unit communication)
- [ ] Hardware Test Session 2 passed (3-unit relay)
- [ ] Deduplication working (PackageManager validated)
- [ ] Peer tracking working (PeerManager validated)
- [ ] Cleanup working (CleanupManager validated)
- [ ] Latency <50ms (performance validated)
- [ ] No regressions in any component

**Removal Command (after validation):**
```bash
cd ~/RoadSense2/roadsense-v2v/hardware/src/network/mesh
rm -rf legacy/
# Or archive: mv legacy/ ~/RoadSense2/docs/legacy/reference_code/
```

**Why remove them?**
- Reduces repo size
- Prevents confusion about which code is active
- Forces reliance on new, tested implementations
- Cleaner codebase for team members

## How to Use

1. **Read** the `.h` files to understand interfaces
2. **Study** the `.cpp.txt` files for implementation logic
3. **Adapt** the code to use new interfaces (`ITransport`, `V2VMessage`)
4. **Test** the new implementation with hardware

## Key Takeaways from Legacy Code

### ESP-NOW Initialization (MESH_Sender.cpp.txt:90-129)
```cpp
WiFi.mode(WIFI_STA);
WiFi.disconnect();  // ← CRITICAL: Prevents auto-connect interference
esp_wifi_set_channel(MESH_CHANNEL, WIFI_SECOND_CHAN_NONE);
esp_now_init();
esp_now_register_send_cb(callback);
esp_now_add_peer(broadcastPeerInfo);
```

**This sequence must not be changed!**

### Deduplication Logic (PackageManager.cpp.txt)
- Key: `sourceMAC + timestamp`
- Uses `std::map<String, Data>`
- 60-second timeout for cleanup
- Prevents message loops in mesh

### Relay Logic (MESH_Receiver.cpp.txt)
- Checks `hopCount` before relay
- Increments `hopCount` on forward
- Max hops: 3
- Prevents infinite loops via PackageManager

### Peer Tracking (PeerManager.cpp.txt)
- Tracks: MAC, lastSeen, hopCount
- 60-second inactivity timeout
- Updates on every received message

## Migration Status

**Phase 2 Session 1:** ✅ COMPLETE
- ported `MESH_Sender` → `EspNowTransport`
- ESP-NOW initialization working
- Basic send/receive functional

**Phase 2 Session 2:** ⏳ IN PROGRESS
- Port `PackageManager` for deduplication
- Adapt `Data` → `V2VMessage`

**Phase 2 Session 3:** ⏳ PENDING
- Port `PeerManager`
- Port `CleanupManager`

**Phase 2 Session 4:** ⏳ PENDING
- Create `MeshManager` orchestrator
- Full integration

## Original Source

**Repository:** `/home/amirkhalifa/Desktop/layer_1/`
**Copied:** November 13, 2025
**Version:** Legacy RoadSense (3-axis accelerometer version)

## Contact

**Questions about legacy code:** Check `docs/legacy/legacy_firmware_overview.md`
**Migration progress:** See `docs/HARDWARE_PROGRESS.md`

---

**REMEMBER:** These files are **REFERENCE ONLY** - do not modify or compile!
