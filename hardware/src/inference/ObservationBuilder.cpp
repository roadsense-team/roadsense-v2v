/**
 * ObservationBuilder.cpp
 *
 * See ObservationBuilder.h for interface documentation.
 */

#include "ObservationBuilder.h"
#include "../utils/MACHelper.h"
#include <Arduino.h>  // millis()
#include <cmath>
// Degrees to radians
static constexpr float kDegToRad = 3.14159265358979f / 180.0f;

// Metres per degree of latitude (approx, Earth radius 6371 km)
static constexpr float kMetresPerDegLat = 111320.0f;

// ============================================================================
// Public
// ============================================================================

void ObservationBuilder::reset() {
    memset(ego_out_,       0, sizeof(ego_out_));
    memset(peers_out_,     0, sizeof(peers_out_));
    memset(peer_mask_out_, 0, sizeof(peer_mask_out_));
    memset(ego_history_,   0, sizeof(ego_history_));
    history_write_ = 0;
    history_full_  = false;
    braking_decay_ = 0.0f;
    last_heading_  = 0.0f;
}

void ObservationBuilder::update(const V2VMessage& ownMsg,
                                const PackageManager& pm,
                                const uint8_t ownMAC[6]) {
    // ------------------------------------------------------------------ //
    // 1. Grab ego dynamics from own V2VMessage                            //
    // ------------------------------------------------------------------ //
    const float ego_speed    = ownMsg.dynamics.speed;
    const float ego_longAccel = ownMsg.dynamics.longAccel;
    const float ego_lat      = static_cast<float>(ownMsg.position.lat);
    const float ego_lon      = static_cast<float>(ownMsg.position.lon);

    // Use cached heading when speed is too low for GPS heading to be valid
    if (ego_speed >= 1.4f && std::isfinite(ownMsg.dynamics.heading)) {
        last_heading_ = ownMsg.dynamics.heading;
    }
    const float ego_heading = last_heading_;

    // ------------------------------------------------------------------ //
    // 2. Collect cone-filtered peers from PackageManager                  //
    // ------------------------------------------------------------------ //
    memset(peers_out_,     0, sizeof(peers_out_));
    memset(peer_mask_out_, 0, sizeof(peer_mask_out_));

    int valid_count       = 0;
    float min_peer_accel  = 0.0f;
    float max_closing_spd = 0.0f;
    bool any_braking_peer = false;

    const auto& all_pkgs = pm.getAllPackages();

    for (const auto& entry : all_pkgs) {
        if (valid_count >= kMaxPeers) break;

        // Find the most-recent package for this MAC
        const std::vector<PackageData>& vec = entry.second;
        if (vec.empty()) continue;

        const PackageData* latest = nullptr;
        for (const PackageData& pkg : vec) {
            if (latest == nullptr || pkg.receivedTime > latest->receivedTime) {
                latest = &pkg;
            }
        }
        if (!latest) continue;

        const V2VMessage& peer = latest->message;

        // Skip own messages
        if (MACHelper::compareMACAddresses(peer.sourceMAC, ownMAC)) continue;

        // Skip stale peers (>500 ms since we received their last message)
        const uint32_t age_ms = static_cast<uint32_t>(millis() - latest->receivedTime);
        if (age_ms > static_cast<uint32_t>(kStalenessThreshMs)) continue;

        const float peer_lat = static_cast<float>(peer.position.lat);
        const float peer_lon = static_cast<float>(peer.position.lon);

        // Skip peers with invalid position
        if (!std::isfinite(peer_lat) || !std::isfinite(peer_lon)) continue;
        if (peer_lat == 0.0f && peer_lon == 0.0f) continue;

        // Front-cone filter (90° cone, 45° half-angle)
        if (!isInCone(ego_heading, ego_lat, ego_lon, peer_lat, peer_lon)) continue;

        // Compute relative position in ego frame
        float rel_x = 0.0f, rel_y = 0.0f;
        toEgoFrame(ego_heading, ego_lat, ego_lon, peer_lat, peer_lon, rel_x, rel_y);

        const float peer_speed  = peer.dynamics.speed;
        const float peer_accel  = peer.dynamics.longAccel;
        const float peer_hdg    = peer.dynamics.heading;

        const float rel_speed   = peer_speed - ego_speed;
        float rel_heading       = peer_hdg - ego_heading;
        // Normalise to [-180, 180]
        rel_heading = std::fmod(rel_heading + 180.0f, 360.0f) - 180.0f;

        // Fill peer feature vector
        peers_out_[valid_count][0] = rel_x          / kMaxDistance;
        peers_out_[valid_count][1] = rel_y          / kMaxDistance;
        peers_out_[valid_count][2] = rel_speed       / kMaxSpeed;
        peers_out_[valid_count][3] = rel_heading     / 180.0f;
        peers_out_[valid_count][4] = peer_accel      / kMaxAccel;
        peers_out_[valid_count][5] = static_cast<float>(age_ms) / kStalenessThreshMs;

        peer_mask_out_[valid_count] = 1.0f;
        valid_count++;

        // Aggregate signals for ego features
        if (peer_accel < min_peer_accel) min_peer_accel = peer_accel;

        const float closing = ego_speed - peer_speed;
        if (closing > max_closing_spd) max_closing_spd = closing;

        if (peer_accel <= kBrakingThreshold) any_braking_peer = true;
    }

    // ------------------------------------------------------------------ //
    // 3. Update braking decay                                             //
    // ------------------------------------------------------------------ //
    if (any_braking_peer) {
        braking_decay_ = 1.0f;
    } else {
        braking_decay_ *= kBrakingDecay;
    }

    // ------------------------------------------------------------------ //
    // 4. Build single-frame ego vector                                    //
    // ------------------------------------------------------------------ //
    float single_ego[kEgoSingleDim];
    single_ego[0] = ego_speed                               / kMaxSpeed;
    single_ego[1] = ego_longAccel                           / kMaxAccel;
    single_ego[2] = static_cast<float>(valid_count)         / static_cast<float>(kMaxPeers);
    single_ego[3] = min_peer_accel                          / kMaxAccel;
    single_ego[4] = braking_decay_;
    single_ego[5] = max_closing_spd                         / kMaxSpeed;

    pushFrame(single_ego);
    buildStackedEgo();
}

// ============================================================================
// Private helpers
// ============================================================================

void ObservationBuilder::pushFrame(const float frame[kEgoSingleDim]) {
    if (!history_full_) {
        // Warm-start: fill all slots with the first frame
        for (int s = 0; s < kEgoStackFrames; ++s) {
            memcpy(ego_history_[s], frame, kEgoSingleDim * sizeof(float));
        }
        history_write_ = 0;   // next write will overwrite slot 0 on second call
        history_full_  = true;
    } else {
        memcpy(ego_history_[history_write_], frame, kEgoSingleDim * sizeof(float));
        history_write_ = (history_write_ + 1) % kEgoStackFrames;
    }
}

void ObservationBuilder::buildStackedEgo() {
    // Most-recent frame is at index (history_write_ - 1 + N) % N
    // because history_write_ points to the NEXT write slot (already advanced
    // by pushFrame on the second and subsequent calls).
    for (int f = 0; f < kEgoStackFrames; ++f) {
        const int src = (history_write_ - 1 - f + kEgoStackFrames * 2) % kEgoStackFrames;
        memcpy(&ego_out_[f * kEgoSingleDim],
               ego_history_[src],
               kEgoSingleDim * sizeof(float));
    }
}

bool ObservationBuilder::isInCone(float ego_heading_deg,
                                  float ego_lat_deg, float ego_lon_deg,
                                  float peer_lat_deg, float peer_lon_deg) {
    // Convert lat/lon delta to local Cartesian (East = +x, North = +y)
    const float dy_north = (peer_lat_deg - ego_lat_deg) * kMetresPerDegLat;
    const float cos_lat  = std::cos(ego_lat_deg * kDegToRad);
    const float dx_east  = (peer_lon_deg - ego_lon_deg) * kMetresPerDegLat * cos_lat;

    // Bearing in Cartesian frame (0=East, CCW)
    const float bearing_rad  = std::atan2(dy_north, dx_east);
    const float bearing_deg  = bearing_rad / kDegToRad;

    // Convert to compass bearing (0=North, CW)
    const float peer_compass = std::fmod(90.0f - bearing_deg + 360.0f, 360.0f);

    // Angular difference in [-180, 180]
    const float diff = std::fmod(peer_compass - ego_heading_deg + 180.0f, 360.0f) - 180.0f;

    return std::fabs(diff) <= kConeHalfAngleDeg;
}

void ObservationBuilder::toEgoFrame(float ego_heading_deg,
                                    float ego_lat_deg, float ego_lon_deg,
                                    float peer_lat_deg, float peer_lon_deg,
                                    float& out_rel_x, float& out_rel_y) {
    // Local Cartesian (East = +x, North = +y)
    const float dy_north = (peer_lat_deg - ego_lat_deg) * kMetresPerDegLat;
    const float cos_lat  = std::cos(ego_lat_deg * kDegToRad);
    const float dx_east  = (peer_lon_deg - ego_lon_deg) * kMetresPerDegLat * cos_lat;

    // Ego's Cartesian angle (CCW from East)
    const float ego_phi  = (90.0f - ego_heading_deg) * kDegToRad;
    const float cos_phi  = std::cos(ego_phi);
    const float sin_phi  = std::sin(ego_phi);

    // Rotate to ego frame: rel_x = forward, rel_y = left
    out_rel_x =  dx_east * cos_phi + dy_north * sin_phi;
    out_rel_y = -dx_east * sin_phi + dy_north * cos_phi;
}
