/**
 * ObservationBuilder.h
 *
 * Converts live ESP32 sensor/V2V data into the observation tensors
 * consumed by ModelRunner::infer().
 *
 * Observation format matches the Python training environment exactly:
 *
 *   ego[18]  — 3 stacked frames × 6 features, most-recent-first:
 *     [0] speed / 30.0
 *     [1] longAccel / 10.0
 *     [2] valid_peer_count / 8.0
 *     [3] min_peer_accel / 10.0
 *     [4] braking_received_decay  (1.0 → 0.95^n per 10 Hz step)
 *     [5] max_closing_speed / 30.0
 *
 *   peers[8][6]  — per peer (cone-filtered, most-recent first):
 *     [0] rel_x_forward / 100.0
 *     [1] rel_y_left    / 100.0
 *     [2] (peer.speed - ego.speed) / 30.0
 *     [3] rel_heading_deg / 180.0
 *     [4] peer.longAccel / 10.0
 *     [5] age_ms / 500.0
 *
 *   peer_mask[8]  — 1.0 for valid peers, 0.0 for empty slots
 *
 * Constants:
 *   BRAKING_THRESHOLD = -2.5 m/s²   (peer accel below this → braking event)
 *   BRAKING_DECAY     = 0.95 / step  (10 Hz → half-life ~1.4 s)
 *   CONE_HALF_ANGLE   = 45°          (front-cone filter)
 *
 * Usage:
 *   ObservationBuilder obs;
 *   obs.reset();                          // once in setup()
 *   obs.update(ownMsg, packageManager, ownMAC);  // at 10 Hz in loop()
 *   float action = runner.infer(obs.ego(),
 *                               (const float(*)[kPeerFeatureDim])obs.peers(),
 *                               obs.peerMask());
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#include "../network/protocol/V2VMessage.h"
#include "../network/mesh/PackageManager.h"
#include "ModelRunner.h"  // for kEgoDim, kMaxPeers, kPeerFeatureDim

class ObservationBuilder {
public:
    static constexpr int   kEgoSingleDim      = 6;
    static constexpr int   kEgoStackFrames    = 3;

    // Normalization constants — must match Python ObservationBuilder
    static constexpr float kMaxSpeed          = 30.0f;
    static constexpr float kMaxAccel          = 10.0f;
    static constexpr float kMaxDistance       = 100.0f;
    static constexpr float kStalenessThreshMs = 500.0f;
    static constexpr float kConeHalfAngleDeg  = 45.0f;
    static constexpr float kBrakingThreshold  = -4.0f;   // m/s² — accel_y (longitudinal) baseline ~-3.2 at rest
    static constexpr float kBrakingDecay      = 0.95f;   // per 10 Hz step

    ObservationBuilder() { reset(); }

    /**
     * Clear frame history and braking decay.
     * Call once from setup().
     */
    void reset();

    /**
     * Build ego/peers/peer_mask from current state.
     * Must be called at 10 Hz to maintain correct decay cadence.
     *
     * @param ownMsg  Most-recent V2VMessage built from own sensors.
     * @param pm      PackageManager holding received peer messages.
     * @param ownMAC  This device's MAC address (6 bytes), used to skip self.
     */
    void update(const V2VMessage& ownMsg,
                const PackageManager& pm,
                const uint8_t ownMAC[6]);

    /** Pointer to stacked ego tensor [kEgoDim = 18]. */
    const float* ego()      const { return ego_out_; }

    /** Pointer to peers tensor flattened [kMaxPeers * kPeerFeatureDim = 48]. */
    const float* peers()    const { return &peers_out_[0][0]; }

    /** Pointer to peer_mask tensor [kMaxPeers = 8]. */
    const float* peerMask() const { return peer_mask_out_; }

    // --- Diagnostic getters (valid after each update() call) ---
    /** Number of valid (cone-filtered, non-stale) peers seen this step. */
    int   activePeerCount()   const { return last_active_peers_; }
    /** True if any peer's longAccel crossed kBrakingThreshold this step. */
    bool  isBrakingPeer()     const { return last_braking_peer_; }
    /** Current braking decay value [0..1] — 1.0 = fresh brake, decays at 0.95/step. */
    float brakingDecay()      const { return braking_decay_; }
    /** Maximum closing speed (ego - peer) across valid peers this step, m/s. */
    float maxClosingSpeed()   const { return last_max_closing_spd_; }
    /** Distance to nearest valid peer in ego frame, metres (0 if no peers). */
    float nearestPeerM()      const { return last_nearest_m_; }
    /** Minimum longAccel across valid peers this step (most negative = hardest brake). */
    float minPeerAccel()      const { return last_min_peer_accel_; }

private:
    float ego_out_[kEgoDim];                          // [18]
    float peers_out_[kMaxPeers][kPeerFeatureDim];     // [8][6]
    float peer_mask_out_[kMaxPeers];                  // [8]

    // Circular buffer for ego frame stacking
    float ego_history_[kEgoStackFrames][kEgoSingleDim];
    int   history_write_ = 0;   // next write index (0..2)
    bool  history_full_  = false;

    float braking_decay_ = 0.0f;
    float last_heading_  = 0.0f;  // cached when GPS speed too low

    // Diagnostic state — updated at end of each update() call
    int   last_active_peers_    = 0;
    bool  last_braking_peer_    = false;
    float last_max_closing_spd_ = 0.0f;
    float last_nearest_m_       = 0.0f;
    float last_min_peer_accel_  = 0.0f;

    /**
     * Push a new single-frame ego vector and advance the ring buffer.
     * On the very first call, fills all three slots with this frame
     * (same warm-start as the Python env's reset()).
     */
    void pushFrame(const float frame[kEgoSingleDim]);

    /**
     * Write stacked ego (most-recent first) into ego_out_.
     */
    void buildStackedEgo();

    /**
     * True when the peer is inside the ego's 90° forward cone.
     * Uses GPS lat/lon for bearing computation.
     */
    static bool isInCone(float ego_heading_deg,
                         float ego_lat_deg, float ego_lon_deg,
                         float peer_lat_deg, float peer_lon_deg);

    /**
     * Convert GPS lat/lon delta to ego-relative (forward, left) metres.
     */
    static void toEgoFrame(float ego_heading_deg,
                           float ego_lat_deg, float ego_lon_deg,
                           float peer_lat_deg, float peer_lon_deg,
                           float& out_rel_x, float& out_rel_y);
};
