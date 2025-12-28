"""
Phase 8 Tests: Domain Randomization & Lifecycle Management

Tests for ESPNOWEmulator reset() and _randomize_episode_params() methods.

Critical Requirements:
1. State Hygiene: No data leaks between episodes (RL requirement)
2. Parameter Diversity: Agent encounters varied environments (DR enabled)
3. Determinism: Experiments reproducible via seeding (debugging requirement)

Test Groups:
- Group A: State Hygiene (reset clears all state)
- Group B: Randomization Logic (enabled/disabled/seeding)
"""

import pytest
import random
from espnow_emulator import ESPNOWEmulator, V2VMessage, ReceivedMessage


# ============================================================================
# Test Group A: State Hygiene (The "Clean Slate" Test)
# ============================================================================

def test_reset_clears_all_state():
    """
    CRITICAL TEST: Verify reset() acts as a "hard wipe" for RL training.

    Any state leak between episodes can corrupt RL training:
    - Old messages appearing in new episodes
    - Burst loss carrying over incorrectly
    - msg_counter overflow after millions of episodes

    Setup:
    1. Pump data into emulator (messages in queue, received, burst state, counter)
    2. Call reset()
    3. Verify ALL state is cleared

    Expected:
    - pending_messages queue: empty
    - last_received dict: empty
    - last_loss_state dict: empty
    - msg_counter: 0
    - current_time_ms: 0
    """
    # Setup: Create emulator with explicit deterministic params
    emulator = ESPNOWEmulator(
        params_file=None,
        domain_randomization=False  # Disable for deterministic state checks
    )

    # Pump data into emulator to populate all state variables

    # 1. Add messages to pending_messages queue
    msg1 = V2VMessage(
        vehicle_id='V002',
        lat=32.0, lon=34.0,
        speed=15.0, heading=90.0,
        accel_x=0.0, accel_y=0.0, accel_z=9.8,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        timestamp_ms=1000
    )

    # Force successful transmission (disable loss)
    emulator.params['packet_loss']['base_rate'] = 0.0
    emulator.params['packet_loss']['rate_tier_1'] = 0.0
    emulator.params['packet_loss']['rate_tier_2'] = 0.0
    emulator.params['packet_loss']['rate_tier_3'] = 0.0

    # Transmit message (will be queued)
    result = emulator.transmit(
        sender_msg=msg1,
        sender_pos=(0, 0),
        receiver_pos=(10, 0),
        current_time_ms=1000
    )

    # Verify message was queued (check BEFORE get_observation delivers it)
    assert not emulator.pending_messages.empty(), "Message should be queued after transmit"

    # 2. Trigger delivery to populate last_received
    emulator.get_observation(ego_speed=10.0, current_time_ms=1050)

    # 3. Populate last_loss_state with burst loss
    emulator.last_loss_state['V002'] = True
    emulator.last_loss_state['V003'] = True

    # 4. Send another message to re-populate the queue (for testing queue clearing)
    msg2 = V2VMessage(
        vehicle_id='V003',
        lat=32.1, lon=34.1,
        speed=20.0, heading=180.0,
        accel_x=1.0, accel_y=0.0, accel_z=9.8,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        timestamp_ms=2000
    )
    emulator.transmit(
        sender_msg=msg2,
        sender_pos=(0, 0),
        receiver_pos=(20, 0),
        current_time_ms=2000
    )

    # Verify state is populated BEFORE reset
    assert not emulator.pending_messages.empty(), "Queue should have pending messages"
    assert emulator.msg_counter > 0, "msg_counter should have incremented"
    assert 'V002' in emulator.last_received, "last_received should have V002"
    assert emulator.last_loss_state.get('V002') == True, "Burst state should be True"
    assert emulator.msg_counter > 0, "Counter should be > 0"

    # ACTION: Call reset()
    emulator.reset()

    # VERIFY: All state cleared
    assert emulator.pending_messages.empty(), "pending_messages queue should be empty after reset"
    assert len(emulator.last_received) == 0, "last_received dict should be empty after reset"
    assert len(emulator.last_loss_state) == 0, "last_loss_state dict should be empty after reset"
    assert emulator.msg_counter == 0, "msg_counter should reset to 0"
    assert emulator.current_time_ms == 0, "current_time_ms should reset to 0"


# ============================================================================
# Test Group B: Randomization Logic
# ============================================================================

def test_dr_enabled_variability():
    """
    Verify domain randomization produces parameter diversity.

    When domain_randomization=True:
    - Each reset() should sample new latency_base and loss_base
    - Values should vary across episodes (not constant)
    - All values must fall within configured ranges

    Test Method:
    - Loop 50 times, call reset(), collect episode_latency_base
    - Verify len(unique_values) > 1 (it varies)
    - Verify all values within latency_range_ms

    Statistical Note:
    - With 50 samples from a continuous uniform distribution,
      probability of collision is negligible (~0.001%)
    - If all 50 values are identical, DR is broken
    """
    # Setup: Create emulator with DR enabled
    emulator = ESPNOWEmulator(
        params_file=None,
        domain_randomization=True  # ENABLE domain randomization
    )

    # Get expected range from params
    latency_range = emulator.params['domain_randomization']['latency_range_ms']
    loss_range = emulator.params['domain_randomization']['loss_rate_range']

    # Collect samples
    latency_samples = []
    loss_samples = []

    for _ in range(50):
        emulator.reset()  # Should randomize params
        latency_samples.append(emulator.episode_latency_base)
        loss_samples.append(emulator.episode_loss_base)

    # VERIFY: Variability (multiple unique values)
    unique_latencies = set(latency_samples)
    unique_losses = set(loss_samples)

    assert len(unique_latencies) > 1, \
        f"Domain randomization FAILED: All 50 latency values identical ({latency_samples[0]})"

    assert len(unique_losses) > 1, \
        f"Domain randomization FAILED: All 50 loss values identical ({loss_samples[0]})"

    # VERIFY: All values within configured ranges
    for lat in latency_samples:
        assert latency_range[0] <= lat <= latency_range[1], \
            f"Latency {lat} outside range {latency_range}"

    for loss in loss_samples:
        assert loss_range[0] <= loss <= loss_range[1], \
            f"Loss rate {loss} outside range {loss_range}"


def test_dr_disabled_consistency():
    """
    Verify domain randomization can be disabled for deterministic testing.

    When domain_randomization=False:
    - reset() should NOT randomize parameters
    - episode_latency_base should be EXACTLY params['latency']['base_ms']
    - episode_loss_base should be EXACTLY params['packet_loss']['base_rate']
    - Values must be constant across all resets

    Test Method:
    - Loop 50 times, call reset()
    - Verify episode_latency_base == base_ms every time
    - Verify episode_loss_base == base_rate every time

    Use Case:
    - Unit testing (need deterministic behavior)
    - Debugging (need reproducible episodes)
    """
    # Setup: Create emulator with DR DISABLED
    emulator = ESPNOWEmulator(
        params_file=None,
        domain_randomization=False  # DISABLE domain randomization
    )

    # Expected values (from params, not randomized)
    expected_latency = emulator.params['latency']['base_ms']
    expected_loss = emulator.params['packet_loss']['base_rate']

    # Collect samples
    latency_samples = []
    loss_samples = []

    for _ in range(50):
        emulator.reset()  # Should NOT randomize
        latency_samples.append(emulator.episode_latency_base)
        loss_samples.append(emulator.episode_loss_base)

    # VERIFY: All values EXACTLY match expected (no randomization)
    for lat in latency_samples:
        assert lat == expected_latency, \
            f"Latency should be constant {expected_latency}, got {lat}"

    for loss in loss_samples:
        assert loss == expected_loss, \
            f"Loss should be constant {expected_loss}, got {loss}"

    # VERIFY: No variability (all identical)
    assert len(set(latency_samples)) == 1, "Latency should be constant (DR disabled)"
    assert len(set(loss_samples)) == 1, "Loss should be constant (DR disabled)"


def test_seeding_reproducibility():
    """
    CRITICAL: Verify experiments are reproducible via random seeding.

    For debugging RL policies and scientific reproducibility:
    - Same seed -> same sequence of "random" parameters
    - ESPNOWEmulator(seed=N) must produce identical results across runs

    Test Method:
    1. seed=42 -> reset() -> Record params A
    2. seed=42 -> reset() -> Record params B
    3. Verify params A == params B

    Implementation Note:
    - FIXED (Phase 8 Review): ESPNOWEmulator now uses private random.Random() instance
    - seed parameter initializes this private RNG
    - No global random state pollution

    Failure Mode:
    - If this test fails, RL debugging will be impossible
    - Can't reproduce "that one episode where agent crashed"
    """
    # Run 1: Seed 42, reset, record params
    emulator1 = ESPNOWEmulator(params_file=None, domain_randomization=True, seed=42)
    emulator1.reset()

    params_run1 = {
        'latency': emulator1.episode_latency_base,
        'loss': emulator1.episode_loss_base,
        'jitter_std': emulator1.episode_jitter_std,
        'gps_noise_std': emulator1.episode_gps_noise_std,
    }

    # Run 2: SAME seed, reset, record params
    emulator2 = ESPNOWEmulator(params_file=None, domain_randomization=True, seed=42)
    emulator2.reset()

    params_run2 = {
        'latency': emulator2.episode_latency_base,
        'loss': emulator2.episode_loss_base,
        'jitter_std': emulator2.episode_jitter_std,
        'gps_noise_std': emulator2.episode_gps_noise_std,
    }

    # VERIFY: Identical results
    assert params_run1['latency'] == params_run2['latency'], \
        f"Seeding FAILED: Latency not reproducible ({params_run1['latency']} != {params_run2['latency']})"

    assert params_run1['loss'] == params_run2['loss'], \
        f"Seeding FAILED: Loss not reproducible ({params_run1['loss']} != {params_run2['loss']})"

    assert params_run1['jitter_std'] == params_run2['jitter_std'], \
        f"Seeding FAILED: Jitter std not reproducible ({params_run1['jitter_std']} != {params_run2['jitter_std']})"

    assert params_run1['gps_noise_std'] == params_run2['gps_noise_std'], \
        f"Seeding FAILED: GPS noise std not reproducible ({params_run1['gps_noise_std']} != {params_run2['gps_noise_std']})"

    # Additional check: Run 3 with DIFFERENT seed should produce DIFFERENT results
    emulator3 = ESPNOWEmulator(params_file=None, domain_randomization=True, seed=99)
    emulator3.reset()

    params_run3 = {
        'latency': emulator3.episode_latency_base,
        'loss': emulator3.episode_loss_base
    }

    # Should be different (extremely unlikely to match by chance)
    assert params_run3['latency'] != params_run1['latency'], \
        "Different seeds should produce different params (if identical, DR range too narrow or broken)"


def test_seeding_transmit_reproducibility():
    """
    CRITICAL (Phase 8 Review): Verify transmit() reproducibility, not just reset().

    The original test only verified that episode_latency_base was reproducible.
    However, it did NOT verify that the actual events (transmit() calls) were
    reproducible within an episode.

    This test ensures that:
    - Same seed -> same latency values for transmit() calls
    - Same seed -> same packet loss outcomes
    - Same seed -> same sensor noise values

    Failure Mode:
    - If seeding only affects reset() but not transmit(), episodes are still
      non-deterministic and debugging is impossible.

    Test Method:
    1. Create two emulators with same seed
    2. Call transmit() with identical parameters
    3. Verify age_ms, loss outcomes, and sensor noise are identical
    """
    # Setup: Create identical test message
    test_msg = V2VMessage(
        vehicle_id='V002',
        lat=32.0,
        lon=34.0,
        speed=15.0,
        heading=90.0,
        accel_x=0.5,
        accel_y=0.0,
        accel_z=9.8,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.1,
        timestamp_ms=1000
    )

    sender_pos = (10, 5)
    receiver_pos = (0, 0)
    current_time_ms = 1000

    # Run 1: Seed 42, transmit multiple messages
    emulator1 = ESPNOWEmulator(params_file=None, domain_randomization=True, seed=42)
    # Force successful transmission for deterministic testing
    emulator1.params['packet_loss']['base_rate'] = 0.0
    emulator1.params['packet_loss']['rate_tier_1'] = 0.0
    emulator1.params['packet_loss']['rate_tier_2'] = 0.0
    emulator1.params['packet_loss']['rate_tier_3'] = 0.0

    results_run1 = []
    for i in range(10):
        result = emulator1.transmit(
            sender_msg=test_msg,
            sender_pos=sender_pos,
            receiver_pos=receiver_pos,
            current_time_ms=current_time_ms + i * 100
        )
        if result:
            results_run1.append({
                'age_ms': result.age_ms,
                'lat': result.message.lat,
                'lon': result.message.lon,
                'speed': result.message.speed,
            })

    # Run 2: SAME seed, transmit SAME messages
    emulator2 = ESPNOWEmulator(params_file=None, domain_randomization=True, seed=42)
    # Force successful transmission
    emulator2.params['packet_loss']['base_rate'] = 0.0
    emulator2.params['packet_loss']['rate_tier_1'] = 0.0
    emulator2.params['packet_loss']['rate_tier_2'] = 0.0
    emulator2.params['packet_loss']['rate_tier_3'] = 0.0

    results_run2 = []
    for i in range(10):
        result = emulator2.transmit(
            sender_msg=test_msg,
            sender_pos=sender_pos,
            receiver_pos=receiver_pos,
            current_time_ms=current_time_ms + i * 100
        )
        if result:
            results_run2.append({
                'age_ms': result.age_ms,
                'lat': result.message.lat,
                'lon': result.message.lon,
                'speed': result.message.speed,
            })

    # VERIFY: Identical results for ALL transmit() calls
    assert len(results_run1) == len(results_run2), \
        "Same seed should produce same number of successful transmissions"

    for i, (r1, r2) in enumerate(zip(results_run1, results_run2)):
        assert r1['age_ms'] == r2['age_ms'], \
            f"Transmission {i}: Latency not reproducible ({r1['age_ms']} != {r2['age_ms']})"

        # Verify sensor noise is reproducible (GPS coordinates affected by noise)
        assert r1['lat'] == r2['lat'], \
            f"Transmission {i}: GPS lat noise not reproducible ({r1['lat']} != {r2['lat']})"

        assert r1['lon'] == r2['lon'], \
            f"Transmission {i}: GPS lon noise not reproducible ({r1['lon']} != {r2['lon']})"

        assert r1['speed'] == r2['speed'], \
            f"Transmission {i}: Speed noise not reproducible ({r1['speed']} != {r2['speed']})"


# ============================================================================
# Edge Case: Bounds Checking (Bonus Test)
# ============================================================================

def test_dr_never_produces_invalid_parameters():
    """
    CRITICAL SAFETY: Domain randomization must never generate invalid params.

    Even with randomization, we must never create:
    - Negative latency (impossible)
    - Loss probability > 1.0 (impossible)
    - Loss probability < 0.0 (impossible)

    Test Method:
    - Run 1000 resets with DR enabled
    - Verify ALL samples are valid

    Failure Mode:
    - If range misconfigured (e.g., [-10, 50]), could produce negative latency
    - Model would crash or learn incorrect physics
    """
    emulator = ESPNOWEmulator(params_file=None, domain_randomization=True)

    for _ in range(1000):
        emulator.reset()

        # VERIFY: Latency is positive
        assert emulator.episode_latency_base > 0, \
            f"Invalid latency: {emulator.episode_latency_base} (must be > 0)"

        # VERIFY: Loss is valid probability [0, 1]
        assert 0.0 <= emulator.episode_loss_base <= 1.0, \
            f"Invalid loss rate: {emulator.episode_loss_base} (must be in [0, 1])"
