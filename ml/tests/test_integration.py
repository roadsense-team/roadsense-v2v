"""
Phase 9 Integration Tests: Statistical Verification & Sim2Real Validation

These tests validate that the ESPNOWEmulator accurately reproduces the statistical
properties of real-world ESP-NOW measurements over large sample sizes (Monte Carlo).

Test Groups:
    A. Statistical Correctness (Fixed Params) - Verify latency/loss distributions
    B. Distance Modeling Integration - Verify distance-dependent behavior
    C. Domain Randomization Integration - Verify episode-level variation

Author: Amir Khalifa
Date: December 29, 2025
Phase: 9 - Integration & Sim2Real Validation
"""

import pytest
import numpy as np
from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

@pytest.fixture
def create_test_message():
    """Factory function to create test V2VMessage instances."""
    def _create(vehicle_id='V002', lat=32.0, lon=34.0, timestamp_ms=0):
        return V2VMessage(
            vehicle_id=vehicle_id,
            lat=lat,
            lon=lon,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=timestamp_ms
        )
    return _create


def create_fixed_params_emulator(base_ms=20, distance_factor=0, jitter_std_ms=5, loss_rate=0.05):
    """
    Create emulator with fixed parameters for statistical testing.

    Disables domain randomization and sets explicit parameter values.
    Uses seed=42 for deterministic behavior.
    """
    params = {
        'latency': {
            'base_ms': base_ms,
            'distance_factor': distance_factor,
            'jitter_std_ms': jitter_std_ms,
        },
        'packet_loss': {
            'base_rate': loss_rate,
            'distance_threshold_1': 1000,  # Very high - disable distance-dependent loss
            'distance_threshold_2': 2000,
            'rate_tier_1': loss_rate,
            'rate_tier_2': loss_rate,
            'rate_tier_3': loss_rate,
        },
        'burst_loss': {
            'enabled': False,
        },
        'sensor_noise': {
            'gps_std_m': 0.0,  # Disable noise for latency testing
            'speed_std_ms': 0.0,
            'accel_std_ms2': 0.0,
            'heading_std_deg': 0.0,
            'gyro_std_rad_s': 0.0,
        },
        'domain_randomization': {
            'latency_range_ms': [base_ms, base_ms],  # Fixed value
            'loss_rate_range': [loss_rate, loss_rate],
            'jitter_std_range_ms': [jitter_std_ms, jitter_std_ms],
            'gps_noise_range_m': [0.0, 0.0],
        }
    }

    # Save params to temp file
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(params, f)
        params_file = f.name

    # Create emulator with fixed params (DR disabled implicitly by equal ranges)
    emulator = ESPNOWEmulator(params_file=params_file, domain_randomization=True, seed=42)

    # Clean up temp file
    import os
    os.unlink(params_file)

    return emulator


# ============================================================================
# Test Group A: Statistical Correctness (Fixed Params)
# ============================================================================

def test_integration_latency_statistics(create_test_message):
    """
    Test A.1: Verify latency distribution over large sample size.

    Setup: Disable DR, set Base=20ms, Distance=0, Jitter=5ms
    Action: Simulate 10,000 transmissions
    Verify:
        - Observed Mean ≈ 20.0ms (±0.5ms tolerance for statistical variation)
        - Observed Std Dev ≈ 5.0ms (±0.5ms tolerance)
        - Observed Min >= 1.0ms (clamping)

    CRITICAL: This validates the core latency model matches theoretical distribution.
    """
    N = 10000  # Large sample for statistical reliability

    # Create emulator with fixed latency parameters
    emulator = create_fixed_params_emulator(
        base_ms=20.0,
        distance_factor=0.0,  # No distance dependency
        jitter_std_ms=5.0,
        loss_rate=0.0  # No loss - we want all packets
    )

    # Collect latency samples
    latencies = []

    for i in range(N):
        msg = create_test_message(timestamp_ms=i * 100)
        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(0, 0),  # Zero distance
            receiver_pos=(0, 0),
            current_time_ms=i * 100
        )

        # Should never be None (loss_rate=0)
        assert result is not None, f"Packet lost at iteration {i} (loss_rate=0, should never happen)"
        latencies.append(result.age_ms)

    # Statistical verification
    latencies = np.array(latencies)

    observed_mean = np.mean(latencies)
    observed_std = np.std(latencies)
    observed_min = np.min(latencies)

    # Verify mean ≈ 20.0ms (±0.5ms tolerance)
    # With N=10000, sampling error should be ~5/sqrt(10000) = 0.05ms, so ±0.5ms is generous
    assert 19.5 <= observed_mean <= 20.5, \
        f"Latency mean {observed_mean:.2f}ms outside expected range [19.5, 20.5]ms"

    # Verify std dev ≈ 5.0ms (±0.5ms tolerance)
    # Std dev has higher sampling variance, so ±0.5ms is reasonable
    assert 4.5 <= observed_std <= 5.5, \
        f"Latency std {observed_std:.2f}ms outside expected range [4.5, 5.5]ms"

    # Verify minimum clamping (latency >= 1.0ms)
    assert observed_min >= 1.0, \
        f"Minimum latency {observed_min:.2f}ms violates clamping constraint (must be >= 1.0ms)"

    print(f"✅ Latency Statistics (N={N}):")
    print(f"   Mean: {observed_mean:.2f}ms (expected: 20.0ms)")
    print(f"   Std:  {observed_std:.2f}ms (expected: 5.0ms)")
    print(f"   Min:  {observed_min:.2f}ms (expected: >= 1.0ms)")


def test_integration_packet_loss_rate(create_test_message):
    """
    Test A.2: Verify packet loss rate converges to expected probability.

    Setup: Disable DR, set Loss=0.05 (5%)
    Action: Simulate 10,000 transmissions
    Verify: Loss count between 400 and 600 (4-6% tolerance)

    CRITICAL: This validates stochastic loss model over large N.
    """
    N = 10000

    # Create emulator with 5% loss rate
    emulator = create_fixed_params_emulator(
        base_ms=20.0,
        distance_factor=0.0,
        jitter_std_ms=5.0,
        loss_rate=0.05  # 5% loss
    )

    # Count packet loss
    loss_count = 0
    success_count = 0

    for i in range(N):
        msg = create_test_message(timestamp_ms=i * 100)
        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(0, 0),
            receiver_pos=(0, 0),
            current_time_ms=i * 100
        )

        if result is None:
            loss_count += 1
        else:
            success_count += 1

    # Calculate observed loss rate
    observed_loss_rate = loss_count / N

    # Verify loss rate ≈ 5% (±1% tolerance)
    # With N=10000, binomial standard error = sqrt(0.05*0.95/10000) ≈ 0.0022 (0.22%)
    # So ±1% tolerance is very conservative
    assert 0.04 <= observed_loss_rate <= 0.06, \
        f"Packet loss rate {observed_loss_rate*100:.2f}% outside expected range [4%, 6%]"

    # Verify counts for debugging
    assert loss_count + success_count == N, "Total count mismatch"

    print(f"✅ Packet Loss Statistics (N={N}):")
    print(f"   Lost:     {loss_count} packets")
    print(f"   Received: {success_count} packets")
    print(f"   Loss Rate: {observed_loss_rate*100:.2f}% (expected: 5.0%)")


# ============================================================================
# Test Group B: Distance Modeling Integration
# ============================================================================

def test_integration_distance_scaling(create_test_message):
    """
    Test B.1: Verify distance-dependent latency scaling.

    Setup: Disable DR, set distance_factor=0.1ms/m, Base=10ms
    Action: Run 1000 tx at 0m, 1000 tx at 100m
    Verify: Mean(100m) - Mean(0m) ≈ 10.0ms (±1.0ms tolerance)

    CRITICAL: This validates distance factor calibration.
    """
    N = 1000  # Per distance

    # Create emulator with distance-dependent latency
    emulator = create_fixed_params_emulator(
        base_ms=10.0,
        distance_factor=0.1,  # 0.1ms per meter
        jitter_std_ms=5.0,
        loss_rate=0.0  # No loss
    )

    # Collect latencies at 0m
    latencies_0m = []
    for i in range(N):
        msg = create_test_message(timestamp_ms=i * 100)
        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(0, 0),  # Zero distance
            receiver_pos=(0, 0),
            current_time_ms=i * 100
        )
        assert result is not None
        latencies_0m.append(result.age_ms)

    # Collect latencies at 100m
    latencies_100m = []
    for i in range(N):
        msg = create_test_message(timestamp_ms=i * 100)
        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(100, 0),  # 100m distance
            receiver_pos=(0, 0),
            current_time_ms=i * 100
        )
        assert result is not None
        latencies_100m.append(result.age_ms)

    # Calculate means
    mean_0m = np.mean(latencies_0m)
    mean_100m = np.mean(latencies_100m)
    delta = mean_100m - mean_0m

    # Verify delta ≈ 10.0ms (100m * 0.1ms/m = 10ms)
    # ±1.0ms tolerance accounts for jitter cancellation across large N
    expected_delta = 10.0
    assert expected_delta - 1.0 <= delta <= expected_delta + 1.0, \
        f"Distance scaling delta {delta:.2f}ms outside expected range [9.0, 11.0]ms"

    print(f"✅ Distance Scaling (N={N} per distance):")
    print(f"   Mean @ 0m:   {mean_0m:.2f}ms")
    print(f"   Mean @ 100m: {mean_100m:.2f}ms")
    print(f"   Delta:       {delta:.2f}ms (expected: 10.0ms)")


# ============================================================================
# Test Group C: Domain Randomization Integration
# ============================================================================

def test_integration_dr_variation(create_test_message):
    """
    Test C.1: Verify domain randomization produces episode-level variation.

    Action: Run 20 episodes. In each episode, run 100 tx and calculate mean latency.
    Verify:
        - The standard deviation of the *means* is significant (> 0)
        - All means fall within the DR range [10, 80]ms

    CRITICAL: This validates DR is working and training will see varied conditions.
    """
    N_EPISODES = 20
    N_SAMPLES_PER_EPISODE = 100

    # Create emulator with DR enabled (wide range)
    params = {
        'latency': {
            'base_ms': 20.0,  # Nominal value (DR will override)
            'distance_factor': 0.0,
            'jitter_std_ms': 5.0,
        },
        'packet_loss': {
            'base_rate': 0.0,  # No loss for simplicity
            'distance_threshold_1': 1000,
            'distance_threshold_2': 2000,
            'rate_tier_1': 0.0,
            'rate_tier_2': 0.0,
            'rate_tier_3': 0.0,
        },
        'burst_loss': {'enabled': False},
        'sensor_noise': {
            'gps_std_m': 0.0,
            'speed_std_ms': 0.0,
            'accel_std_ms2': 0.0,
            'heading_std_deg': 0.0,
            'gyro_std_rad_s': 0.0,
        },
        'domain_randomization': {
            'latency_range_ms': [10, 80],  # Wide range for clear variation
            'loss_rate_range': [0.0, 0.0],
            'jitter_std_range_ms': [5, 5],  # Fixed jitter for simplicity
            'gps_noise_range_m': [0.0, 0.0],
        }
    }

    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(params, f)
        params_file = f.name

    emulator = ESPNOWEmulator(params_file=params_file, domain_randomization=True, seed=42)

    import os
    os.unlink(params_file)

    # Run multiple episodes and collect mean latencies
    episode_means = []

    for episode in range(N_EPISODES):
        # Reset for new episode (re-randomizes parameters)
        emulator.reset()

        # Collect latencies for this episode
        latencies = []
        for i in range(N_SAMPLES_PER_EPISODE):
            msg = create_test_message(timestamp_ms=i * 100)
            result = emulator.transmit(
                sender_msg=msg,
                sender_pos=(0, 0),
                receiver_pos=(0, 0),
                current_time_ms=i * 100
            )
            assert result is not None
            latencies.append(result.age_ms)

        # Calculate mean for this episode
        episode_mean = np.mean(latencies)
        episode_means.append(episode_mean)

        # Verify this episode's mean is within DR range
        assert 10.0 <= episode_mean <= 80.0, \
            f"Episode {episode} mean {episode_mean:.2f}ms outside DR range [10, 80]ms"

    # Verify variation across episodes
    episode_means = np.array(episode_means)
    std_of_means = np.std(episode_means)

    # Std dev of means should be significant (not all episodes identical)
    # With range [10, 80], uniform distribution would give std ≈ 20ms
    # We require std > 5ms to confirm variation is happening
    assert std_of_means > 5.0, \
        f"Std dev of episode means {std_of_means:.2f}ms too low (DR not producing variation)"

    print(f"✅ Domain Randomization (N={N_EPISODES} episodes, {N_SAMPLES_PER_EPISODE} samples each):")
    print(f"   Episode means: min={np.min(episode_means):.2f}ms, max={np.max(episode_means):.2f}ms")
    print(f"   Std of means:  {std_of_means:.2f}ms (confirms variation)")
    print(f"   All means within DR range [10, 80]ms: ✅")


# ============================================================================
# Pytest Markers (Optional)
# ============================================================================

# Mark all integration tests as 'slow' for optional filtering in CI
pytestmark = pytest.mark.slow
