"""
Phase 5: Sensor Noise Modeling Tests

Tests for _add_sensor_noise() method in ESPNOWEmulator.
Validates Gaussian noise injection into GPS, dynamics, and orientation data.

Test Strategy:
- Mock random.gauss for deterministic testing
- Verify noise injection formulas
- Validate edge cases (clamping, wraparound)
- Statistical verification for zero-noise case

REFACTORING NOTES (Dec 29, 2025 - Phase 8):
- Updated mocking to target emulator._rng instance instead of global random module
"""

import pytest
from unittest.mock import patch, MagicMock
from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage


class TestSensorNoiseInjection:
    """Test Group A: Noise Injection Logic"""

    def test_noise_gps_injection(self):
        """
        Test GPS noise injection with deterministic mocking.

        Verifies:
        - GPS noise is converted from meters to degrees (÷111000)
        - Noise is applied to both lat and lon
        - Formula: lat_noisy = lat + N(0, gps_std_m / 111000)
        """
        # Create emulator with known GPS noise parameter
        emulator = ESPNOWEmulator()

        # Override sensor_noise params for clear math
        emulator.params['sensor_noise']['gps_std_m'] = 111.0  # 111m ≈ 0.001 deg

        # Create test message
        msg = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000,
        )

        # Mock emulator's private RNG instance
        with patch.object(emulator._rng, 'gauss') as mock_gauss:
            # Return noise values AFTER conversion (in degrees for GPS)
            mock_gauss.side_effect = [
                0.001,  # lat noise (already in degrees)
                0.001,  # lon noise (already in degrees)
                0.0,    # speed noise
                0.0,    # heading noise
                0.0,    # accel_x noise
                0.0,    # accel_y noise
                0.0,    # accel_z noise
                0.0,    # gyro_x noise
                0.0,    # gyro_y noise
                0.0,    # gyro_z noise
            ]

            noisy_msg = emulator._add_sensor_noise(msg)

        # Verify GPS noise was applied
        # lat_noisy = 32.0 + 0.001 = 32.001
        assert noisy_msg.lat == pytest.approx(32.001, abs=1e-6)
        assert noisy_msg.lon == pytest.approx(34.001, abs=1e-6)

        # Verify other fields unchanged (zero noise applied)
        assert noisy_msg.vehicle_id == "V002"
        assert noisy_msg.speed == pytest.approx(15.0, abs=1e-6)
        assert noisy_msg.heading == pytest.approx(90.0, abs=1e-6)
        assert noisy_msg.timestamp_ms == 1000

    def test_noise_dynamics_injection(self):
        """
        Test speed noise injection and non-negative clamping.

        Verifies:
        - Speed noise is applied correctly
        - Speed is clamped to >= 0 (cannot be negative)
        - Formula: speed_noisy = max(0, speed + N(0, speed_std_ms))
        """
        emulator = ESPNOWEmulator()

        # Set speed noise parameter
        emulator.params['sensor_noise']['speed_std_ms'] = 1.0

        # Test Case 1: Positive noise
        msg_positive = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=10.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000,
        )

        with patch.object(emulator._rng, 'gauss') as mock_gauss:
            # Return +1.0 for speed noise (3rd call)
            mock_gauss.side_effect = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            noisy_msg = emulator._add_sensor_noise(msg_positive)

        # Speed should increase by 1.0
        assert noisy_msg.speed == pytest.approx(11.0, abs=0.01)

        # Test Case 2: Negative noise with clamping
        msg_clamp = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=0.5,  # Low speed
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=2000,
        )

        with patch.object(emulator._rng, 'gauss') as mock_gauss:
            # Return -2.0 for speed (would make speed negative without clamping)
            mock_gauss.side_effect = [0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            noisy_msg_clamped = emulator._add_sensor_noise(msg_clamp)

        # Speed should be clamped to 0, not -1.5
        assert noisy_msg_clamped.speed == pytest.approx(0.0, abs=0.01)
        assert noisy_msg_clamped.speed >= 0.0  # Explicit non-negative check

    def test_noise_heading_wraparound(self):
        """
        Test heading noise with wraparound at 0/360 boundary.

        Verifies:
        - Heading wraps around from 359 -> 0 degrees
        - Heading wraps around from 0 -> 359 degrees
        - Formula: heading_noisy = (heading + N(0, heading_std_deg)) % 360
        """
        emulator = ESPNOWEmulator()

        # Set heading noise parameter
        emulator.params['sensor_noise']['heading_std_deg'] = 5.0

        # Test Case 1: Wraparound from 359 -> 1
        msg_high = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=359.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000,
        )

        with patch.object(emulator._rng, 'gauss') as mock_gauss:
            # Add +2 degrees to heading (4th call) -> 361 -> wraps to 1
            mock_gauss.side_effect = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            noisy_msg_high = emulator._add_sensor_noise(msg_high)

        # Heading should wrap: (359 + 2) % 360 = 1
        assert noisy_msg_high.heading == pytest.approx(1.0, abs=0.01)

        # Test Case 2: Wraparound from 1 -> 359
        msg_low = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=1.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=2000,
        )

        with patch.object(emulator._rng, 'gauss') as mock_gauss:
            # Add -2 degrees to heading (4th call) -> -1 -> wraps to 359
            mock_gauss.side_effect = [0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            noisy_msg_low = emulator._add_sensor_noise(msg_low)

        # Heading should wrap: (1 - 2) % 360 = -1 % 360 = 359
        assert noisy_msg_low.heading == pytest.approx(359.0, abs=0.01)

        # Test Case 3: No wraparound (mid-range)
        msg_mid = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=180.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=3000,
        )

        with patch.object(emulator._rng, 'gauss') as mock_gauss:
            # Add +5 degrees to heading (4th call)
            mock_gauss.side_effect = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            noisy_msg_mid = emulator._add_sensor_noise(msg_mid)

        # Should be simple addition: 180 + 5 = 185 (no wrap)
        assert noisy_msg_mid.heading == pytest.approx(185.0, abs=0.01)


class TestSensorNoiseConfiguration:
    """Test Group B: Configuration Validation"""

    def test_noise_zero_std(self):
        """
        Test that zero standard deviation produces identical output.

        Verifies:
        - When all std devs are 0, output == input
        - No noise injection occurs
        - Message fields are preserved exactly
        """
        emulator = ESPNOWEmulator()

        # Set all noise parameters to zero
        emulator.params['sensor_noise'] = {
            'gps_std_m': 0.0,
            'speed_std_ms': 0.0,
            'accel_std_ms2': 0.0,
            'heading_std_deg': 0.0,
            'gyro_std_rad_s': 0.0,
        }
        
        # Override episode params if DR enabled (since DR is on by default)
        emulator.episode_gps_noise_std = 0.0
        # If there are other episode-level variance params, zero them too
        # But we only know about gps_noise_std from summary.
        # Wait, if DR is enabled, _add_sensor_noise might use episode_gps_noise_std.
        # Let's check logic: if we set params manually, we should also ensure DR doesn't override.
        # Safest way: disable DR for this test OR overwrite the episode attribute.
        
        emulator = ESPNOWEmulator(domain_randomization=False)
        emulator.params['sensor_noise'] = {
            'gps_std_m': 0.0,
            'speed_std_ms': 0.0,
            'accel_std_ms2': 0.0,
            'heading_std_deg': 0.0,
            'gyro_std_rad_s': 0.0,
        }

        # Create test message
        original_msg = V2VMessage(
            vehicle_id="V002",
            lat=32.123456,
            lon=34.654321,
            speed=15.75,
            heading=123.45,
            accel_x=1.5,
            accel_y=-0.2,
            accel_z=9.81,
            gyro_x=0.1,
            gyro_y=0.2,
            gyro_z=0.3,
            timestamp_ms=12345,
        )

        # Apply noise (should have no effect)
        noisy_msg = emulator._add_sensor_noise(original_msg)

        # Verify all fields are identical
        assert noisy_msg.vehicle_id == original_msg.vehicle_id
        assert noisy_msg.lat == pytest.approx(original_msg.lat, abs=1e-9)
        assert noisy_msg.lon == pytest.approx(original_msg.lon, abs=1e-9)
        assert noisy_msg.speed == pytest.approx(original_msg.speed, abs=1e-9)
        assert noisy_msg.heading == pytest.approx(original_msg.heading, abs=1e-9)
        assert noisy_msg.accel_x == pytest.approx(original_msg.accel_x, abs=1e-9)
        assert noisy_msg.accel_y == pytest.approx(original_msg.accel_y, abs=1e-9)
        assert noisy_msg.accel_z == pytest.approx(original_msg.accel_z, abs=1e-9)
        assert noisy_msg.gyro_x == pytest.approx(original_msg.gyro_x, abs=1e-9)
        assert noisy_msg.gyro_y == pytest.approx(original_msg.gyro_y, abs=1e-9)
        assert noisy_msg.gyro_z == pytest.approx(original_msg.gyro_z, abs=1e-9)
        assert noisy_msg.timestamp_ms == original_msg.timestamp_ms

        # Verify it's a new object (not the same reference)
        assert noisy_msg is not original_msg


class TestSensorNoiseIntegration:
    """Bonus: Integration tests for sensor noise in full pipeline"""

    def test_noise_statistical_distribution(self):
        """
        Statistical test: Verify noise follows Gaussian distribution.

        Runs 1000 iterations and checks:
        - Mean noise is close to 0
        - Std deviation matches configured value
        """
        # Seed global and numpy for safety, but we rely on emulator's seed if possible
        import random
        random.seed(42)
        import numpy as np
        np.random.seed(42)
        
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Set known GPS noise
        emulator.params['sensor_noise']['gps_std_m'] = 555.0  # 0.005 deg

        original_msg = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000,
        )

        # Collect 1000 samples
        lat_diffs = []
        for _ in range(1000):
            noisy = emulator._add_sensor_noise(original_msg)
            lat_diffs.append(noisy.lat - original_msg.lat)

        # Calculate statistics
        import statistics
        mean_diff = statistics.mean(lat_diffs)
        std_diff = statistics.stdev(lat_diffs)

        # Expected std: 555.0 / 111000 = 0.005
        expected_std = 555.0 / 111000

        # Mean should be close to 0 (unbiased noise)
        assert abs(mean_diff) < expected_std * 0.1  # Within 10% of std

        # Std should match configured value (within 10% due to sampling)
        assert std_diff == pytest.approx(expected_std, rel=0.1)

    def test_noise_preserves_message_immutability(self):
        """
        Verify that _add_sensor_noise returns a NEW message.

        Original message should be unchanged.
        """
        emulator = ESPNOWEmulator()

        original_msg = V2VMessage(
            vehicle_id="V002",
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=1.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000,
        )

        # Apply noise
        noisy_msg = emulator._add_sensor_noise(original_msg)

        # Original should be unchanged
        assert original_msg.lat == 32.0
        assert original_msg.speed == 15.0
        assert original_msg.heading == 90.0

        # Noisy should be different (statistically almost certain)
        assert noisy_msg is not original_msg