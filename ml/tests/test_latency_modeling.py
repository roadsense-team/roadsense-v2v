"""
Phase 3 Tests: Latency Modeling (REFACTORED)

Tests for the _get_latency() method in ESPNOWEmulator.

Test Coverage:
- Deterministic base latency (mocked jitter)
- Distance scaling factor
- Minimum clamping (1ms floor)
- Statistical distribution validation
- Domain randomization across episodes

Mathematical Model:
    L = max(1.0, L_base + (d × F_dist) + N(0, σ_jitter))

Where:
    - L_base: Base latency (from params or randomized per episode)
    - d: Distance in meters
    - F_dist: Distance factor (ms/meter)
    - N(0, σ): Gaussian jitter with std dev σ

REFACTORING NOTES (Dec 28, 2025):
- Added explicit parameter setting (no dependency on defaults)
- Replaced `==` with `pytest.approx()` for all float comparisons
- Renamed edge case tests to clarify intent
- See: docs/PHASE3_CODE_REVIEW.md

REFACTORING NOTES (Dec 29, 2025 - Phase 8):
- Updated mocking to target emulator._rng instance instead of global random module
- Removed module-level patches
"""

import pytest
import numpy as np
from unittest.mock import patch
from espnow_emulator.espnow_emulator import ESPNOWEmulator


# Test constants (explicit values, not dependent on implementation defaults)
TEST_BASE_MS = 20.0
TEST_DISTANCE_FACTOR = 0.1
TEST_JITTER_STD_MS = 5.0


class TestLatencyDeterministic:
    """
    Test Group A: Deterministic Logic

    FIXED (Phase 8 Review): Mock emulator._rng.gauss instead of global random.gauss
    Tests are decoupled from implementation defaults.
    """

    def test_latency_deterministic_base(self):
        """
        Test: Base latency with zero distance and zero jitter.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Explicitly set test parameters
        emulator.params['latency']['base_ms'] = TEST_BASE_MS
        emulator.params['latency']['distance_factor'] = TEST_DISTANCE_FACTOR
        emulator.params['latency']['jitter_std_ms'] = TEST_JITTER_STD_MS
        emulator.episode_jitter_std = TEST_JITTER_STD_MS

        # Mock the emulator's private RNG instance
        with patch.object(emulator._rng, 'gauss', return_value=0) as mock_gauss:
            result = emulator._get_latency(distance_m=0.0)
            expected = TEST_BASE_MS
            assert result == pytest.approx(expected, abs=1e-6)
            mock_gauss.assert_called_once_with(0, TEST_JITTER_STD_MS)

    def test_latency_distance_scaling(self):
        """
        Test: Distance scaling factor.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        emulator.params['latency']['base_ms'] = TEST_BASE_MS
        emulator.params['latency']['distance_factor'] = TEST_DISTANCE_FACTOR
        emulator.params['latency']['jitter_std_ms'] = TEST_JITTER_STD_MS
        emulator.episode_jitter_std = TEST_JITTER_STD_MS

        with patch.object(emulator._rng, 'gauss', return_value=0):
            distance = 100.0
            result = emulator._get_latency(distance_m=distance)
            expected = TEST_BASE_MS + (distance * TEST_DISTANCE_FACTOR)
            assert result == pytest.approx(expected, abs=1e-6)

    def test_latency_minimum_clamp(self):
        """
        Test: Latency minimum clamping to 1.0ms.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        emulator.params['latency']['base_ms'] = 5.0
        emulator.params['latency']['distance_factor'] = 0.0
        emulator.params['latency']['jitter_std_ms'] = 10.0
        emulator.episode_jitter_std = 10.0

        # Mock jitter to be -25.0, which would make result 5 - 25 = -20
        # Should be clamped to 1.0
        with patch.object(emulator._rng, 'gauss', return_value=-25.0):
            result = emulator._get_latency(distance_m=0.0)
            assert result == pytest.approx(1.0, abs=1e-6)


class TestLatencyStatistical:
    """
    Test Group B: Statistical Validation
    """

    def test_latency_distribution_statistics(self):
        """
        Test: Statistical properties of latency distribution.
        """
        # Set random seed for reproducibility
        import random
        random.seed(42)
        np.random.seed(42)

        # Create emulator with explicit parameters
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Override params
        emulator.params['latency']['base_ms'] = TEST_BASE_MS
        emulator.params['latency']['jitter_std_ms'] = TEST_JITTER_STD_MS
        emulator.params['latency']['distance_factor'] = 0.0
        emulator.episode_jitter_std = TEST_JITTER_STD_MS
        
        # We assume the emulator uses its internal RNG which is seeded by default (random.Random()) 
        # or we can seed it explicitly if the test relies on specific values.
        # But here we are testing statistical properties (mean/std), so any seed is fine 
        # as long as N is large enough.

        # Collect 1000 samples
        n_samples = 1000
        samples = [emulator._get_latency(distance_m=0.0) for _ in range(n_samples)]

        # Calculate statistics
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)

        expected_mean = TEST_BASE_MS
        expected_std = TEST_JITTER_STD_MS

        assert mean == pytest.approx(expected_mean, abs=0.5)
        assert std == pytest.approx(expected_std, abs=0.5)
        assert np.min(samples) >= 1.0


class TestLatencyDomainRandomization:
    """
    Test Group C: Domain Randomization
    """

    def test_latency_domain_randomization_enabled(self):
        """
        Test: Domain randomization varies base latency across episodes.
        """
        emulator = ESPNOWEmulator(domain_randomization=True)

        dr_min = 10.0
        dr_max = 80.0
        emulator.params['domain_randomization']['latency_range_ms'] = [dr_min, dr_max]

        n_episodes = 20
        base_latencies = []

        for _ in range(n_episodes):
            emulator.reset()
            # Mock gauss to 0 to isolate base latency
            with patch.object(emulator._rng, 'gauss', return_value=0):
                latency = emulator._get_latency(distance_m=0.0)
                base_latencies.append(latency)

        for base in base_latencies:
            assert dr_min <= base <= dr_max

        unique_values = len(set(base_latencies))
        assert unique_values > 1

    def test_latency_domain_randomization_disabled(self):
        """
        Test: With DR disabled, base latency is constant across episodes.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)
        emulator.params['latency']['base_ms'] = TEST_BASE_MS

        expected_base = TEST_BASE_MS
        n_episodes = 10
        base_latencies = []

        for _ in range(n_episodes):
            emulator.reset()
            with patch.object(emulator._rng, 'gauss', return_value=0):
                latency = emulator._get_latency(distance_m=0.0)
                base_latencies.append(latency)

        for base in base_latencies:
            assert base == pytest.approx(expected_base, abs=1e-6)

        unique_values = len(set(base_latencies))
        assert unique_values == 1


class TestLatencyEdgeCases:
    """
    Additional edge case tests for robustness.
    """

    def test_latency_edge_case_zero_distance(self):
        """
        Test: Zero distance edge case.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        emulator.params['latency']['base_ms'] = TEST_BASE_MS
        emulator.params['latency']['distance_factor'] = TEST_DISTANCE_FACTOR
        emulator.params['latency']['jitter_std_ms'] = TEST_JITTER_STD_MS
        emulator.episode_jitter_std = TEST_JITTER_STD_MS

        with patch.object(emulator._rng, 'gauss', return_value=0):
            result = emulator._get_latency(distance_m=0.0)
            expected = TEST_BASE_MS
            assert result == pytest.approx(expected, abs=1e-6)

    def test_latency_edge_case_large_distance(self):
        """
        Test: Very large distance.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        emulator.params['latency']['base_ms'] = TEST_BASE_MS
        emulator.params['latency']['distance_factor'] = TEST_DISTANCE_FACTOR
        emulator.params['latency']['jitter_std_ms'] = TEST_JITTER_STD_MS
        emulator.episode_jitter_std = TEST_JITTER_STD_MS

        large_distance = 500.0
        with patch.object(emulator._rng, 'gauss', return_value=0):
            result = emulator._get_latency(distance_m=large_distance)
            expected = TEST_BASE_MS + (large_distance * TEST_DISTANCE_FACTOR)
            assert result == pytest.approx(expected, abs=1e-6)
            assert result > TEST_BASE_MS

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])