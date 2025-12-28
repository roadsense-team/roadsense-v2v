"""
Phase 4: Packet Loss Modeling Tests

Tests for ESPNOWEmulator packet loss probability calculation and burst loss logic.

Test Groups:
- A: Distance-Dependent Probability (3-tier model)
- B: Burst Loss Logic (temporal correlation)
- C: Domain Randomization (loss rate variation)

Following TDD methodology - tests written FIRST before implementation verification.

REFACTORING NOTES (Dec 29, 2025 - Phase 8):
- Updated mocking to target emulator._rng instance instead of global random module
"""

import pytest
from unittest.mock import patch
from espnow_emulator.espnow_emulator import ESPNOWEmulator


# ============================================================================
# Test Group A: Distance-Dependent Probability (3-Tier Model)
# ============================================================================

class TestDistanceDependentLoss:
    """
    Test the 3-tier distance-based packet loss probability model.

    Model:
    - Tier 1: d < T1 => P = base_rate
    - Tier 2: T1 <= d < T2 => P = base + (d-T1)/(T2-T1) * (tier2 - base)
    - Tier 3: d >= T2 => P = tier3
    """

    @pytest.fixture
    def emulator_fixed_params(self, tmp_path):
        """Create emulator with explicit, fixed parameters for testing."""
        params = {
            "latency": {
                "base_ms": 15,
                "distance_factor": 0.1,
                "jitter_std_ms": 5,
            },
            "packet_loss": {
                "base_rate": 0.01,           # 1% base loss
                "distance_threshold_1": 50,   # First threshold at 50m
                "distance_threshold_2": 100,  # Second threshold at 100m
                "rate_tier_1": 0.01,          # Same as base (tier 1)
                "rate_tier_2": 0.10,          # 10% loss in transition
                "rate_tier_3": 0.50,          # 50% loss at far range
            },
            "burst_loss": {
                "enabled": False,  # Disable burst for basic distance tests
                "mean_burst_length": 1,
            },
            "sensor_noise": {
                "gps_std_m": 5.0,
                "speed_std_ms": 0.5,
                "accel_std_ms2": 0.2,
                "heading_std_deg": 3.0,
            },
            "domain_randomization": {
                "latency_range_ms": [10, 80],
                "loss_rate_range": [0.0, 0.15],
            }
        }

        # Write params to temp file
        params_file = tmp_path / "test_params.json"
        import json
        with open(params_file, 'w') as f:
            json.dump(params, f)

        # Create emulator with domain randomization DISABLED for deterministic tests
        emulator = ESPNOWEmulator(
            params_file=str(params_file),
            domain_randomization=False
        )

        return emulator

    def test_loss_tier_1_close_range(self, emulator_fixed_params):
        """
        Test Tier 1: Close range (d < threshold_1).

        At d = 10m (< 50m), probability should equal base_rate.
        """
        emulator = emulator_fixed_params
        distance = 10.0  # Well below threshold_1 (50m)

        prob = emulator._get_loss_probability(distance)

        # Should be exactly base_rate (0.01)
        assert prob == pytest.approx(0.01, abs=1e-6)

    def test_loss_tier_2_interpolation(self, emulator_fixed_params):
        """
        Test Tier 2: Transition zone (threshold_1 <= d < threshold_2).

        At d = 75m (midpoint between 50m and 100m), probability should be
        the midpoint between base_rate (0.01) and rate_tier_2 (0.10).

        Expected: 0.01 + (75-50)/(100-50) * (0.10 - 0.01) = 0.01 + 0.5 * 0.09 = 0.055
        """
        emulator = emulator_fixed_params
        distance = 75.0  # Midpoint in transition zone

        prob = emulator._get_loss_probability(distance)

        # Linear interpolation: base + t * (tier2 - base)
        # t = (75 - 50) / (100 - 50) = 0.5
        # prob = 0.01 + 0.5 * (0.10 - 0.01) = 0.055
        expected = 0.055
        assert prob == pytest.approx(expected, abs=1e-6)

    def test_loss_tier_3_far_range(self, emulator_fixed_params):
        """
        Test Tier 3: Far range (d >= threshold_2).

        At d = 150m (> 100m), probability should equal rate_tier_3.
        """
        emulator = emulator_fixed_params
        distance = 150.0  # Well beyond threshold_2 (100m)

        prob = emulator._get_loss_probability(distance)

        # Should be exactly rate_tier_3 (0.50)
        assert prob == pytest.approx(0.50, abs=1e-6)

    def test_loss_boundary_conditions(self, emulator_fixed_params):
        """
        Test exact boundary values at thresholds.

        - At d = 50m (threshold_1): Should start transition (base_rate)
        - At d = 100m (threshold_2): Should reach tier_2 rate
        """
        emulator = emulator_fixed_params

        # Test at threshold_1 (50m) - start of tier 2
        prob_at_t1 = emulator._get_loss_probability(50.0)
        # At start of tier 2: t = 0, so prob = base_rate
        assert prob_at_t1 == pytest.approx(0.01, abs=1e-6)

        # Test at threshold_2 (100m) - start of tier 3
        prob_at_t2 = emulator._get_loss_probability(100.0)
        # At d >= threshold_2, use tier_3 rate
        assert prob_at_t2 == pytest.approx(0.50, abs=1e-6)

        # Test just below threshold_2 (99.9m) - end of tier 2
        prob_just_below_t2 = emulator._get_loss_probability(99.9)
        # t = (99.9 - 50) / (100 - 50) = 0.998
        # prob = 0.01 + 0.998 * (0.10 - 0.01) = 0.01 + 0.08982 = 0.09982
        expected = 0.01 + (99.9 - 50) / (100 - 50) * (0.10 - 0.01)
        assert prob_just_below_t2 == pytest.approx(expected, abs=1e-4)


# ============================================================================
# Test Group B: Burst Loss Logic
# ============================================================================

class TestBurstLoss:
    """
    Test burst loss temporal correlation logic.

    If burst_loss.enabled == True and last packet from vehicle was lost,
    then _check_burst_loss() has 50% chance to return True (hardcoded).
    """

    @pytest.fixture
    def emulator_burst_disabled(self, tmp_path):
        """Create emulator with burst loss DISABLED."""
        params = {
            "packet_loss": {
                "base_rate": 0.02,
                "distance_threshold_1": 50,
                "distance_threshold_2": 100,
                "rate_tier_1": 0.02,
                "rate_tier_2": 0.10,
                "rate_tier_3": 0.40,
            },
            "burst_loss": {
                "enabled": False,  # Explicitly disabled
                "mean_burst_length": 1,
            },
        }

        params_file = tmp_path / "burst_disabled.json"
        import json
        with open(params_file, 'w') as f:
            json.dump(params, f)

        return ESPNOWEmulator(str(params_file), domain_randomization=False)

    @pytest.fixture
    def emulator_burst_enabled(self, tmp_path):
        """Create emulator with burst loss ENABLED."""
        params = {
            "packet_loss": {
                "base_rate": 0.02,
                "distance_threshold_1": 50,
                "distance_threshold_2": 100,
                "rate_tier_1": 0.02,
                "rate_tier_2": 0.10,
                "rate_tier_3": 0.40,
            },
            "burst_loss": {
                "enabled": True,  # Explicitly enabled
                "mean_burst_length": 2, # p_continue = 0.5
            },
        }

        params_file = tmp_path / "burst_enabled.json"
        import json
        with open(params_file, 'w') as f:
            json.dump(params, f)

        return ESPNOWEmulator(str(params_file), domain_randomization=False)

    def test_burst_loss_disabled(self, emulator_burst_disabled):
        """
        Test that burst loss returns False when disabled.

        Even if last packet was lost, burst loss should not trigger.
        """
        emulator = emulator_burst_disabled
        vehicle_id = "V002"

        # Inject previous loss state
        emulator.last_loss_state[vehicle_id] = True

        # Check burst loss (should be False because disabled)
        result = emulator._check_burst_loss(vehicle_id)

        assert result is False, "Burst loss should be False when disabled"

    def test_burst_loss_enabled_logic(self, emulator_burst_enabled):
        """
        Test burst loss logic when enabled.

        If last packet was lost, _check_burst_loss() has 50% chance to return True.
        We mock random.random() to control the outcome.
        """
        emulator = emulator_burst_enabled
        vehicle_id = "V002"

        # Set previous packet as lost
        emulator.last_loss_state[vehicle_id] = True

        # Test 1: Mock random.random() to return 0.3 (< 0.5) => should return True
        # FIXED (Phase 8): Mock emulator's private RNG instance
        with patch.object(emulator._rng, 'random', return_value=0.3):
            result = emulator._check_burst_loss(vehicle_id)
            assert result is True, "Should return True when random < 0.5 and burst enabled"

        # Test 2: Mock random.random() to return 0.7 (> 0.5) => should return False
        # FIXED (Phase 8): Mock emulator's private RNG instance
        with patch.object(emulator._rng, 'random', return_value=0.7):
            result = emulator._check_burst_loss(vehicle_id)
            assert result is False, "Should return False when random > 0.5"

    def test_burst_loss_no_history(self, emulator_burst_enabled):
        """
        Test burst loss when there's no previous loss history.

        If last_loss_state is False or vehicle not in dict, should return False.
        """
        emulator = emulator_burst_enabled
        vehicle_id = "V002"

        # Test 1: Vehicle not in last_loss_state dict
        if vehicle_id in emulator.last_loss_state:
            del emulator.last_loss_state[vehicle_id]

        result = emulator._check_burst_loss(vehicle_id)
        assert result is False, "Should return False when vehicle has no loss history"

        # Test 2: last_loss_state is False (previous packet succeeded)
        emulator.last_loss_state[vehicle_id] = False
        result = emulator._check_burst_loss(vehicle_id)
        assert result is False, "Should return False when last packet was not lost"


# ============================================================================
# Test Group C: Domain Randomization
# ============================================================================

class TestLossDomainRandomization:
    """
    Test that base loss rate varies across episodes when domain randomization is enabled.
    """

    @pytest.fixture
    def emulator_dr_enabled(self, tmp_path):
        """Create emulator with domain randomization ENABLED."""
        params = {
            "packet_loss": {
                "base_rate": 0.02,
                "distance_threshold_1": 50,
                "distance_threshold_2": 100,
                "rate_tier_1": 0.02,
                "rate_tier_2": 0.10,
                "rate_tier_3": 0.40,
            },
            "domain_randomization": {
                "latency_range_ms": [10, 80],
                "loss_rate_range": [0.0, 0.15],  # Loss can vary from 0% to 15%
            },
        }

        params_file = tmp_path / "dr_enabled.json"
        import json
        with open(params_file, 'w') as f:
            json.dump(params, f)

        # Domain randomization ENABLED
        return ESPNOWEmulator(str(params_file), domain_randomization=True)

    def test_loss_domain_randomization(self, emulator_dr_enabled):
        """
        Test that loss probability varies across episodes.

        When domain randomization is enabled, resetting the emulator should
        randomize the base loss rate within the specified range [0.0, 0.15].
        """
        emulator = emulator_dr_enabled

        # Collect base loss rates across 20 episodes
        base_rates = []

        for _ in range(20):
            emulator.reset()
            # At distance=0, tier 1 applies, so we get the episode base rate
            prob = emulator._get_loss_probability(0.0)
            base_rates.append(prob)

        # Verify variation
        unique_rates = set(base_rates)

        # Should have multiple unique values (at least 15 out of 20)
        assert len(unique_rates) >= 15, \
            f"Expected high variation in loss rates, got only {len(unique_rates)} unique values"

        # All rates should be within domain randomization range [0.0, 0.15]
        for rate in base_rates:
            assert 0.0 <= rate <= 0.15, \
                f"Loss rate {rate} outside expected range [0.0, 0.15]"

        # Statistical check: mean should be roughly mid-range (0.075 ± 0.04)
        import statistics
        mean_rate = statistics.mean(base_rates)
        assert 0.03 <= mean_rate <= 0.12, \
            f"Mean loss rate {mean_rate:.4f} should be near mid-range (~0.075)"


# ============================================================================
# Edge Cases and Integration
# ============================================================================

class TestPacketLossEdgeCases:
    """Additional edge case tests for robustness."""

    @pytest.fixture
    def emulator_standard(self, tmp_path):
        """Standard emulator for edge case testing."""
        params = {
            "packet_loss": {
                "base_rate": 0.01,
                "distance_threshold_1": 50,
                "distance_threshold_2": 100,
                "rate_tier_1": 0.01,
                "rate_tier_2": 0.10,
                "rate_tier_3": 0.50,
            },
        }

        params_file = tmp_path / "standard.json"
        import json
        with open(params_file, 'w') as f:
            json.dump(params, f)

        return ESPNOWEmulator(str(params_file), domain_randomization=False)

    def test_loss_at_zero_distance(self, emulator_standard):
        """Test loss probability at exactly 0m distance."""
        prob = emulator_standard._get_loss_probability(0.0)
        assert prob == pytest.approx(0.01, abs=1e-6)

    def test_loss_at_very_large_distance(self, emulator_standard):
        """Test loss probability at extreme distance (1000m)."""
        prob = emulator_standard._get_loss_probability(1000.0)
        # Should be tier_3 rate
        assert prob == pytest.approx(0.50, abs=1e-6)

    def test_loss_probability_always_valid(self, emulator_standard):
        """Test that loss probability is always in valid range [0, 1]."""
        test_distances = [0, 1, 25, 49, 50, 51, 75, 99, 100, 101, 150, 500]

        for dist in test_distances:
            prob = emulator_standard._get_loss_probability(dist)
            assert 0.0 <= prob <= 1.0, \
                f"Probability {prob} at distance {dist}m is outside [0, 1]"