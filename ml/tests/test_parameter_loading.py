"""
Test suite for ESPNOWEmulator parameter loading and validation.

This module tests Phase 2 functionality:
- Basic initialization (with/without params file)
- File handling edge cases (missing, corrupted)
- Recursive merging (partial config overrides)
- Data validation (ranges, types, logic)

Author: Amir Khalifa
Date: December 26, 2025
"""

import pytest
import json
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from espnow_emulator.espnow_emulator import ESPNOWEmulator


# ============================================================================
# Test Group A: Basic Initialization
# ============================================================================

def test_init_defaults():
    """
    Test initialization with params_file=None uses default parameters.

    Verifies that when no params file is provided, the emulator initializes
    with safe default values from _default_params().
    """
    emulator = ESPNOWEmulator(params_file=None, domain_randomization=False)

    # Verify default latency parameters
    assert emulator.params['latency']['base_ms'] == 15
    assert emulator.params['latency']['distance_factor'] == 0.1
    assert emulator.params['latency']['jitter_std_ms'] == 8

    # Verify default packet loss parameters
    assert emulator.params['packet_loss']['base_rate'] == 0.02
    assert emulator.params['packet_loss']['distance_threshold_1'] == 50
    assert emulator.params['packet_loss']['distance_threshold_2'] == 100

    # Verify burst loss defaults
    assert emulator.params['burst_loss']['enabled'] == False
    assert emulator.params['burst_loss']['mean_burst_length'] == 1

    # Verify domain randomization parameters exist
    assert 'domain_randomization' in emulator.params
    assert 'latency_range_ms' in emulator.params['domain_randomization']


def test_init_with_valid_file():
    """
    Test initialization with a valid params file.

    Verifies that when a valid JSON file is provided, the emulator loads
    the parameters correctly and they override defaults.
    """
    # Use the existing fixture file
    fixture_path = Path(__file__).parent / 'fixtures' / 'emulator_params.json'

    emulator = ESPNOWEmulator(params_file=str(fixture_path), domain_randomization=False)

    # Verify values match the fixture file
    assert emulator.params['latency']['base_ms'] == 12.0
    assert emulator.params['latency']['distance_factor'] == 0.15
    assert emulator.params['latency']['jitter_std_ms'] == 8.0

    assert emulator.params['packet_loss']['base_rate'] == 0.02
    assert emulator.params['packet_loss']['distance_threshold_1'] == 50
    assert emulator.params['packet_loss']['rate_tier_3'] == 0.40

    assert emulator.params['burst_loss']['enabled'] == True
    assert emulator.params['burst_loss']['mean_burst_length'] == 1.8


# ============================================================================
# Test Group B: File Handling Edge Cases
# ============================================================================

def test_init_file_not_found():
    """
    Test initialization with non-existent file raises FileNotFoundError.

    Explicit failure is better than silently falling back to defaults,
    as it helps catch configuration errors during development.
    """
    non_existent_path = '/tmp/nonexistent_emulator_params_12345.json'

    with pytest.raises(FileNotFoundError):
        ESPNOWEmulator(params_file=non_existent_path)


def test_init_corrupted_json():
    """
    Test initialization with corrupted JSON raises JSONDecodeError.

    Invalid JSON syntax should fail immediately with a clear error,
    not cause cryptic errors later during runtime.
    """
    # Create temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{ "latency": { "base_ms": ')  # Incomplete JSON
        temp_path = f.name

    try:
        with pytest.raises(json.JSONDecodeError):
            ESPNOWEmulator(params_file=temp_path)
    finally:
        os.unlink(temp_path)


# ============================================================================
# Test Group C: Recursive Merging (Partial Config)
# ============================================================================

def test_recursive_merge_logic():
    """
    Test recursive merge allows partial JSON files to override defaults.

    This is critical: if a user provides only {"latency": {"base_ms": 999}},
    other latency parameters (jitter_std_ms) should remain at default values.
    This prevents crashes from missing keys.
    """
    # Create temporary file with ONLY latency.base_ms override
    partial_config = {
        "latency": {
            "base_ms": 999
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(partial_config, f)
        temp_path = f.name

    try:
        emulator = ESPNOWEmulator(params_file=temp_path, domain_randomization=False)

        # Verify: base_ms is overridden
        assert emulator.params['latency']['base_ms'] == 999

        # Verify: other latency params still have default values
        assert emulator.params['latency']['jitter_std_ms'] == 8  # Default
        assert emulator.params['latency']['distance_factor'] == 0.1  # Default

        # Verify: packet_loss section completely default (not in partial config)
        assert emulator.params['packet_loss']['base_rate'] == 0.02
        assert emulator.params['packet_loss']['distance_threshold_1'] == 50

    finally:
        os.unlink(temp_path)


# ============================================================================
# Test Group D: Data Validation & Logic
# ============================================================================

def test_invalid_probability_range():
    """
    Test that probability values outside [0, 1] raise ValueError.

    Probabilities must be in [0.0, 1.0]. Values like 1.5 are invalid
    and should be caught during initialization, not during training.
    """
    invalid_config = {
        "packet_loss": {
            "base_rate": 1.5  # INVALID: > 1.0
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Probability.*must be between 0 and 1"):
            ESPNOWEmulator(params_file=temp_path)
    finally:
        os.unlink(temp_path)


def test_negative_latency():
    """
    Test that negative latency values raise ValueError.

    Latency cannot be negative (violates causality). This should be
    caught during validation.
    """
    invalid_config = {
        "latency": {
            "base_ms": -10  # INVALID: negative
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Latency.*must be positive"):
            ESPNOWEmulator(params_file=temp_path)
    finally:
        os.unlink(temp_path)


def test_inverted_thresholds():
    """
    Test that inverted distance thresholds raise ValueError.

    Logical constraint: threshold_1 must be < threshold_2.
    If threshold_1 (100) > threshold_2 (50), the interpolation logic breaks.
    """
    invalid_config = {
        "packet_loss": {
            "distance_threshold_1": 100,
            "distance_threshold_2": 50  # INVALID: threshold_1 > threshold_2
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="distance_threshold_1.*must be less than.*distance_threshold_2"):
            ESPNOWEmulator(params_file=temp_path)
    finally:
        os.unlink(temp_path)


# ============================================================================
# Edge Cases (Bonus Tests)
# ============================================================================

def test_empty_json_uses_defaults():
    """
    Test that an empty JSON file {} results in default parameters.

    This is valid behavior - empty override means "use all defaults".
    """
    empty_config = {}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(empty_config, f)
        temp_path = f.name

    try:
        emulator = ESPNOWEmulator(params_file=temp_path, domain_randomization=False)

        # Should have all default values
        assert emulator.params['latency']['base_ms'] == 15  # Default
        assert emulator.params['packet_loss']['base_rate'] == 0.02  # Default

    finally:
        os.unlink(temp_path)


def test_extra_keys_allowed():
    """
    Test that extra keys in JSON are allowed (forward compatibility).

    If user adds {"future_feature": 1}, emulator should ignore it
    rather than crashing. This supports forward compatibility.
    """
    config_with_extra = {
        "latency": {
            "base_ms": 20
        },
        "future_feature_xyz": {
            "experimental_value": 123
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_with_extra, f)
        temp_path = f.name

    try:
        # Should not raise an error
        emulator = ESPNOWEmulator(params_file=temp_path, domain_randomization=False)

        # Verify known parameter is loaded
        assert emulator.params['latency']['base_ms'] == 20

        # Extra key is either ignored or passed through (implementation choice)
        # We don't care as long as it doesn't crash

    finally:
        os.unlink(temp_path)
