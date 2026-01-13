"""
Unit tests for HazardInjector (Phase 6).
"""
from unittest.mock import Mock

import pytest

from envs.hazard_injector import HazardInjector


@pytest.fixture
def mock_sumo():
    """Mock SUMOConnection."""
    mock = Mock()
    mock.is_vehicle_active = Mock(return_value=True)
    mock.set_vehicle_speed = Mock()
    return mock


@pytest.fixture
def injector():
    """HazardInjector with fixed seed."""
    return HazardInjector(seed=42)


def test_hazard_injector_respects_probability():
    """~30% of episodes get a hazard (statistical test)."""
    hazard_count = 0

    for i in range(1000):
        inj = HazardInjector(seed=i)
        inj.reset()
        if inj._episode_will_have_hazard:
            hazard_count += 1

    assert 200 <= hazard_count <= 400, f"Got {hazard_count}, expected ~300"


def test_hazard_injector_only_injects_in_window(mock_sumo):
    """Hazards only injected between steps 30-80."""
    inj = HazardInjector(seed=42)

    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    assert inj.maybe_inject(step=10, sumo=mock_sumo) is False
    assert inj.maybe_inject(step=29, sumo=mock_sumo) is False
    assert inj.maybe_inject(step=81, sumo=mock_sumo) is False

    assert inj.maybe_inject(step=50, sumo=mock_sumo) is True


def test_hazard_injector_emergency_brake_stops_lead(mock_sumo):
    """EMERGENCY_BRAKE hazard sets lead (V002) speed to 0."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=mock_sumo)

    mock_sumo.set_vehicle_speed.assert_called_once_with("V002", 0.0)


def test_hazard_injector_returns_true_when_injected(mock_sumo):
    """maybe_inject() returns True if hazard occurred."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    result_before = inj.maybe_inject(step=49, sumo=mock_sumo)
    assert result_before is False

    result_at = inj.maybe_inject(step=50, sumo=mock_sumo)
    assert result_at is True

    result_after = inj.maybe_inject(step=51, sumo=mock_sumo)
    assert result_after is False
