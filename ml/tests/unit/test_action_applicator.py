"""
Unit tests for ActionApplicator continuous action semantics (Phase B).
"""
from unittest.mock import MagicMock

import pytest

from envs.action_applicator import ActionApplicator
from envs.sumo_connection import VehicleState


def _make_sumo(speed: float) -> MagicMock:
    mock_sumo = MagicMock()
    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0.0,
        y=0.0,
        speed=speed,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )
    return mock_sumo


def test_action_applicator_zero_releases_to_cf_model():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(20.0)

    actual_decel = applicator.apply(mock_sumo, 0.0)

    mock_sumo.release_vehicle_speed.assert_called_once_with("V001")
    mock_sumo.set_vehicle_speed.assert_not_called()
    assert actual_decel == pytest.approx(0.0)


def test_action_applicator_below_threshold_releases():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(15.0)

    actual_decel = applicator.apply(mock_sumo, 0.01)

    mock_sumo.release_vehicle_speed.assert_called_once_with("V001")
    mock_sumo.set_vehicle_speed.assert_not_called()
    assert actual_decel == pytest.approx(0.0)


def test_action_applicator_at_threshold_applies_decel():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(20.0)

    # Just above threshold should apply deceleration, not release
    action = applicator.RELEASE_THRESHOLD + 0.01
    actual_decel = applicator.apply(mock_sumo, action)

    mock_sumo.release_vehicle_speed.assert_not_called()
    mock_sumo.set_vehicle_speed.assert_called_once()
    assert actual_decel > 0.0


def test_action_applicator_one_means_max_decel():
    applicator = ActionApplicator()
    assert applicator.MAX_DECEL == pytest.approx(8.0)
    mock_sumo = _make_sumo(20.0)

    actual_decel = applicator.apply(mock_sumo, 1.0)

    expected_new_speed = 20.0 - (applicator.MAX_DECEL * applicator.STEP_DT)
    mock_sumo.set_vehicle_speed.assert_called_once_with("V001", expected_new_speed)
    assert actual_decel == pytest.approx(applicator.MAX_DECEL)


def test_action_applicator_half_means_half_max_decel():
    applicator = ActionApplicator()
    assert 0.5 * applicator.MAX_DECEL == pytest.approx(4.0)
    mock_sumo = _make_sumo(20.0)

    actual_decel = applicator.apply(mock_sumo, 0.5)

    expected_decel = 0.5 * applicator.MAX_DECEL
    expected_new_speed = 20.0 - (expected_decel * applicator.STEP_DT)
    mock_sumo.set_vehicle_speed.assert_called_once_with("V001", expected_new_speed)
    assert actual_decel == pytest.approx(expected_decel)


def test_action_applicator_clips_below_zero():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(10.0)

    actual_decel = applicator.apply(mock_sumo, -0.2)

    mock_sumo.release_vehicle_speed.assert_called_once_with("V001")
    assert actual_decel == pytest.approx(0.0)


def test_action_applicator_clips_above_one():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(10.0)

    actual_decel = applicator.apply(mock_sumo, 1.7)

    expected_new_speed = 10.0 - (applicator.MAX_DECEL * applicator.STEP_DT)
    mock_sumo.set_vehicle_speed.assert_called_once_with("V001", expected_new_speed)
    assert actual_decel == pytest.approx(applicator.MAX_DECEL)


def test_action_applicator_speed_floor_is_zero():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(0.2)

    actual_decel = applicator.apply(mock_sumo, 1.0)

    mock_sumo.set_vehicle_speed.assert_called_once_with("V001", 0.0)
    expected_actual_decel = 0.2 / applicator.STEP_DT
    assert actual_decel == pytest.approx(expected_actual_decel)


def test_action_applicator_returns_actual_decel_applied():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(3.0)

    actual_decel = applicator.apply(mock_sumo, 0.6)

    expected_decel = 0.6 * applicator.MAX_DECEL
    assert actual_decel == pytest.approx(expected_decel)


def test_action_sensitivity_release_vs_max_brake():
    """Confirm action=0 and action=1 produce materially different behavior."""
    applicator = ActionApplicator()

    mock_release = _make_sumo(20.0)
    decel_release = applicator.apply(mock_release, 0.0)

    mock_brake = _make_sumo(20.0)
    decel_brake = applicator.apply(mock_brake, 1.0)

    # Release should call release_vehicle_speed, brake should call set_vehicle_speed
    mock_release.release_vehicle_speed.assert_called_once()
    mock_release.set_vehicle_speed.assert_not_called()
    mock_brake.set_vehicle_speed.assert_called_once()

    # Decel values must be materially different
    assert decel_release == pytest.approx(0.0)
    assert decel_brake == pytest.approx(applicator.MAX_DECEL)
    assert decel_brake - decel_release > 1.0


def test_action_applicator_decel_at_full_action():
    applicator = ActionApplicator()
    mock_sumo = _make_sumo(20.0)

    applicator.apply(mock_sumo, 1.0)

    mock_sumo.set_vehicle_speed.assert_called_once_with("V001", pytest.approx(19.2))
