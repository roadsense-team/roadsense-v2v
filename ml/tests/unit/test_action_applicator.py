"""
Unit tests for ActionApplicator (Phase 4).
"""
from unittest.mock import MagicMock

import pytest

from envs.action_applicator import ActionApplicator
from envs.sumo_connection import VehicleState


def test_action_applicator_maintain_returns_zero_decel():
    """Action 0 (MAINTAIN) produces 0.0 deceleration."""
    applicator = ActionApplicator()
    assert applicator.get_deceleration(0) == 0.0


def test_action_applicator_caution_returns_0_5_decel():
    """Action 1 (CAUTION) produces 0.5 m/s speed reduction."""
    applicator = ActionApplicator()
    assert applicator.get_deceleration(1) == 0.5


def test_action_applicator_brake_returns_2_0_decel():
    """Action 2 (BRAKE) produces 2.0 m/s speed reduction."""
    applicator = ActionApplicator()
    assert applicator.get_deceleration(2) == 2.0


def test_action_applicator_emergency_returns_4_5_decel():
    """Action 3 (EMERGENCY) produces 4.5 m/s speed reduction."""
    applicator = ActionApplicator()
    assert applicator.get_deceleration(3) == 4.5


def test_action_applicator_invalid_action_raises():
    """Action outside [0,3] raises ValueError."""
    applicator = ActionApplicator()
    with pytest.raises(ValueError):
        applicator.get_deceleration(4)
    with pytest.raises(ValueError):
        applicator.get_deceleration(-1)


def test_action_applicator_speed_floor_is_zero():
    """Speed cannot go negative after action."""
    applicator = ActionApplicator()
    mock_sumo = MagicMock()
    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0,
        y=0,
        speed=2.0,
        acceleration=0,
        heading=0,
        lane_position=0,
    )

    actual_decel = applicator.apply(mock_sumo, ActionApplicator.EMERGENCY)

    mock_sumo.set_vehicle_speed.assert_called_with("V001", 0.0)
    assert actual_decel == 2.0


def test_apply_action_calls_set_vehicle_speed():
    """ActionApplicator.apply() calls sumo.set_vehicle_speed()."""
    applicator = ActionApplicator()
    mock_sumo = MagicMock()
    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0,
        y=0,
        speed=20.0,
        acceleration=0,
        heading=0,
        lane_position=0,
    )

    applicator.apply(mock_sumo, ActionApplicator.BRAKE)

    mock_sumo.set_vehicle_speed.assert_called_once()


def test_apply_emergency_on_moving_vehicle_reduces_speed():
    """Emergency brake on 20m/s vehicle results in 15.5m/s."""
    applicator = ActionApplicator()
    mock_sumo = MagicMock()
    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0,
        y=0,
        speed=20.0,
        acceleration=0,
        heading=0,
        lane_position=0,
    )

    applicator.apply(mock_sumo, ActionApplicator.EMERGENCY)

    mock_sumo.set_vehicle_speed.assert_called_with("V001", 15.5)


def test_apply_maintain_does_not_change_speed():
    """MAINTAIN action leaves speed unchanged."""
    applicator = ActionApplicator()
    mock_sumo = MagicMock()
    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0,
        y=0,
        speed=15.0,
        acceleration=0,
        heading=0,
        lane_position=0,
    )

    applicator.apply(mock_sumo, ActionApplicator.MAINTAIN)

    mock_sumo.set_vehicle_speed.assert_called_with("V001", 15.0)


def test_sequential_brake_actions_accumulate():
    """Multiple BRAKE actions continue reducing speed."""
    applicator = ActionApplicator()
    mock_sumo = MagicMock()

    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0,
        y=0,
        speed=20.0,
        acceleration=0,
        heading=0,
        lane_position=0,
    )
    applicator.apply(mock_sumo, ActionApplicator.BRAKE)
    mock_sumo.set_vehicle_speed.assert_called_with("V001", 18.0)

    mock_sumo.get_vehicle_state.return_value = VehicleState(
        vehicle_id="V001",
        x=0,
        y=0,
        speed=18.0,
        acceleration=0,
        heading=0,
        lane_position=0,
    )
    applicator.apply(mock_sumo, ActionApplicator.BRAKE)
    mock_sumo.set_vehicle_speed.assert_called_with("V001", 16.0)
