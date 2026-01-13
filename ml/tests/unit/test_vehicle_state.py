"""
Phase 1: VehicleState tests (TDD).
"""

import dataclasses

import pytest

from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage
from envs.sumo_connection import VehicleState


def test_vehicle_state_creation_with_valid_inputs():
    """VehicleState accepts valid position, speed, accel, heading."""
    state = VehicleState(
        vehicle_id="V001",
        x=10.0,
        y=5.0,
        speed=12.5,
        acceleration=0.3,
        heading=90.0,
        lane_position=42.0,
    )

    assert state.vehicle_id == "V001"
    assert state.x == 10.0
    assert state.y == 5.0
    assert state.speed == 12.5
    assert state.acceleration == 0.3
    assert state.heading == 90.0
    assert state.lane_position == 42.0


def test_vehicle_state_is_immutable_frozen_dataclass():
    """VehicleState fields cannot be modified after creation."""
    state = VehicleState(
        vehicle_id="V001",
        x=0.0,
        y=0.0,
        speed=1.0,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        state.speed = 5.0


def test_vehicle_state_equality_same_values_are_equal():
    """Two VehicleState with same values compare equal."""
    state_a = VehicleState(
        vehicle_id="V002",
        x=1.0,
        y=2.0,
        speed=3.0,
        acceleration=0.1,
        heading=180.0,
        lane_position=10.0,
    )
    state_b = VehicleState(
        vehicle_id="V002",
        x=1.0,
        y=2.0,
        speed=3.0,
        acceleration=0.1,
        heading=180.0,
        lane_position=10.0,
    )

    assert state_a == state_b


def test_vehicle_state_repr_contains_vehicle_id():
    """String representation includes vehicle_id for debugging."""
    state = VehicleState(
        vehicle_id="V003",
        x=0.0,
        y=0.0,
        speed=0.0,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )

    assert "V003" in repr(state)


def test_vehicle_state_negative_speed_handled():
    """Negative speed either raises ValueError or clamps to 0."""
    state = VehicleState(
        vehicle_id="V004",
        x=0.0,
        y=0.0,
        speed=-1.0,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )

    assert state.speed == 0.0


def test_to_v2v_message_converts_sumo_coords_to_v2v_format():
    """SUMO x,y converts to V2VMessage lat/lon using meters-per-degree."""
    meters_per_deg = ESPNOWEmulator.METERS_PER_DEG_LAT
    state = VehicleState(
        vehicle_id="V002",
        x=111.0,
        y=222.0,
        speed=12.0,
        acceleration=0.5,
        heading=45.0,
        lane_position=0.0,
    )

    msg = state.to_v2v_message(timestamp_ms=1000)

    assert isinstance(msg, V2VMessage)
    assert msg.lat == pytest.approx(222.0 / meters_per_deg, abs=1e-9)
    assert msg.lon == pytest.approx(111.0 / meters_per_deg, abs=1e-9)


def test_to_v2v_message_sets_timestamp_correctly():
    """Timestamp in message matches provided simulation time."""
    state = VehicleState(
        vehicle_id="V002",
        x=0.0,
        y=0.0,
        speed=0.0,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )

    msg = state.to_v2v_message(timestamp_ms=12345)

    assert msg.timestamp_ms == 12345


def test_to_v2v_message_preserves_speed_and_acceleration():
    """Speed and accel transfer without modification."""
    state = VehicleState(
        vehicle_id="V003",
        x=0.0,
        y=0.0,
        speed=18.5,
        acceleration=-1.2,
        heading=180.0,
        lane_position=0.0,
    )

    msg = state.to_v2v_message(timestamp_ms=2000)

    assert msg.speed == 18.5
    assert msg.accel_x == -1.2
