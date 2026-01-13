"""
Unit tests for ObservationBuilder (Phase 3).
"""
import numpy as np
import pytest

from envs.observation_builder import ObservationBuilder
from envs.sumo_connection import VehicleState


def _make_ego_state(speed: float = 15.0, x: float = 0.0, y: float = 0.0) -> VehicleState:
    return VehicleState(
        vehicle_id="V001",
        x=x,
        y=y,
        speed=speed,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )


def _make_emulator_obs(
    v002_valid: bool = True,
    v003_valid: bool = True,
    v002_age_ms: int = 100,
    v003_age_ms: int = 200,
    v002_lat: float = 0.0001,
    v002_lon: float = 0.0002,
    v003_lat: float = 0.0003,
    v003_lon: float = 0.0004,
) -> dict:
    return {
        "ego_speed": 15.0,
        "timestamp_ms": 1000,
        "v002_lat": v002_lat,
        "v002_lon": v002_lon,
        "v002_speed": 18.0,
        "v002_heading": 0.0,
        "v002_accel_x": 1.0,
        "v002_age_ms": v002_age_ms,
        "v002_valid": v002_valid,
        "v003_lat": v003_lat,
        "v003_lon": v003_lon,
        "v003_speed": 20.0,
        "v003_heading": 0.0,
        "v003_accel_x": -1.0,
        "v003_age_ms": v003_age_ms,
        "v003_valid": v003_valid,
    }


def test_build_observation_returns_ndarray_shape_11():
    """Returns np.ndarray with shape (11,)."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    obs = _make_emulator_obs()

    result = builder.build(ego_state, obs)

    assert isinstance(result, np.ndarray)
    assert result.shape == (11,)


def test_build_observation_dtype_is_float32():
    """All observation values use float32 dtype."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    obs = _make_emulator_obs()

    result = builder.build(ego_state, obs)

    assert result.dtype == np.float32


def test_build_observation_ego_speed_normalized():
    """Ego speed is normalized by max speed (30 m/s)."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(speed=15.0)
    obs = _make_emulator_obs()

    result = builder.build(ego_state, obs)

    assert result[0] == pytest.approx(0.5)


def test_build_observation_stale_message_valid_is_zero():
    """Stale messages set valid flag to 0.0."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    obs = _make_emulator_obs(v002_valid=False, v002_age_ms=600)

    result = builder.build(ego_state, obs)

    assert result[5] == 0.0


def test_build_observation_missing_vehicle_uses_defaults():
    """Missing vehicle data defaults to zeros."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    obs = _make_emulator_obs(
        v003_valid=False,
        v003_age_ms=9999,
        v003_lat=0.0,
        v003_lon=0.0,
    )
    obs["v003_speed"] = 0.0
    obs["v003_accel_x"] = 0.0

    result = builder.build(ego_state, obs)

    assert np.allclose(result[6:11], 0.0)
