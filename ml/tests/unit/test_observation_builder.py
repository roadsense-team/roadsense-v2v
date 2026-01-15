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


def _make_peer_obs(
    x: float,
    y: float,
    speed: float = 18.0,
    heading: float = 0.0,
    accel: float = 1.0,
    age_ms: int = 100,
    valid: bool = True,
) -> dict:
    return {
        "x": x,
        "y": y,
        "speed": speed,
        "heading": heading,
        "accel": accel,
        "age_ms": age_ms,
        "valid": valid,
    }


def test_build_observation_returns_dict_shapes():
    """Returns dict with expected shapes."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    peers = [_make_peer_obs(10.0, 0.0), _make_peer_obs(20.0, 5.0)]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert isinstance(result, dict)
    assert result["ego"].shape == (4,)
    assert result["peers"].shape == (builder.MAX_PEERS, 6)
    assert result["peer_mask"].shape == (builder.MAX_PEERS,)


def test_build_observation_dtype_is_float32():
    """All observation values use float32 dtype."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    peers = [_make_peer_obs(10.0, 0.0)]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["ego"].dtype == np.float32
    assert result["peers"].dtype == np.float32
    assert result["peer_mask"].dtype == np.float32


def test_build_observation_ego_speed_normalized():
    """Ego speed is normalized by max speed (30 m/s)."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(speed=15.0)

    result = builder.build(ego_state, [], (ego_state.x, ego_state.y))

    assert result["ego"][0] == pytest.approx(0.5)


def test_build_observation_peer_mask_counts_valid_peers():
    """Peer mask has one entry per valid peer."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    peers = [
        _make_peer_obs(10.0, 0.0, valid=True),
        _make_peer_obs(20.0, 0.0, valid=True),
        _make_peer_obs(30.0, 0.0, valid=True),
    ]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(3.0)


def test_build_observation_skips_invalid_peers():
    """Invalid peers are masked out."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    peers = [
        _make_peer_obs(10.0, 0.0, valid=True),
        _make_peer_obs(20.0, 0.0, valid=False),
    ]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(1.0)
