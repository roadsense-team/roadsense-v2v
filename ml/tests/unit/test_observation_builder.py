"""
Unit tests for ObservationBuilder (Phase 3) - Corrected for SUMO Frame.
"""
import numpy as np
import pytest

from envs.observation_builder import ObservationBuilder
from envs.sumo_connection import VehicleState


def _make_ego_state(speed: float = 15.0, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> VehicleState:
    return VehicleState(
        vehicle_id="V001",
        x=x,
        y=y,
        speed=speed,
        acceleration=0.0,
        heading=heading,
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
    ego_state = _make_ego_state() # Heading 0 (North)
    # Peers North of ego
    peers = [_make_peer_obs(0.0, 10.0), _make_peer_obs(2.0, 20.0)]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert isinstance(result, dict)
    assert result["ego"].shape == (5,)
    assert result["peers"].shape == (builder.MAX_PEERS, 6)
    assert result["peer_mask"].shape == (builder.MAX_PEERS,)


def test_build_observation_dtype_is_float32():
    """All observation values use float32 dtype."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    peers = [_make_peer_obs(0.0, 10.0)]

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
    # Peers North of ego (Forward)
    peers = [
        _make_peer_obs(0.0, 10.0, valid=True),
        _make_peer_obs(0.0, 20.0, valid=True),
        _make_peer_obs(0.0, 30.0, valid=True),
    ]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(3.0)


def test_build_observation_skips_invalid_peers():
    """Invalid peers are masked out."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state()
    # Peer North (Forward)
    peers = [
        _make_peer_obs(0.0, 10.0, valid=True),
        _make_peer_obs(0.0, 20.0, valid=False),
    ]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(1.0)


def test_build_observation_zero_peers_is_well_formed():
    """With no peers at all, observation is valid with all-zero peer data."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(speed=10.0)

    result = builder.build(ego_state, [], (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(0.0)
    assert np.all(result["peers"] == 0.0)
    assert np.isfinite(result["ego"]).all()
    assert result["ego"][3] == pytest.approx(0.0)  # peer_count/MAX_PEERS = 0
    assert result["ego"][4] == pytest.approx(0.0)  # min_peer_accel = 0 when no peers


def test_build_observation_peers_behind_ego_filtered_by_cone():
    """Peers directly behind ego are excluded by cone filter."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(heading=0.0)  # Facing North
    # Peer directly South (behind)
    peers = [_make_peer_obs(0.0, -50.0, valid=True)]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(0.0)


def test_build_observation_min_peer_accel_reflects_braking():
    """min_peer_accel in ego vector reflects the most negative peer accel."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(heading=0.0)
    # Two peers ahead, one braking hard
    peers = [
        _make_peer_obs(0.0, 10.0, accel=-3.0, valid=True),
        _make_peer_obs(0.0, 20.0, accel=-7.0, valid=True),
    ]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    # min_peer_accel = -7.0, normalized by MAX_ACCEL=10 -> -0.7
    assert result["ego"][4] == pytest.approx(-0.7)


def test_build_observation_min_peer_accel_ignores_behind():
    """min_peer_accel only considers cone-filtered (forward) peers."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(heading=0.0)  # Facing North
    # One peer behind braking hard, one ahead not braking
    peers = [
        _make_peer_obs(0.0, -50.0, accel=-8.0, valid=True),  # behind
        _make_peer_obs(0.0, 20.0, accel=-1.0, valid=True),   # ahead
    ]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    # Behind peer filtered by cone, only ahead peer counted
    assert result["ego"][4] == pytest.approx(-0.1)


def test_build_observation_all_stale_peers_excluded():
    """Peers with age above staleness threshold are excluded."""
    builder = ObservationBuilder()
    ego_state = _make_ego_state(heading=0.0)
    # Peer ahead but with stale data (age > 500ms threshold)
    peers = [_make_peer_obs(0.0, 30.0, valid=False, age_ms=600)]

    result = builder.build(ego_state, peers, (ego_state.x, ego_state.y))

    assert result["peer_mask"].sum() == pytest.approx(0.0)
