"""
Unit tests for ObservationBuilder cone filtering (Phase A) - Corrected for SUMO Frame.
"""

import math
import pytest
from envs.observation_builder import ObservationBuilder
from envs.sumo_connection import VehicleState


def _make_ego_state(
    heading: float = 0.0,
    speed: float = 15.0,
    x: float = 0.0,
    y: float = 0.0,
) -> VehicleState:
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


def test_cone_filter_peer_directly_ahead_north_is_in_cone():
    builder = ObservationBuilder()
    # Heading 0 (North), Peer at (0, 10) (North of ego)
    assert builder.is_in_cone(0.0, (0.0, 0.0), 0.0, 10.0) is True


def test_cone_filter_peer_directly_ahead_east_is_in_cone():
    builder = ObservationBuilder()
    # Heading 90 (East), Peer at (10, 0) (East of ego)
    assert builder.is_in_cone(90.0, (0.0, 0.0), 10.0, 0.0) is True


def test_cone_filter_peer_directly_behind_is_out_of_cone():
    builder = ObservationBuilder()
    # Heading 0 (North), Peer at (0, -10) (South of ego)
    assert builder.is_in_cone(0.0, (0.0, 0.0), 0.0, -10.0) is False


def test_cone_filter_peer_at_edge_45_deg_is_in_cone():
    builder = ObservationBuilder()
    # Heading 0 (North), Peer at 45 deg clockwise (North-East)
    # (x=sin(45), y=cos(45))
    x = 10.0 * math.sin(math.radians(45.0))
    y = 10.0 * math.cos(math.radians(45.0))
    assert builder.is_in_cone(0.0, (0.0, 0.0), x, y) is True


def test_cone_filter_peer_at_46_deg_is_out_of_cone():
    builder = ObservationBuilder()
    # Heading 0 (North), Peer at 46 deg clockwise
    x = 10.0 * math.sin(math.radians(46.0))
    y = 10.0 * math.cos(math.radians(46.0))
    assert builder.is_in_cone(0.0, (0.0, 0.0), x, y) is False


def test_cone_filter_peer_count_reduced_after_filtering():
    builder = ObservationBuilder()
    ego_state = _make_ego_state(heading=0.0) # North
    peers = [
        _make_peer_obs(0.0, 10.0),   # Ahead (North)
        _make_peer_obs(0.0, -10.0),  # Behind (South)
        _make_peer_obs(1.0, 8.0),    # Ahead-ish (North-East)
    ]

    observation = builder.build(ego_state, peers, (ego_state.x, ego_state.y))
    # Should keep Ahead and Ahead-ish
    assert observation["peer_mask"].sum() == pytest.approx(2.0)


def test_cone_filter_heading_wraps_around_360():
    builder = ObservationBuilder()
    # Heading 359 (Almost North), Peer at (0, 10) (North)
    assert builder.is_in_cone(359.0, (0.0, 0.0), 0.0, 10.0) is True


def test_relative_coordinates_aligned_with_heading():
    builder = ObservationBuilder()
    ego_state = _make_ego_state(heading=90.0) # East
    # Peer 10m directly ahead of ego (which is facing East)
    peers = [_make_peer_obs(10.0, 0.0)] 
    
    observation = builder.build(ego_state, peers, (0.0, 0.0))
    # rel_x should be 10.0 / MAX_DISTANCE = 0.1
    # rel_y should be 0.0
    assert observation["peers"][0, 0] == pytest.approx(0.1)
    assert observation["peers"][0, 1] == pytest.approx(0.0)
