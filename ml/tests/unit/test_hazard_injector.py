"""
Unit tests for HazardInjector.
"""
from unittest.mock import Mock

import pytest

from envs.hazard_injector import HazardInjector
from envs.sumo_connection import VehicleState


def _make_vehicle_state(vehicle_id, x, y, speed=10.0, heading=0.0):
    return VehicleState(
        vehicle_id=vehicle_id, x=x, y=y,
        speed=speed, acceleration=0.0, heading=heading, lane_position=0.0,
    )


def _make_sumo(vehicle_states):
    """Mock SUMOConnection with given vehicles."""
    mock = Mock()
    ids = [vs.vehicle_id for vs in vehicle_states]
    state_map = {vs.vehicle_id: vs for vs in vehicle_states}

    mock.get_active_vehicle_ids.return_value = ids
    mock.is_vehicle_active.side_effect = lambda vid: vid in ids
    mock.get_vehicle_state.side_effect = lambda vid: state_map[vid]
    mock.set_vehicle_speed = Mock()
    return mock


@pytest.fixture
def mock_sumo():
    """Mock SUMOConnection with V001 ego and V002 nearest front peer."""
    return _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
    ])


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


def test_hazard_targets_nearest_peer(mock_sumo):
    """Nearest strategy targets nearest front peer."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=mock_sumo)

    mock_sumo.set_vehicle_speed.assert_called_once_with("V002", 0.0)
    assert inj.hazard_target == "V002"


def test_hazard_targets_closest_among_multiple_peers():
    """With multiple peers, targets the nearest one."""
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=30.0),
        _make_vehicle_state("V003", x=0.0, y=12.0),
        _make_vehicle_state("V004", x=0.0, y=50.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=sumo)

    sumo.set_vehicle_speed.assert_called_once_with("V003", 0.0)
    assert inj.hazard_target == "V003"
    assert inj.hazard_source_rank_ahead == 1


def test_hazard_skips_when_no_peers():
    """No injection if ego is the only vehicle."""
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is False
    sumo.set_vehicle_speed.assert_not_called()


def test_hazard_skips_when_ego_not_active():
    """No injection if ego is not in the simulation."""
    sumo = _make_sumo([
        _make_vehicle_state("V002", x=10.0, y=0.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is False
    sumo.set_vehicle_speed.assert_not_called()


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


def test_hazard_injector_only_fires_once(mock_sumo):
    """Hazard is injected at most once per episode."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=mock_sumo)
    inj.maybe_inject(step=50, sumo=mock_sumo)

    assert mock_sumo.set_vehicle_speed.call_count == 1


def test_hazard_reset_clears_target():
    """Reset clears the previous target."""
    inj = HazardInjector(seed=42)
    inj._hazard_target = "V003"
    inj.reset()
    assert inj.hazard_target is None


def test_hazard_ignores_peers_behind_ego():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=-10.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is False
    sumo.set_vehicle_speed.assert_not_called()


def test_uniform_front_peers_samples_multiple_sources():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
        _make_vehicle_state("V004", x=0.0, y=30.0),
    ])

    sampled = set()
    for seed in range(80):
        inj = HazardInjector(
            seed=seed,
            target_strategy=HazardInjector.TARGET_STRATEGY_UNIFORM_FRONT_PEERS,
        )
        inj._episode_will_have_hazard = True
        inj._hazard_step = 50
        inj.maybe_inject(step=50, sumo=sumo)
        sampled.add(inj.hazard_target)

    assert sampled >= {"V002", "V003", "V004"}


def test_fixed_vehicle_id_targets_requested_front_vehicle():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        fixed_vehicle_id="V003",
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is True
    sumo.set_vehicle_speed.assert_called_once_with("V003", 0.0)
    assert inj.hazard_source_rank_ahead == 2


def test_fixed_vehicle_id_missing_returns_false():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        fixed_vehicle_id="V099",
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    assert inj.maybe_inject(step=50, sumo=sumo) is False
    sumo.set_vehicle_speed.assert_not_called()


def test_fixed_vehicle_id_behind_ego_returns_false():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=-10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        fixed_vehicle_id="V002",
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    assert inj.maybe_inject(step=50, sumo=sumo) is False
    sumo.set_vehicle_speed.assert_not_called()


def test_fixed_rank_ahead_targets_requested_rank():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
        _make_vehicle_state("V004", x=0.0, y=30.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_RANK_AHEAD,
        fixed_rank_ahead=2,
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is True
    sumo.set_vehicle_speed.assert_called_once_with("V003", 0.0)
    assert inj.hazard_source_rank_ahead == 2


def test_reset_options_override_target_strategy_and_schedule():
    inj = HazardInjector(seed=1)

    inj.reset(
        options={
            "target_strategy": HazardInjector.TARGET_STRATEGY_FIXED_RANK_AHEAD,
            "fixed_rank_ahead": 3,
            "force_hazard": True,
            "hazard_step": 40,
        }
    )

    assert inj.target_strategy == HazardInjector.TARGET_STRATEGY_FIXED_RANK_AHEAD
    assert inj.hazard_step == 40
    assert inj._episode_will_have_hazard is True
