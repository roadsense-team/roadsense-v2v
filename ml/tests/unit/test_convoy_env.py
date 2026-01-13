"""
Unit tests for ConvoyEnv (Phase 6).
"""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from envs.convoy_env import ConvoyEnv
from envs.sumo_connection import VehicleState


def _make_state(
    vehicle_id: str,
    x: float,
    y: float,
    speed: float = 20.0,
    acceleration: float = 0.0,
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        speed=speed,
        acceleration=acceleration,
        heading=0.0,
        lane_position=0.0,
    )


def _set_states(mock_sumo: Mock, states: dict) -> None:
    mock_sumo.get_vehicle_state = Mock(side_effect=lambda vid: states[vid])


@pytest.fixture
def mock_sumo():
    """Mock SUMOConnection for unit tests."""
    mock = Mock()
    mock.start = Mock()
    mock.stop = Mock()
    mock.step = Mock()
    mock.set_vehicle_speed = Mock()
    mock.get_simulation_time = Mock(return_value=0.0)
    mock.is_vehicle_active = Mock(return_value=True)
    mock.get_vehicle_state = Mock(
        return_value=_make_state("V001", x=0.0, y=0.0)
    )
    return mock


@pytest.fixture
def mock_emulator():
    """Mock ESPNOWEmulator for unit tests."""
    mock = Mock()
    mock.clear = Mock()
    mock.transmit = Mock()
    mock.get_observation = Mock(return_value={
        "v002_lat": 0.0,
        "v002_lon": 0.0,
        "v002_speed": 20.0,
        "v002_accel_x": 0.0,
        "v002_age_ms": 10,
        "v002_valid": True,
        "v003_lat": 0.0,
        "v003_lon": 0.0,
        "v003_speed": 20.0,
        "v003_accel_x": 0.0,
        "v003_age_ms": 15,
        "v003_valid": True,
    })
    return mock


@pytest.fixture
def env_with_mocks(mock_sumo, mock_emulator, tmp_path):
    """ConvoyEnv with mocked dependencies."""
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    with patch("envs.convoy_env.SUMOConnection", return_value=mock_sumo):
        env = ConvoyEnv(
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=False,
        )
        yield env
        env.close()


def test_reset_returns_observation_and_info(env_with_mocks):
    """reset() returns (obs, info) tuple."""
    result = env_with_mocks.reset()

    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_reset_observation_shape_is_11(env_with_mocks):
    """Initial observation has shape (11,)."""
    obs, _ = env_with_mocks.reset()

    assert obs.shape == (11,)
    assert obs.dtype == np.float32


def test_reset_clears_emulator_message_queue(env_with_mocks, mock_emulator):
    """No stale messages from previous episode."""
    env_with_mocks.reset()

    mock_emulator.clear.assert_called_once()


def test_reset_restarts_sumo_simulation(env_with_mocks, mock_sumo):
    """SUMO simulation restarts on reset."""
    env_with_mocks.reset()
    assert mock_sumo.start.call_count == 1

    env_with_mocks.reset()
    assert mock_sumo.stop.call_count == 1
    assert mock_sumo.start.call_count == 2


def test_reset_with_seed_is_reproducible(tmp_path, mock_emulator):
    """Same seed produces same hazard injection pattern."""
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    with patch("envs.convoy_env.SUMOConnection") as mock_sumo_class:
        mock_sumo = Mock()
        mock_sumo.start = Mock()
        mock_sumo.stop = Mock()
        mock_sumo.get_simulation_time = Mock(return_value=0.0)
        mock_sumo.is_vehicle_active = Mock(return_value=True)
        mock_sumo.get_vehicle_state = Mock(
            return_value=_make_state("V001", x=0.0, y=0.0)
        )
        mock_sumo_class.return_value = mock_sumo

        env = ConvoyEnv(
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=True,
        )

        env.reset(seed=42)
        pattern1 = env.hazard_injector._hazard_step

        env.reset(seed=42)
        pattern2 = env.hazard_injector._hazard_step

        assert pattern1 == pattern2

        env.close()


def test_collision_sets_terminated_true(env_with_mocks, mock_sumo):
    """Distance < 5m -> terminated=True."""
    env_with_mocks.reset()

    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=4.0, y=0.0, speed=0.0),
        "V003": _make_state("V003", x=100.0, y=0.0),
    }
    _set_states(mock_sumo, states)

    _, _, terminated, truncated, _ = env_with_mocks.step(0)

    assert terminated is True
    assert truncated is False


def test_collision_reward_is_negative_100(env_with_mocks, mock_sumo):
    """Collision step reward is -100."""
    env_with_mocks.reset()

    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=3.0, y=0.0, speed=0.0),
        "V003": _make_state("V003", x=100.0, y=0.0),
    }
    _set_states(mock_sumo, states)

    _, _, _, _, info = env_with_mocks.step(0)

    assert info["reward_safety"] == -100.0


def test_no_collision_terminated_is_false(env_with_mocks, mock_sumo):
    """Normal step (safe distance) -> terminated=False."""
    env_with_mocks.reset()

    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=30.0, y=0.0),
        "V003": _make_state("V003", x=60.0, y=0.0),
    }
    _set_states(mock_sumo, states)

    _, _, terminated, _, _ = env_with_mocks.step(0)

    assert terminated is False


def test_max_steps_sets_truncated_true(env_with_mocks, mock_sumo):
    """After max_steps -> truncated=True."""
    env_with_mocks.max_steps = 5
    env_with_mocks.reset()

    def make_state(vid):
        x = 0.0 if vid == "V001" else 30.0 if vid == "V002" else 60.0
        return _make_state(vid, x=x, y=0.0)

    mock_sumo.get_vehicle_state = Mock(side_effect=make_state)

    truncated = False
    for i in range(10):
        _, _, terminated, truncated, _ = env_with_mocks.step(0)
        if truncated or terminated:
            break

    assert truncated is True
    assert i == 4


def test_vehicle_exit_sets_truncated_true(env_with_mocks, mock_sumo):
    """If lead vehicle leaves simulation -> truncated=True."""
    env_with_mocks.reset()

    def is_active(vid):
        if vid == "V002":
            return False
        return True

    mock_sumo.is_vehicle_active = Mock(side_effect=is_active)

    _, _, _, truncated, _ = env_with_mocks.step(0)

    assert truncated is True


def test_truncated_does_not_apply_collision_reward(env_with_mocks, mock_sumo):
    """Truncation is not collision; no -100 penalty."""
    env_with_mocks.max_steps = 1
    env_with_mocks.reset()

    def make_state(vid):
        x = 0.0 if vid == "V001" else 30.0 if vid == "V002" else 60.0
        return _make_state(vid, x=x, y=0.0)

    mock_sumo.get_vehicle_state = Mock(side_effect=make_state)

    _, _, terminated, truncated, info = env_with_mocks.step(0)

    assert truncated is True
    assert terminated is False
    assert info["reward_safety"] != -100.0
