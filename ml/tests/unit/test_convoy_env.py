"""
Unit tests for ConvoyEnv (Phase 6).
"""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ml.envs.convoy_env import ConvoyEnv
from envs.observation_builder import ObservationBuilder
from envs.sumo_connection import VehicleState
from espnow_emulator.espnow_emulator import ReceivedMessage, V2VMessage


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


def _make_received_message(
    source_id: str,
    x: float,
    y: float,
    age_ms: int,
    hop_count: int,
    timestamp_ms: int = 1000,
) -> ReceivedMessage:
    meters_per_deg = 111000.0
    msg = V2VMessage(
        vehicle_id=source_id,
        lat=y / meters_per_deg,
        lon=x / meters_per_deg,
        speed=15.0,
        heading=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        timestamp_ms=timestamp_ms,
        hop_count=hop_count,
        source_id=source_id,
    )
    return ReceivedMessage(
        message=msg,
        age_ms=age_ms,
        received_at_ms=timestamp_ms + age_ms,
    )


def _make_dataset_dir(dataset_dir: Path) -> None:
    train_ids = ["train_000", "train_001"]
    eval_ids = ["eval_000", "eval_001"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({
            "dataset_id": dataset_dir.name,
            "train_scenarios": train_ids,
            "eval_scenarios": eval_ids,
        }),
        encoding="utf-8",
    )
    for scenario_id in train_ids:
        scenario_dir = dataset_dir / "train" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "scenario.sumocfg").write_text(
            "<configuration></configuration>",
            encoding="utf-8",
        )
    for scenario_id in eval_ids:
        scenario_dir = dataset_dir / "eval" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "scenario.sumocfg").write_text(
            "<configuration></configuration>",
            encoding="utf-8",
        )


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
    mock.set_config = Mock()
    return mock


@pytest.fixture
def mock_emulator():
    """Mock ESPNOWEmulator for unit tests."""
    mock = Mock()
    mock.clear = Mock()
    mock.simulate_mesh_step = Mock(return_value={})
    return mock


@pytest.fixture
def env_with_mocks(mock_sumo, mock_emulator, tmp_path):
    """ConvoyEnv with mocked dependencies."""
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    with patch("envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
         patch("traci.vehicle.getIDList", return_value=["V001", "V002", "V003"]):
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
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_step_accepts_numpy_action(env_with_mocks):
    """step() accepts numpy array actions from vectorized envs."""
    obs, info = env_with_mocks.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    result = env_with_mocks.step(np.array([0.3], dtype=np.float32))

    assert isinstance(result, tuple)
    assert len(result) == 5


def test_action_space_is_box_1(env_with_mocks):
    """action_space is Box(shape=(1,)) with range [0,1]."""
    from gymnasium.spaces import Box

    assert isinstance(env_with_mocks.action_space, Box)
    assert env_with_mocks.action_space.shape == (1,)
    assert env_with_mocks.action_space.low[0] == pytest.approx(0.0)
    assert env_with_mocks.action_space.high[0] == pytest.approx(1.0)


def test_convoy_env_uses_simulate_mesh_step(env_with_mocks, mock_sumo, mock_emulator):
    """_step_espnow delegates to emulator simulate_mesh_step."""
    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=0.0, y=20.0),
        "V003": _make_state("V003", x=0.0, y=40.0),
    }
    _set_states(mock_sumo, states)

    env_with_mocks._step_espnow(states["V001"], current_time_ms=1000)

    mock_emulator.simulate_mesh_step.assert_called_once_with(
        vehicle_states=states,
        ego_id="V001",
        current_time_ms=1000,
    )


def test_convoy_env_mesh_step_returns_correct_peer_count(
    env_with_mocks,
    mock_sumo,
    mock_emulator,
):
    """Observation mask reflects direct + relayed peer count from mesh."""
    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=0.0, y=20.0),
        "V003": _make_state("V003", x=0.0, y=40.0),
    }
    _set_states(mock_sumo, states)
    mock_emulator.simulate_mesh_step.return_value = {
        "V002": _make_received_message("V002", x=0.0, y=20.0, age_ms=10, hop_count=0),
        "V003": _make_received_message("V003", x=0.0, y=40.0, age_ms=20, hop_count=1),
    }

    obs, _ = env_with_mocks._step_espnow(states["V001"], current_time_ms=1000)

    assert int(np.count_nonzero(obs["peer_mask"])) == 2


def test_convoy_env_observation_includes_relayed_peers(
    env_with_mocks,
    mock_sumo,
    mock_emulator,
):
    """Relayed source appears in peer observation features."""
    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=0.0, y=20.0),
        "V003": _make_state("V003", x=0.0, y=100.0),
    }
    _set_states(mock_sumo, states)
    mock_emulator.simulate_mesh_step.return_value = {
        "V003": _make_received_message("V003", x=0.0, y=100.0, age_ms=14, hop_count=1),
    }

    obs, _ = env_with_mocks._step_espnow(states["V001"], current_time_ms=1000)

    assert obs["peer_mask"][0] == pytest.approx(1.0)
    assert obs["peers"][0][0] == pytest.approx(1.0, abs=1e-6)


def test_convoy_env_hop_count_reflected_in_message_age(
    env_with_mocks,
    mock_sumo,
    mock_emulator,
):
    """Relayed hop latency is preserved in age feature normalization."""
    states = {
        "V001": _make_state("V001", x=0.0, y=0.0),
        "V002": _make_state("V002", x=0.0, y=20.0),
        "V003": _make_state("V003", x=0.0, y=40.0),
    }
    _set_states(mock_sumo, states)
    mock_emulator.simulate_mesh_step.return_value = {
        "V002": _make_received_message("V002", x=0.0, y=20.0, age_ms=14, hop_count=1),
    }

    obs, _ = env_with_mocks._step_espnow(states["V001"], current_time_ms=1000)

    expected_age_norm = 14.0 / ObservationBuilder.STALENESS_THRESHOLD
    assert obs["peers"][0][5] == pytest.approx(expected_age_norm)


def test_step_accepts_float_action(env_with_mocks):
    """step() accepts scalar float action."""
    env_with_mocks.reset()
    result = env_with_mocks.step(0.5)
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_reset_observation_shape_is_dict(env_with_mocks):
    """Initial observation has Dict entries with expected shapes."""
    obs, _ = env_with_mocks.reset()

    assert set(obs.keys()) == {"ego", "peers", "peer_mask"}
    assert obs["ego"].shape == (4,)
    assert obs["peers"].shape == (ObservationBuilder.MAX_PEERS, 6)
    assert obs["peer_mask"].shape == (ObservationBuilder.MAX_PEERS,)
    assert obs["ego"].dtype == np.float32
    assert obs["peers"].dtype == np.float32
    assert obs["peer_mask"].dtype == np.float32


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


def test_reset_waits_for_ego_spawn(env_with_mocks, mock_sumo):
    """reset() steps until V001 is active."""
    mock_sumo.is_vehicle_active = Mock(
        side_effect=[False, False, True],
    )

    env_with_mocks.reset()

    assert mock_sumo.step.call_count == 2


def test_reset_times_out_when_ego_missing(tmp_path, mock_emulator):
    """reset() raises if V001 never appears."""
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    with patch("envs.convoy_env.SUMOConnection") as mock_sumo_class, \
         patch("traci.vehicle.getIDList", return_value=["V001"]):
        mock_sumo = Mock()
        mock_sumo.start = Mock()
        mock_sumo.stop = Mock()
        mock_sumo.step = Mock()
        mock_sumo.get_simulation_time = Mock(return_value=0.0)
        mock_sumo.is_vehicle_active = Mock(return_value=False)
        mock_sumo_class.return_value = mock_sumo

        env = ConvoyEnv(
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=False,
        )
        env.MAX_STARTUP_STEPS = 3

        with pytest.raises(RuntimeError):
            env.reset()

        mock_sumo.stop.assert_called_once()
        env.close()


def test_reset_with_seed_is_reproducible(tmp_path, mock_emulator):
    """Same seed produces same hazard injection pattern."""
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    with patch("envs.convoy_env.SUMOConnection") as mock_sumo_class, \
         patch("traci.vehicle.getIDList", return_value=["V001"]):
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


def test_ego_exit_sets_truncated_true(env_with_mocks, mock_sumo):
    """If ego leaves simulation -> truncated=True."""
    env_with_mocks.reset()

    mock_sumo.is_vehicle_active = Mock(return_value=False)

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


class TestConvoyEnvDataset:
    """Tests for dataset-based ConvoyEnv."""

    def test_init_with_dataset_dir(self, tmp_path, mock_emulator, mock_sumo):
        dataset_dir = tmp_path / "dataset"
        _make_dataset_dir(dataset_dir)

        with patch("envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
             patch("traci.vehicle.getIDList", return_value=["V001"]):
            env = ConvoyEnv(
                dataset_dir=str(dataset_dir),
                scenario_mode="train",
                scenario_seed=123,
                emulator=mock_emulator,
                hazard_injection=False,
            )

        assert env.scenario_manager is not None
        env.close()

    def test_init_mutual_exclusion(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _make_dataset_dir(dataset_dir)

        with pytest.raises(ValueError):
            ConvoyEnv(
                sumo_cfg="ml/scenarios/base/scenario.sumocfg",
                dataset_dir=str(dataset_dir),
            )

        with pytest.raises(ValueError):
            ConvoyEnv()

    def test_reset_switches_scenario(self, tmp_path, mock_emulator, mock_sumo):
        dataset_dir = tmp_path / "dataset"
        _make_dataset_dir(dataset_dir)

        with patch("envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
             patch("traci.vehicle.getIDList", return_value=["V001"]):
            env = ConvoyEnv(
                dataset_dir=str(dataset_dir),
                scenario_mode="eval",
                emulator=mock_emulator,
                hazard_injection=False,
            )

            env.reset()
            env.reset()

        first_call = mock_sumo.set_config.call_args_list[0].args[0]
        second_call = mock_sumo.set_config.call_args_list[1].args[0]

        assert first_call.endswith("eval_000/scenario.sumocfg")
        assert second_call.endswith("eval_001/scenario.sumocfg")
        env.close()

    def test_reset_returns_scenario_id(self, tmp_path, mock_emulator, mock_sumo):
        dataset_dir = tmp_path / "dataset"
        _make_dataset_dir(dataset_dir)

        with patch("envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
             patch("traci.vehicle.getIDList", return_value=["V001"]):
            env = ConvoyEnv(
                dataset_dir=str(dataset_dir),
                scenario_mode="eval",
                emulator=mock_emulator,
                hazard_injection=False,
            )

            _, info = env.reset()

        assert info["scenario_id"] == "eval_000"
        env.close()

    def test_backward_compatible_sumo_cfg(self, tmp_path, mock_emulator, mock_sumo):
        cfg = tmp_path / "single.sumocfg"
        cfg.write_text("<configuration></configuration>")

        with patch("envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
             patch("traci.vehicle.getIDList", return_value=["V001"]):
            env = ConvoyEnv(
                sumo_cfg=str(cfg),
                emulator=mock_emulator,
                hazard_injection=False,
            )

            _, info = env.reset()

        assert info["scenario_id"] is None
        env.close()
