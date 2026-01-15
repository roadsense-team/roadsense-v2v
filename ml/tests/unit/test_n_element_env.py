"""
Unit tests for n-element ConvoyEnv observation space.
"""
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

import ml.envs
from envs.sumo_connection import VehicleState
from ml.espnow_emulator.espnow_emulator import ReceivedMessage


def _make_state(vehicle_id: str, x: float, y: float) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        speed=10.0,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )


def _make_mock_sumo(state_map: dict) -> Mock:
    mock = Mock()
    mock.start = Mock()
    mock.stop = Mock()
    mock.step = Mock()
    mock.set_vehicle_speed = Mock()
    mock.get_simulation_time = Mock(return_value=0.0)
    mock.is_vehicle_active = Mock(return_value=True)
    mock.get_vehicle_state = Mock(side_effect=lambda vid: state_map[vid])
    return mock


def _make_mock_emulator() -> Mock:
    mock = Mock()
    mock.clear = Mock()
    last_received = {}

    def _transmit(sender_msg, sender_pos, receiver_pos, current_time_ms):
        received = ReceivedMessage(
            message=sender_msg,
            age_ms=10,
            received_at_ms=current_time_ms + 10,
        )
        last_received[sender_msg.vehicle_id] = received
        return received

    def _sync_and_get_messages(_current_time_ms):
        return last_received.copy()

    mock.transmit = Mock(side_effect=_transmit)
    mock.sync_and_get_messages = Mock(side_effect=_sync_and_get_messages)
    return mock


def test_env_observation_space_is_dict(tmp_path):
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    states = {"V001": _make_state("V001", 0.0, 0.0)}
    mock_sumo = _make_mock_sumo(states)
    mock_emulator = _make_mock_emulator()

    with patch("ml.envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
         patch("traci.vehicle.getIDList", return_value=["V001"]):
        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=False,
        )

        try:
            from gymnasium.spaces import Dict as DictSpace
            assert isinstance(env.observation_space, DictSpace)
        finally:
            env.close()


def test_reset_returns_dict_keys(tmp_path):
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    states = {"V001": _make_state("V001", 0.0, 0.0)}
    mock_sumo = _make_mock_sumo(states)
    mock_emulator = _make_mock_emulator()

    with patch("ml.envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
         patch("traci.vehicle.getIDList", return_value=["V001"]):
        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=False,
        )

        try:
            obs, _ = env.reset()
            assert set(obs.keys()) == {"ego", "peers", "peer_mask"}
        finally:
            env.close()


def test_peer_mask_all_zero_with_no_peers(tmp_path):
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    states = {"V001": _make_state("V001", 0.0, 0.0)}
    mock_sumo = _make_mock_sumo(states)
    mock_emulator = _make_mock_emulator()

    with patch("ml.envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
         patch("traci.vehicle.getIDList", return_value=["V001"]):
        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=False,
        )

        try:
            obs, _ = env.reset()
            assert np.all(obs["peer_mask"] == 0.0)
        finally:
            env.close()


def test_peer_mask_three_peers(tmp_path):
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration></configuration>")

    states = {
        "V001": _make_state("V001", 0.0, 0.0),
        "V002": _make_state("V002", 10.0, 0.0),
        "V003": _make_state("V003", 20.0, 0.0),
        "V004": _make_state("V004", 30.0, 0.0),
    }
    mock_sumo = _make_mock_sumo(states)
    mock_emulator = _make_mock_emulator()

    with patch("ml.envs.convoy_env.SUMOConnection", return_value=mock_sumo), \
         patch("traci.vehicle.getIDList", return_value=["V001", "V002", "V003", "V004"]):
        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=str(cfg),
            emulator=mock_emulator,
            hazard_injection=False,
        )

        try:
            obs, _ = env.reset()
            assert np.count_nonzero(obs["peer_mask"] == 1.0) == 3
        finally:
            env.close()
