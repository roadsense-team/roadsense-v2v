"""
Unit tests for ConvoyEnv -> ESPNOWEmulator integration (Phase 3).
"""
from unittest.mock import MagicMock, patch

from ml.envs.convoy_env import ConvoyEnv
from envs.sumo_connection import VehicleState
from espnow_emulator.espnow_emulator import ESPNOWEmulator, ReceivedMessage, V2VMessage


def _make_state(vehicle_id: str, x: float, y: float, speed: float = 10.0) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        speed=speed,
        acceleration=0.0,
        heading=0.0,
        lane_position=0.0,
    )


def test_step_espnow_calls_simulate_mesh_step_with_all_states():
    """_step_espnow delegates to mesh API with all active vehicle states."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    states = {
        "V001": _make_state("V001", 0.0, 0.0),
        "V002": _make_state("V002", 10.0, 0.0),
        "V003": _make_state("V003", 20.0, 0.0),
    }
    env.sumo = MagicMock()
    env.sumo.get_vehicle_state.side_effect = lambda vid: states[vid]
    emulator.simulate_mesh_step.return_value = {}

    with patch("traci.vehicle.getIDList", return_value=["V001", "V002", "V003"]):
        env._step_espnow(states["V001"], current_time_ms=1000)

    emulator.simulate_mesh_step.assert_called_once_with(
        vehicle_states=states,
        ego_id="V001",
        current_time_ms=1000,
    )


def test_step_espnow_returns_non_ego_peer_states_for_distance_logic():
    """Returned peer list still contains all non-ego states."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    states = {
        "V001": _make_state("V001", 0.0, 0.0),
        "V002": _make_state("V002", 10.0, 0.0),
        "V003": _make_state("V003", 20.0, 0.0),
    }
    env.sumo = MagicMock()
    env.sumo.get_vehicle_state.side_effect = lambda vid: states[vid]
    emulator.simulate_mesh_step.return_value = {}

    with patch("traci.vehicle.getIDList", return_value=["V001", "V002", "V003"]):
        _, peer_states = env._step_espnow(states["V001"], current_time_ms=1000)

    assert {state.vehicle_id for state in peer_states} == {"V002", "V003"}


def test_step_espnow_uses_mesh_messages_to_build_peer_observation():
    """Observation peer count follows mesh-visible messages."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    states = {
        "V001": _make_state("V001", 0.0, 0.0),
        "V002": _make_state("V002", 0.0, 30.0),
    }
    env.sumo = MagicMock()
    env.sumo.get_vehicle_state.side_effect = lambda vid: states[vid]
    msg = states["V002"].to_v2v_message(timestamp_ms=1000)
    emulator.simulate_mesh_step.return_value = {
        "V002": ReceivedMessage(message=msg, age_ms=12, received_at_ms=1012)
    }

    with patch("traci.vehicle.getIDList", return_value=["V001", "V002"]):
        obs, _ = env._step_espnow(states["V001"], current_time_ms=1000)

    assert int(obs["peer_mask"].sum()) == 1


def test_step_espnow_normalizes_mesh_message_age():
    """Age feature reflects mesh age_ms from relay chain."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    states = {
        "V001": _make_state("V001", 0.0, 0.0),
        "V002": _make_state("V002", 0.0, 30.0),
    }
    env.sumo = MagicMock()
    env.sumo.get_vehicle_state.side_effect = lambda vid: states[vid]
    msg = states["V002"].to_v2v_message(timestamp_ms=1000)
    emulator.simulate_mesh_step.return_value = {
        "V002": ReceivedMessage(message=msg, age_ms=14, received_at_ms=1014)
    }

    with patch("traci.vehicle.getIDList", return_value=["V001", "V002"]):
        obs, _ = env._step_espnow(states["V001"], current_time_ms=1000)

    assert obs["peers"][0][5] == 14.0 / env.obs_builder.STALENESS_THRESHOLD


def test_message_not_visible_before_arrival_time():
    """
    Message transmitted at t=100 with latency=50ms is not visible before t=150.
    """
    emulator = ESPNOWEmulator(domain_randomization=False, seed=42)
    emulator.params["latency"]["base_ms"] = 50
    emulator.params["latency"]["distance_factor"] = 0.0
    emulator.params["latency"]["jitter_std_ms"] = 0
    emulator.params["packet_loss"]["base_rate"] = 0

    msg = V2VMessage(
        vehicle_id="V002",
        lat=0.0,
        lon=0.0,
        speed=15.0,
        heading=90.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        timestamp_ms=100,
    )

    emulator.transmit(
        sender_msg=msg,
        sender_pos=(0.0, 0.0),
        receiver_pos=(0.0, 0.0),
        current_time_ms=100,
    )

    obs_early = emulator.get_observation(ego_speed=10.0, current_time_ms=100)
    assert obs_early["v002_valid"] is False

    obs_late = emulator.get_observation(ego_speed=10.0, current_time_ms=160)
    assert obs_late["v002_valid"] is True
