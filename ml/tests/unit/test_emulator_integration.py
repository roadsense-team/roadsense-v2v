"""
Unit tests for ConvoyEnv -> ESPNOWEmulator integration (Phase 3).
"""
from unittest.mock import MagicMock

from envs.convoy_env import ConvoyEnv
from envs.sumo_connection import VehicleState
from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage


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


def test_transmit_vehicle_states_calls_emulator_for_each_peer():
    """Transmit is called once per peer vehicle."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    ego_state = _make_state("V001", 0.0, 0.0)
    peers = [_make_state("V002", 10.0, 0.0), _make_state("V003", 20.0, 0.0)]

    env._transmit_vehicle_states(ego_state, peers, current_time_ms=1000)

    assert emulator.transmit.call_count == 2


def test_transmit_vehicle_states_uses_sender_and_receiver_positions():
    """Sender/receiver positions match vehicle states."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    ego_state = _make_state("V001", 1.0, 2.0)
    peers = [_make_state("V002", 10.0, 3.0), _make_state("V003", 20.0, 4.0)]

    env._transmit_vehicle_states(ego_state, peers, current_time_ms=1000)

    for call, peer in zip(emulator.transmit.call_args_list, peers):
        assert call.kwargs["sender_pos"] == (peer.x, peer.y)
        assert call.kwargs["receiver_pos"] == (ego_state.x, ego_state.y)


def test_transmit_vehicle_states_sets_message_timestamp():
    """Sender message timestamp matches current time."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    ego_state = _make_state("V001", 0.0, 0.0)
    peers = [_make_state("V002", 10.0, 0.0)]

    env._transmit_vehicle_states(ego_state, peers, current_time_ms=1234)

    sent_msg = emulator.transmit.call_args.kwargs["sender_msg"]
    assert sent_msg.timestamp_ms == 1234


def test_transmit_vehicle_states_empty_peer_list_no_calls():
    """No peers means no transmissions."""
    emulator = MagicMock()
    env = ConvoyEnv(sumo_cfg="dummy.sumocfg", emulator=emulator)
    ego_state = _make_state("V001", 0.0, 0.0)

    env._transmit_vehicle_states(ego_state, [], current_time_ms=1000)

    emulator.transmit.assert_not_called()


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
