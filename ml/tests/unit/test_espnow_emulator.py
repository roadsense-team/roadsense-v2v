"""
Phase C mesh-relay unit tests for ESPNOWEmulator.
"""

from dataclasses import dataclass

from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage


@dataclass(frozen=True)
class _State:
    vehicle_id: str
    x: float
    y: float
    heading: float = 0.0
    speed: float = 15.0
    acceleration: float = 0.0

    def to_v2v_message(self, timestamp_ms: int) -> V2VMessage:
        meters_per_deg = ESPNOWEmulator.METERS_PER_DEG_LAT
        return V2VMessage(
            vehicle_id=self.vehicle_id,
            lat=self.y / meters_per_deg,
            lon=self.x / meters_per_deg,
            speed=self.speed,
            heading=self.heading,
            accel_x=self.acceleration,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=timestamp_ms,
        )


def _make_emulator(max_range_m: float = 100.0, base_latency_ms: int = 10) -> ESPNOWEmulator:
    emulator = ESPNOWEmulator(domain_randomization=False, seed=123)

    emulator.params["packet_loss"]["base_rate"] = 0.0
    emulator.params["packet_loss"]["rate_tier_1"] = 0.0
    emulator.params["packet_loss"]["rate_tier_2"] = 0.0
    emulator.params["packet_loss"]["rate_tier_3"] = 0.0
    emulator.params["packet_loss"]["distance_threshold_1"] = max_range_m / 2.0
    emulator.params["packet_loss"]["distance_threshold_2"] = max_range_m

    emulator.params["latency"]["base_ms"] = base_latency_ms
    emulator.params["latency"]["distance_factor"] = 0.0
    emulator.params["latency"]["jitter_std_ms"] = 0.0

    emulator.params["sensor_noise"]["gps_std_m"] = 0.0
    emulator.params["sensor_noise"]["speed_std_ms"] = 0.0
    emulator.params["sensor_noise"]["accel_std_ms2"] = 0.0
    emulator.params["sensor_noise"]["heading_std_deg"] = 0.0
    emulator.params["sensor_noise"]["gyro_std_rad_s"] = 0.0

    return emulator


def test_mesh_direct_message_has_hop_count_zero():
    emulator = _make_emulator()
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 30.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V002" in received
    assert received["V002"].message.hop_count == 0
    assert received["V002"].message.source_id == "V002"


def test_mesh_relayed_message_has_hop_count_one():
    emulator = _make_emulator(max_range_m=100.0)
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 60.0),
        "V003": _State("V003", 0.0, 120.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V003" in received
    assert received["V003"].message.hop_count == 1
    assert received["V003"].message.source_id == "V003"


def test_mesh_ego_receives_relayed_message_from_out_of_range_vehicle():
    emulator = _make_emulator(max_range_m=100.0)
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 60.0),
        "V003": _State("V003", 0.0, 120.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V003" in received


def test_mesh_behind_vehicle_not_relayed():
    emulator = _make_emulator(max_range_m=100.0)
    states = {
        "V001": _State("V001", 0.0, 20.0),
        "V002": _State("V002", 0.0, -40.0),
        "V003": _State("V003", 0.0, -100.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V003" not in received


def test_mesh_duplicate_message_deduplicated_by_source_id():
    emulator = _make_emulator(max_range_m=100.0)
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 30.0),
        "V003": _State("V003", 0.0, 60.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V003" in received
    assert received["V003"].message.hop_count == 0


def test_mesh_relay_adds_latency_per_hop():
    emulator = _make_emulator(max_range_m=100.0, base_latency_ms=7)
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 60.0),
        "V003": _State("V003", 0.0, 120.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert received["V003"].age_ms == 14


def test_mesh_three_vehicle_convoy_ego_gets_both_peers():
    emulator = _make_emulator(max_range_m=100.0)
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 60.0),
        "V003": _State("V003", 0.0, 120.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert set(received.keys()) == {"V002", "V003"}


def test_mesh_relay_respects_esp_now_range_limit():
    emulator = _make_emulator(max_range_m=50.0)
    states = {
        "V001": _State("V001", 0.0, 0.0),
        "V002": _State("V002", 0.0, 40.0),
        "V003": _State("V003", 0.0, 100.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V002" in received
    assert "V003" not in received


def test_mesh_no_relay_when_all_peers_behind():
    emulator = _make_emulator(max_range_m=90.0)
    states = {
        "V001": _State("V001", 0.0, 140.0),
        "V002": _State("V002", 0.0, 80.0),
        "V003": _State("V003", 0.0, 20.0),
        "V004": _State("V004", 20.0, 10.0),
    }

    received = emulator.simulate_mesh_step(states, ego_id="V001", current_time_ms=1000)

    assert "V002" in received
    assert "V003" not in received
    assert "V004" not in received
