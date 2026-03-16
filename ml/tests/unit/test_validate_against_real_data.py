"""
Unit tests for sim-to-real replay helpers (Run 020 — decay-based braking signal).
"""

import numpy as np
import pytest

from ml.scripts import validate_against_real_data


def test_decay_mode_resets_to_one_on_trigger():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=0.0,
        mode="decay",
    )
    assert value == pytest.approx(1.0)


def test_decay_mode_decays_without_trigger():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=1.0,
        mode="decay",
    )
    assert value == pytest.approx(0.95)

    # Two steps of decay
    value2 = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=value,
        mode="decay",
    )
    assert value2 == pytest.approx(0.95 * 0.95)


def test_decay_mode_retrigger_resets_to_one():
    """Trigger during decay resets back to 1.0."""
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=1.0,
        mode="decay",
    )
    assert value == pytest.approx(0.95)

    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=value,
        mode="decay",
    )
    assert value == pytest.approx(1.0)


def test_latched_mode_stays_at_one():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=0.0,
        mode="latched",
    )
    assert value == pytest.approx(1.0)

    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=value,
        mode="latched",
    )
    assert value == pytest.approx(1.0)  # sticky


def test_instant_mode_no_memory():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=0.0,
        mode="instant",
    )
    assert value == pytest.approx(1.0)

    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=value,
        mode="instant",
    )
    assert value == pytest.approx(0.0)


def test_off_mode_forces_zero():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=1.0,
        mode="off",
    )
    assert value == pytest.approx(0.0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown braking_received_mode"):
        validate_against_real_data._update_braking_received_decay(
            any_braking_this_step=False,
            current_decay=0.0,
            mode="bad-mode",
        )


class _FakeModel:
    def predict(self, observation, deterministic=True):
        return np.array([0.0], dtype=np.float32), None


def test_run_replay_extracts_current_ego_indices(tmp_path):
    tx_rows = [
        validate_against_real_data.SensorRow(
            timestamp_local_ms=0,
            msg_timestamp=0,
            vehicle_id="V001",
            lat=0.0,
            lon=0.0,
            speed=10.0,
            heading=0.0,
            accel_fwd=0.0,
            hop_count=0,
        )
    ]
    rx_rows = [
        validate_against_real_data.SensorRow(
            timestamp_local_ms=0,
            msg_timestamp=0,
            vehicle_id="V002",
            lat=0.0001,  # ~11.1m north of ego, inside the forward cone
            lon=0.0,
            speed=8.0,
            heading=0.0,
            accel_fwd=-4.0,
            hop_count=0,
        )
    ]

    validate_against_real_data.run_replay(
        tx_rows=tx_rows,
        rx_rows=rx_rows,
        model=_FakeModel(),
        output_dir=tmp_path,
        recording_name="unit",
        braking_received_mode="decay",
    )

    timeseries = np.load(tmp_path / "timeseries_unit.npz")
    assert timeseries["peer_count"].tolist() == [1]
    assert timeseries["min_peer_accel"].tolist() == pytest.approx([-4.0])
    assert timeseries["braking_received"].tolist() == pytest.approx([1.0])
    assert "max_closing_speed" in timeseries


def test_build_peer_observations_excludes_self_rx():
    """Run 023 H4: V001 self-messages in RX log are excluded by default."""
    current_time_ms = 1000
    rx_rows = [
        validate_against_real_data.SensorRow(
            timestamp_local_ms=900,
            msg_timestamp=900,
            vehicle_id="V001",  # self-RX
            lat=0.0,
            lon=0.0,
            speed=10.0,
            heading=0.0,
            accel_fwd=0.0,
            hop_count=0,
        ),
        validate_against_real_data.SensorRow(
            timestamp_local_ms=950,
            msg_timestamp=950,
            vehicle_id="V002",
            lat=0.0001,
            lon=0.0,
            speed=8.0,
            heading=0.0,
            accel_fwd=-3.0,
            hop_count=0,
        ),
    ]

    # Default: excludes V001
    peers = validate_against_real_data.build_peer_observations(rx_rows, current_time_ms)
    assert len(peers) == 1
    assert peers[0]["accel"] == pytest.approx(-3.0)

    # Explicit None: includes V001
    peers_all = validate_against_real_data.build_peer_observations(
        rx_rows, current_time_ms, exclude_vehicle_id=None
    )
    assert len(peers_all) == 2
