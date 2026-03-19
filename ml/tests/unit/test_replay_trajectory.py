"""Unit tests for ReplayTrajectory loading and augmentation."""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.envs.replay_trajectory import (
    AugmentConfig,
    PeerSnapshot,
    ReplayTrajectory,
    TrajectorySnapshot,
    gps_to_meters,
    METERS_PER_DEG_LAT,
    STEP_MS,
)


# ── Fixtures ──────────────────────────────────────────────────────────

TX_HEADER = [
    "timestamp_local_ms", "msg_timestamp", "vehicle_id",
    "lat", "lon", "speed", "heading",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "hop_count", "source_mac",
]

RX_HEADER = [
    "timestamp_local_ms", "msg_timestamp", "from_vehicle_id",
    "lat", "lon", "speed", "heading",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "hop_count", "source_mac",
]


def _make_tx_row(t_ms, lat=32.86, lon=35.40, speed=10.0, heading=0.0,
                 accel_y=0.0, vid="V001"):
    return [
        str(t_ms), str(t_ms), vid,
        str(lat), str(lon), str(speed), str(heading),
        "0.0", str(accel_y), "9.81",
        "0.0", "0.0", "0.0",
        "0.0", "0.0", "0.0",
        "0", "AA:BB:CC:DD:EE:FF",
    ]


def _make_rx_row(t_ms, vid="V002", lat=32.861, lon=35.401, speed=10.0,
                 heading=0.0, accel_y=0.0, hop=0):
    return [
        str(t_ms), str(t_ms), vid,
        str(lat), str(lon), str(speed), str(heading),
        "0.0", str(accel_y), "9.81",
        "0.0", "0.0", "0.0",
        "0.0", "0.0", "0.0",
        str(hop), "11:22:33:44:55:66",
    ]


@pytest.fixture
def simple_csvs(tmp_path):
    """Create minimal TX/RX CSVs with 5 timesteps at 100ms intervals."""
    tx_path = tmp_path / "V001_tx.csv"
    rx_path = tmp_path / "V001_rx.csv"

    with open(tx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        for i in range(5):
            t = 1000 + i * 100
            w.writerow(_make_tx_row(t, speed=10.0, accel_y=-0.5 * i))

    with open(rx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_HEADER)
        for i in range(5):
            t = 1000 + i * 100
            w.writerow(_make_rx_row(t, vid="V002", speed=9.0, accel_y=-1.0))
            w.writerow(_make_rx_row(t, vid="V003", speed=11.0, accel_y=0.5))

    return str(tx_path), str(rx_path)


@pytest.fixture
def self_rx_csvs(tmp_path):
    """CSVs where RX contains V001 self-messages."""
    tx_path = tmp_path / "V001_tx.csv"
    rx_path = tmp_path / "V001_rx.csv"

    with open(tx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        w.writerow(_make_tx_row(1000))

    with open(rx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_HEADER)
        # Self message (V001)
        w.writerow(_make_rx_row(1000, vid="V001"))
        # Real peer
        w.writerow(_make_rx_row(1000, vid="V002"))

    return str(tx_path), str(rx_path)


# ── Loading tests ─────────────────────────────────────────────────────

class TestLoading:
    def test_load_parses_tx_csv_into_snapshots(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        assert len(snaps) == 5

    def test_load_parses_rx_csv_into_peer_snapshots(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        # Each timestep should have 2 peers (V002, V003)
        assert len(snaps[0].peers) == 2

    def test_load_uses_forward_axis_y_by_default(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx, forward_axis="y")
        snaps = traj.load()
        # First snapshot ego_accel should be 0.0 (accel_y for i=0)
        assert snaps[0].ego_accel == 0.0
        # Second snapshot ego_accel should be -0.5 (accel_y for i=1)
        assert abs(snaps[1].ego_accel - (-0.5)) < 1e-6

    def test_load_converts_gps_to_meters(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        expected_x, expected_y = gps_to_meters(32.86, 35.40)
        assert abs(snaps[0].ego_x - expected_x) < 1.0
        assert abs(snaps[0].ego_y - expected_y) < 1.0

    def test_load_excludes_self_rx_messages(self, self_rx_csvs):
        tx, rx = self_rx_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        # Only V002 should survive, not V001
        assert len(snaps) >= 1
        for snap in snaps:
            for p in snap.peers:
                assert p.vehicle_id != "V001"

    def test_load_snaps_to_10hz_grid(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        for i in range(1, len(snaps)):
            dt = snaps[i].time_ms - snaps[i - 1].time_ms
            assert dt == STEP_MS

    def test_snapshot_timestamps_are_monotonic(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        for i in range(1, len(snaps)):
            assert snaps[i].time_ms > snaps[i - 1].time_ms

    def test_snapshot_peer_accel_matches_csv(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        snaps = traj.load()
        v002_peers = [p for p in snaps[0].peers if p.vehicle_id == "V002"]
        assert len(v002_peers) == 1
        assert abs(v002_peers[0].accel - (-1.0)) < 1e-6

    def test_load_handles_missing_rx_rows(self, tmp_path):
        """Timestamps with no RX data → empty peers."""
        tx_path = tmp_path / "V001_tx.csv"
        rx_path = tmp_path / "V001_rx.csv"

        with open(tx_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(TX_HEADER)
            w.writerow(_make_tx_row(1000))

        with open(rx_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(RX_HEADER)
            # No data rows

        traj = ReplayTrajectory(str(tx_path), str(rx_path))
        snaps = traj.load()
        assert len(snaps) >= 1
        assert len(snaps[0].peers) == 0

    def test_load_caches_result(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        s1 = traj.load()
        s2 = traj.load()
        assert s1 is s2


# ── GPS conversion ────────────────────────────────────────────────────

class TestGPSConversion:
    def test_gps_to_meters_known_values(self):
        x, y = gps_to_meters(32.86, 35.40)
        assert abs(x - 35.40 * METERS_PER_DEG_LAT) < 0.01
        assert abs(y - 32.86 * METERS_PER_DEG_LAT) < 0.01


# ── Augmentation tests ────────────────────────────────────────────────

class TestAugmentation:
    def test_augment_default_config_is_identity(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        base = traj.load()
        aug = traj.augment(rng, AugmentConfig())
        assert len(aug) == len(base)
        for b, a in zip(base, aug):
            assert abs(b.ego_speed - a.ego_speed) < 1e-6
            assert abs(b.ego_accel - a.ego_accel) < 1e-6
            assert len(a.peers) == len(b.peers)

    def test_augment_accel_scale_multiplies_peer_accel(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        config = AugmentConfig(accel_scale=(2.0, 2.0))
        base = traj.load()
        aug = traj.augment(rng, config)
        for b, a in zip(base, aug):
            for bp, ap in zip(b.peers, a.peers):
                assert abs(ap.accel - bp.accel * 2.0) < 1e-6

    def test_augment_speed_scale_affects_all_vehicles(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        config = AugmentConfig(speed_scale=(0.5, 0.5))
        base = traj.load()
        aug = traj.augment(rng, config)
        for b, a in zip(base, aug):
            assert abs(a.ego_speed - b.ego_speed * 0.5) < 1e-6
            for bp, ap in zip(b.peers, a.peers):
                assert abs(ap.speed - bp.speed * 0.5) < 1e-6

    def test_augment_gps_noise_adds_gaussian_to_positions(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        config = AugmentConfig(gps_noise_m=(5.0, 5.0))
        base = traj.load()
        aug = traj.augment(rng, config)
        # At least one peer position should differ
        any_different = False
        for b, a in zip(base, aug):
            for bp, ap in zip(b.peers, a.peers):
                if abs(ap.x - bp.x) > 0.01 or abs(ap.y - bp.y) > 0.01:
                    any_different = True
        assert any_different

    def test_augment_packet_drop_removes_peers_randomly(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        config = AugmentConfig(drop_rate=(0.99, 0.99))
        base = traj.load()
        aug = traj.augment(rng, config)
        base_peers = sum(len(s.peers) for s in base)
        aug_peers = sum(len(s.peers) for s in aug)
        assert aug_peers < base_peers

    def test_augment_ego_offset_shifts_initial_position(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        config = AugmentConfig(ego_offset_m=(10.0, 10.0))
        base = traj.load()
        aug = traj.augment(rng, config)
        # Ego position should be offset
        dx = aug[0].ego_x - base[0].ego_x
        dy = aug[0].ego_y - base[0].ego_y
        offset_mag = (dx**2 + dy**2) ** 0.5
        assert abs(offset_mag - 10.0) < 0.1

    def test_augment_seed_is_reproducible(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        config = AugmentConfig(
            accel_scale=(0.5, 1.5),
            gps_noise_m=(1.0, 3.0),
        )
        aug1 = traj.augment(np.random.default_rng(99), config)
        aug2 = traj.augment(np.random.default_rng(99), config)
        for a, b in zip(aug1, aug2):
            assert abs(a.ego_accel - b.ego_accel) < 1e-6
            for pa, pb in zip(a.peers, b.peers):
                assert abs(pa.x - pb.x) < 1e-6

    def test_augment_synthetic_hazard_creates_braking_event(self, tmp_path):
        """Use a longer fixture (50 steps) so the hazard ramp actually peaks."""
        tx_path = tmp_path / "V001_tx.csv"
        rx_path = tmp_path / "V001_rx.csv"
        with open(tx_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(TX_HEADER)
            for i in range(50):
                w.writerow(_make_tx_row(1000 + i * 100, speed=10.0))
        with open(rx_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(RX_HEADER)
            for i in range(50):
                w.writerow(_make_rx_row(1000 + i * 100, vid="V002", speed=10.0))

        traj = ReplayTrajectory(str(tx_path), str(rx_path))
        rng = np.random.default_rng(42)
        config = AugmentConfig(inject_brake=True)
        aug = traj.augment(rng, config)
        min_accel = min(
            p.accel for s in aug for p in s.peers
        ) if any(s.peers for s in aug) else 0.0
        assert min_accel < -3.0

    def test_augment_synthetic_hazard_decelerates_peer_speed(self, simple_csvs):
        tx, rx = simple_csvs
        traj = ReplayTrajectory(tx, rx)
        rng = np.random.default_rng(42)
        config = AugmentConfig(inject_brake=True)
        base = traj.load()
        aug = traj.augment(rng, config)
        # Find target peer: should have lower speed than base in later snaps
        # (we can't know which peer exactly, but at least one should be slower)
        found_slower = False
        for b, a in zip(base, aug):
            for bp, ap in zip(b.peers, a.peers):
                if bp.vehicle_id == ap.vehicle_id and ap.speed < bp.speed - 0.1:
                    found_slower = True
        assert found_slower
