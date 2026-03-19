"""Unit tests for ReplayConvoyEnv."""

import csv
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from ml.envs.replay_convoy_env import ReplayConvoyEnv


# ── Test CSV Fixtures ─────────────────────────────────────────────────

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


def _make_rx_row(t_ms, vid="V002", lat=32.8601, lon=35.40, speed=10.0,
                 heading=0.0, accel_y=0.0, hop=0):
    """Default peer is ~11m north of ego (ahead in heading=0)."""
    return [
        str(t_ms), str(t_ms), vid,
        str(lat), str(lon), str(speed), str(heading),
        "0.0", str(accel_y), "9.81",
        "0.0", "0.0", "0.0",
        "0.0", "0.0", "0.0",
        str(hop), "11:22:33:44:55:66",
    ]


@pytest.fixture
def recordings_dir(tmp_path):
    """Create a recordings directory with one 100-step recording.

    Both ego and peer move north at 10 m/s, maintaining ~25m gap.
    Peer lat increases each step so it isn't stationary.
    """
    rec_dir = tmp_path / "recordings" / "recording_01"
    rec_dir.mkdir(parents=True)

    tx_path = rec_dir / "V001_tx.csv"
    rx_path = rec_dir / "V001_rx.csv"

    n_steps = 100
    base_lat = 32.86
    # 25m ahead in lat degrees (north)
    peer_offset_lat = 25.0 / 111_000.0
    speed_lat_per_step = 10.0 * 0.1 / 111_000.0  # 10 m/s * 0.1s / m_per_deg

    with open(tx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            lat = base_lat + i * speed_lat_per_step
            w.writerow(_make_tx_row(t, lat=lat, speed=10.0, heading=0.0))

    with open(rx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            lat = base_lat + peer_offset_lat + i * speed_lat_per_step
            w.writerow(_make_rx_row(
                t, vid="V002", lat=lat, speed=10.0, heading=0.0
            ))

    return str(tmp_path / "recordings")


@pytest.fixture
def braking_recordings_dir(tmp_path):
    """Recording where V002 brakes hard mid-episode.

    Both vehicles move north. Peer starts 25m ahead, brakes from step 40.
    """
    rec_dir = tmp_path / "recordings" / "recording_01"
    rec_dir.mkdir(parents=True)

    tx_path = rec_dir / "V001_tx.csv"
    rx_path = rec_dir / "V001_rx.csv"

    n_steps = 100
    base_lat = 32.86
    peer_offset_lat = 25.0 / 111_000.0
    speed_lat_per_step = 10.0 * 0.1 / 111_000.0

    with open(tx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            lat = base_lat + i * speed_lat_per_step
            w.writerow(_make_tx_row(t, lat=lat, speed=10.0, heading=0.0))

    # Peer brakes from step 40: speed drops, accel = -5.0
    peer_pos = base_lat + peer_offset_lat
    peer_speed = 10.0
    with open(rx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            accel = -5.0 if i >= 40 else 0.0
            if i >= 40:
                peer_speed = max(0.0, peer_speed - 5.0 * 0.1)
            peer_pos += peer_speed * 0.1 / 111_000.0
            w.writerow(_make_rx_row(
                t, vid="V002", lat=peer_pos, speed=peer_speed,
                heading=0.0, accel_y=accel,
            ))

    return str(tmp_path / "recordings")


@pytest.fixture
def env(recordings_dir):
    """Non-augmented env for deterministic tests."""
    return ReplayConvoyEnv(
        recordings_dir=recordings_dir,
        augment=False,
        max_steps=500,
        seed=42,
    )


@pytest.fixture
def braking_env(braking_recordings_dir):
    """Non-augmented env with braking peer."""
    return ReplayConvoyEnv(
        recordings_dir=braking_recordings_dir,
        augment=False,
        max_steps=500,
        seed=42,
    )


# ── Space checks ──────────────────────────────────────────────────────

class TestObservationSpace:
    def test_observation_space_ego_shape(self, env):
        assert env.observation_space["ego"].shape == (6,)

    def test_observation_space_peers_shape(self, env):
        assert env.observation_space["peers"].shape == (8, 6)

    def test_observation_space_mask_shape(self, env):
        assert env.observation_space["peer_mask"].shape == (8,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0


# ── Reset tests ───────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_obs_info_tuple(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_obs_ego_shape_is_6(self, env):
        obs, _ = env.reset()
        assert obs["ego"].shape == (6,)

    def test_reset_obs_peers_shape(self, env):
        obs, _ = env.reset()
        assert obs["peers"].shape == (8, 6)
        assert obs["peer_mask"].shape == (8,)

    def test_reset_initializes_ego_at_recording_start(self, env):
        obs, _ = env.reset()
        # ego[0] = speed/30 = 10/30 ≈ 0.333
        assert abs(obs["ego"][0] - 10.0 / 30.0) < 0.01

    def test_reset_clears_braking_received_decay(self, env):
        obs, _ = env.reset()
        # ego[4] = braking_received = 0.0 after reset
        assert obs["ego"][4] == 0.0

    def test_reset_info_contains_metadata(self, env):
        _, info = env.reset()
        assert "recording_index" in info
        assert "num_snapshots" in info


# ── Step tests ────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(np.array([0.0]))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_timestep(self, env):
        env.reset()
        _, _, _, _, info1 = env.step(np.array([0.0]))
        _, _, _, _, info2 = env.step(np.array([0.0]))
        assert info2["step_idx"] > info1["step_idx"]

    def test_step_zero_action_ego_coasts(self, env):
        env.reset()
        obs, _, _, _, info = env.step(np.array([0.0]))
        # No deceleration
        assert abs(info["deceleration"]) < 0.01

    def test_step_full_brake_ego_decelerates(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0]))
        assert abs(info["deceleration"]) > 7.0  # close to MAX_DECEL

    def test_step_obs_reflects_real_peer_positions(self, env):
        obs, _ = env.reset()
        # At least one peer should be visible (V002 is ahead)
        assert obs["peer_mask"].sum() >= 1

    def test_step_ego_position_changes_with_action(self, env):
        env.reset()
        # Coast for 10 steps
        for _ in range(10):
            env.step(np.array([0.0]))
        speed1 = env._ego.speed
        # Brake for 10 steps
        for _ in range(10):
            env.step(np.array([1.0]))
        speed2 = env._ego.speed
        assert speed2 < speed1

    def test_step_braking_peer_triggers_decay_signal(self, braking_env):
        braking_env.reset()
        # Step past the braking onset (step 40+)
        for i in range(50):
            obs, _, _, _, info = braking_env.step(np.array([0.0]))
        # braking_received should be > 0 after peer brakes
        assert info["braking_received_decay"] > 0.0

    def test_step_reward_uses_reward_calculator(self, env):
        env.reset()
        _, reward, _, _, info = env.step(np.array([0.0]))
        # Reward info should contain standard fields
        assert "reward_safety" in info
        assert "reward_comfort" in info

    def test_step_distance_is_ego_to_nearest_peer(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([0.0]))
        # V002 is ~11m north of ego → distance should be ~11m
        assert 5.0 < info["distance"] < 50.0


# ── Termination tests ─────────────────────────────────────────────────

class TestTermination:
    def test_end_of_recording_truncates(self, recordings_dir):
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=9999,  # Higher than recording length
            seed=42,
        )
        env.reset()
        truncated = False
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(np.array([0.0]))
            if terminated or truncated:
                break
        assert truncated

    def test_max_steps_truncates(self, recordings_dir):
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=10,
            seed=42,
        )
        env.reset()
        truncated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(np.array([0.0]))
            if terminated or truncated:
                break
        assert truncated


# ── Info dict tests ───────────────────────────────────────────────────

class TestInfo:
    def test_info_contains_distance_and_reward_breakdown(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([0.0]))
        assert "distance" in info
        assert "reward_safety" in info
        assert "reward_comfort" in info
        assert "reward_appropriateness" in info

    def test_info_contains_braking_received_decay(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([0.0]))
        assert "braking_received_decay" in info

    def test_info_contains_deceleration(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([0.5]))
        assert "deceleration" in info

    def test_info_contains_replay_reward_gating_diagnostics(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([0.0]))
        assert "reward_config" in info
        assert "ignoring_signal_active" in info
        assert "ignoring_geometry_active" in info

    def test_info_contains_recorded_and_reward_geometry(self, recorded_ego_shadow_env):
        recorded_ego_shadow_env.reset()
        _, _, _, _, info = recorded_ego_shadow_env.step(np.array([1.0]))
        assert "recorded_distance" in info
        assert "reward_distance" in info
        assert "recorded_closing_rate" in info
        assert "reward_closing_rate" in info
        assert "reward_geometry_source" in info


class TestReplayRewardConfig:
    def test_default_replay_reward_config_is_tightened(self, env):
        assert env.reward_config["ignoring_hazard_threshold"] == pytest.approx(0.15)
        assert env.reward_config["ignoring_require_danger_geometry"] is False
        assert env.reward_config["ignoring_use_any_braking_peer"] is True

    def test_custom_reward_config_overrides_default(self, recordings_dir):
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=500,
            seed=42,
            reward_config={
                "ignoring_hazard_threshold": 0.5,
                "ignoring_danger_distance": 25.0,
            },
        )
        assert env.reward_config["ignoring_hazard_threshold"] == pytest.approx(0.5)
        assert env.reward_config["ignoring_danger_distance"] == pytest.approx(25.0)
        assert env.reward_config["ignoring_require_danger_geometry"] is False


# ── Augmentation integration tests ────────────────────────────────────

class TestAugmentation:
    def test_augment_enabled_produces_varied_episodes(self, recordings_dir):
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=True,
            max_steps=50,
            seed=42,
        )
        obs1, _ = env.reset(seed=1)
        speeds1 = [obs1["ego"][0]]
        for _ in range(10):
            o, _, _, _, _ = env.step(np.array([0.0]))
            speeds1.append(o["ego"][0])

        obs2, _ = env.reset(seed=2)
        speeds2 = [obs2["ego"][0]]
        for _ in range(10):
            o, _, _, _, _ = env.step(np.array([0.0]))
            speeds2.append(o["ego"][0])

        # With different seeds, augmentation should produce different trajectories
        # (at least one speed should differ due to speed_scale augmentation)
        any_diff = any(abs(a - b) > 0.001 for a, b in zip(speeds1, speeds2))
        assert any_diff

    def test_augment_disabled_produces_same_episode(self, recordings_dir):
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=50,
            seed=42,
        )
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        # Without augmentation, same recording → same initial obs
        np.testing.assert_allclose(obs1["ego"], obs2["ego"], atol=1e-6)

    def test_synthetic_hazard_produces_braking_signal(self, braking_recordings_dir):
        """braking_recordings_dir already has a braking peer from step 40."""
        env = ReplayConvoyEnv(
            recordings_dir=braking_recordings_dir,
            augment=False,
            max_steps=500,
            seed=42,
        )
        env.reset()
        found_braking = False
        for _ in range(99):
            _, _, terminated, truncated, info = env.step(np.array([0.0]))
            if info["braking_received_decay"] > 0.5:
                found_braking = True
                break
            if terminated or truncated:
                break
        assert found_braking, "Expected braking signal from hard-braking peer"


# ── Recorded ego mode tests ──────────────────────────────────────────

@pytest.fixture
def recorded_ego_env(recordings_dir):
    """Env using recorded ego state (matching validator)."""
    return ReplayConvoyEnv(
        recordings_dir=recordings_dir,
        augment=False,
        max_steps=500,
        seed=42,
        use_recorded_ego=True,
    )


@pytest.fixture
def recorded_ego_shadow_env(braking_recordings_dir):
    """Recorded-ego observations with shadow-ego reward geometry enabled."""
    return ReplayConvoyEnv(
        recordings_dir=braking_recordings_dir,
        augment=False,
        max_steps=500,
        seed=42,
        use_recorded_ego=True,
        use_shadow_reward_geometry=True,
        reward_config={
            "ignoring_hazard_threshold": 0.3,
            "ignoring_require_danger_geometry": True,
        },
    )


class TestRecordedEgoMode:
    def test_recorded_ego_obs_matches_recording_speed(self, recorded_ego_env):
        """In recorded-ego mode, ego speed should stay at recorded value
        regardless of RL action."""
        recorded_ego_env.reset()
        # Brake hard for 10 steps
        for _ in range(10):
            obs, _, _, _, info = recorded_ego_env.step(np.array([1.0]))
        # ego[0] = speed/30 should still be ~10/30 (recorded speed)
        assert abs(obs["ego"][0] - 10.0 / 30.0) < 0.05

    def test_recorded_ego_speed_invariant_to_action(self, recorded_ego_env):
        """Stepping with action=0 vs action=1 should produce same ego speed."""
        recorded_ego_env.reset()
        obs0, _, _, _, _ = recorded_ego_env.step(np.array([0.0]))
        recorded_ego_env.reset()
        obs1, _, _, _, _ = recorded_ego_env.step(np.array([1.0]))
        assert abs(obs0["ego"][0] - obs1["ego"][0]) < 1e-6

    def test_recorded_ego_decel_from_action(self, recorded_ego_env):
        """In recorded-ego mode, deceleration should come from action × MAX_DECEL."""
        recorded_ego_env.reset()
        _, _, _, _, info = recorded_ego_env.step(np.array([0.5]))
        expected = 0.5 * 8.0  # action × MAX_DECEL
        assert abs(info["deceleration"] - expected) < 0.01

    def test_recorded_ego_no_collision_termination(self, recordings_dir):
        """In recorded-ego mode, episode should never terminate on collision
        (real driver didn't collide)."""
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=9999,
            seed=42,
            use_recorded_ego=True,
        )
        env.reset()
        terminated = False
        for _ in range(99):
            _, _, terminated, truncated, _ = env.step(np.array([1.0]))
            if terminated or truncated:
                break
        # Should truncate at end of recording, never terminate
        assert not terminated

    def test_recorded_ego_reward_depends_on_action(self, recorded_ego_env):
        """Even in recorded-ego mode, different actions should give different rewards."""
        recorded_ego_env.reset()
        _, r0, _, _, _ = recorded_ego_env.step(np.array([0.0]))
        recorded_ego_env.reset()
        _, r1, _, _, _ = recorded_ego_env.step(np.array([1.0]))
        # Full braking incurs comfort penalty → different reward
        assert r0 != r1

    def test_recorded_ego_accel_from_recording(self, recorded_ego_env):
        """In recorded-ego mode, ego accel should come from TX CSV, not kinematics."""
        recorded_ego_env.reset()
        obs, _, _, _, _ = recorded_ego_env.step(np.array([1.0]))
        # Recording has accel_y=0.0, so ego[1] (accel/10) should be ~0
        assert abs(obs["ego"][1]) < 0.05

    def test_shadow_reward_geometry_keeps_recorded_observation_speed(
        self, recorded_ego_shadow_env
    ):
        """Shadow reward geometry must not alter the recorded ego observation."""
        recorded_ego_shadow_env.reset()
        obs, _, _, _, info = recorded_ego_shadow_env.step(np.array([1.0]))
        assert abs(obs["ego"][0] - 10.0 / 30.0) < 0.05
        assert info["reward_geometry_source"] == "shadow"
        assert info["observation_ego_speed"] == pytest.approx(10.0)
        assert info["ego_speed"] < info["observation_ego_speed"]

    def test_shadow_reward_distance_diverges_from_recorded_distance(
        self, recorded_ego_shadow_env
    ):
        """Full braking should widen the shadow gap while recorded gap stays fixed."""
        recorded_ego_shadow_env.reset()
        info = None
        for _ in range(10):
            _, _, _, _, info = recorded_ego_shadow_env.step(np.array([1.0]))
        assert info is not None
        assert info["reward_distance"] > info["recorded_distance"] + 3.0

    def test_shadow_reward_geometry_changes_ignoring_penalty(
        self, recorded_ego_shadow_env, braking_recordings_dir
    ):
        """Ignoring penalty should depend on the shadow ego trajectory, not the
        human-safe recorded geometry."""
        recorded_ego_shadow_env.reset()
        max_no_brake_penalty = 0.0
        for _ in range(70):
            _, _, terminated, truncated, info = recorded_ego_shadow_env.step(
                np.array([0.0])
            )
            max_no_brake_penalty = min(
                max_no_brake_penalty, info["reward_ignoring_hazard"]
            )
            if terminated or truncated:
                break

        full_brake_env = ReplayConvoyEnv(
            recordings_dir=braking_recordings_dir,
            augment=False,
            max_steps=500,
            seed=42,
            use_recorded_ego=True,
            use_shadow_reward_geometry=True,
            reward_config={
                "ignoring_hazard_threshold": 0.3,
                "ignoring_require_danger_geometry": True,
            },
        )
        full_brake_env.reset()
        max_full_brake_penalty = 0.0
        for _ in range(70):
            _, _, terminated, truncated, info = full_brake_env.step(np.array([1.0]))
            max_full_brake_penalty = min(
                max_full_brake_penalty, info["reward_ignoring_hazard"]
            )
            if terminated or truncated:
                break

        assert max_no_brake_penalty < -0.1
        assert max_full_brake_penalty == pytest.approx(0.0)


# ── Braking cone-filter parity tests ────────────────────────────────

@pytest.fixture
def behind_peer_braking_dir(tmp_path):
    """Recording where V002 is BEHIND ego (heading=0, peer lat < ego lat)
    and brakes hard. Should NOT trigger braking_received because peer
    is outside the front cone."""
    rec_dir = tmp_path / "recordings" / "recording_01"
    rec_dir.mkdir(parents=True)

    tx_path = rec_dir / "V001_tx.csv"
    rx_path = rec_dir / "V001_rx.csv"

    n_steps = 60
    base_lat = 32.86
    speed_lat_per_step = 10.0 * 0.1 / 111_000.0

    with open(tx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            lat = base_lat + i * speed_lat_per_step
            w.writerow(_make_tx_row(t, lat=lat, speed=10.0, heading=0.0))

    # Peer is 25m BEHIND ego (lower lat)
    peer_offset_lat = -25.0 / 111_000.0
    with open(rx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            lat = base_lat + peer_offset_lat + i * speed_lat_per_step
            accel = -8.0 if i >= 20 else 0.0
            w.writerow(_make_rx_row(
                t, vid="V002", lat=lat, speed=10.0,
                heading=0.0, accel_y=accel,
            ))

    return str(tmp_path / "recordings")


class TestBrakingConeFilter:
    def test_behind_peer_braking_not_detected(self, behind_peer_braking_dir):
        """Braking peer BEHIND ego should NOT trigger braking_received
        (cone filter excludes it)."""
        env = ReplayConvoyEnv(
            recordings_dir=behind_peer_braking_dir,
            augment=False,
            max_steps=500,
            seed=42,
        )
        env.reset()
        max_braking = 0.0
        for _ in range(59):
            _, _, terminated, truncated, info = env.step(np.array([0.0]))
            max_braking = max(max_braking, info["braking_received_decay"])
            if terminated or truncated:
                break
        assert max_braking < 0.01, (
            f"Behind-ego braking peer should not trigger braking_received, "
            f"got max={max_braking}"
        )

    def test_front_peer_braking_detected(self, braking_recordings_dir):
        """Braking peer AHEAD of ego should trigger braking_received."""
        env = ReplayConvoyEnv(
            recordings_dir=braking_recordings_dir,
            augment=False,
            max_steps=500,
            seed=42,
        )
        env.reset()
        found = False
        for _ in range(99):
            _, _, terminated, truncated, info = env.step(np.array([0.0]))
            if info["braking_received_decay"] > 0.5:
                found = True
                break
            if terminated or truncated:
                break
        assert found, "Front peer braking should trigger braking_received"


# ── Random start / skip-stationary tests ─────────────────────────────

@pytest.fixture
def stationary_startup_dir(tmp_path):
    """Recording with 20 stationary steps (speed=0) then 80 moving steps."""
    rec_dir = tmp_path / "recordings" / "recording_01"
    rec_dir.mkdir(parents=True)

    tx_path = rec_dir / "V001_tx.csv"
    rx_path = rec_dir / "V001_rx.csv"

    n_steps = 100
    base_lat = 32.86
    peer_offset_lat = 25.0 / 111_000.0
    speed_lat_per_step = 10.0 * 0.1 / 111_000.0

    with open(tx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            speed = 0.0 if i < 20 else 10.0
            lat = base_lat + max(0, i - 20) * speed_lat_per_step
            w.writerow(_make_tx_row(t, lat=lat, speed=speed, heading=0.0))

    with open(rx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_HEADER)
        for i in range(n_steps):
            t = 1000 + i * 100
            lat = base_lat + peer_offset_lat + max(0, i - 20) * speed_lat_per_step
            w.writerow(_make_rx_row(
                t, vid="V002", lat=lat, speed=(0.0 if i < 20 else 10.0),
                heading=0.0,
            ))

    return str(tmp_path / "recordings")


class TestStartOffset:
    def test_skip_stationary_startup(self, stationary_startup_dir):
        """Env should skip the stationary startup and begin at first
        moving snapshot."""
        env = ReplayConvoyEnv(
            recordings_dir=stationary_startup_dir,
            augment=False,
            max_steps=500,
            seed=42,
        )
        obs, info = env.reset()
        # Should start at moving snapshot (speed ~ 10 m/s → ego[0] ~ 0.333)
        assert obs["ego"][0] > 0.1, (
            f"Expected non-zero speed after skipping startup, got ego[0]={obs['ego'][0]}"
        )
        assert info["start_offset"] >= 20

    def test_random_start_varies_offset(self, recordings_dir):
        """With random_start=True, different seeds should produce
        different start offsets."""
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=50,
            seed=42,
            random_start=True,
        )
        offsets = set()
        for s in range(10):
            _, info = env.reset(seed=s)
            offsets.add(info["start_offset"])
        assert len(offsets) > 1, "random_start should produce varied offsets"

    def test_no_random_start_is_deterministic(self, recordings_dir):
        """Without random_start, reset always starts at same offset."""
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=False,
            max_steps=50,
            seed=42,
            random_start=False,
        )
        offsets = set()
        for s in range(5):
            _, info = env.reset(seed=s)
            offsets.add(info["start_offset"])
        assert len(offsets) == 1
