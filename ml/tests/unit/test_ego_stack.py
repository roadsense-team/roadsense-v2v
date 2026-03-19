"""
Unit tests for Run 025 — Temporal Ego Stack.

Tests:
- ReplayConvoyEnv: ego_stack_frames param, obs shape, history, order
- ConvoyEnv: observation space shape (requires SUMO mocking)
- DeepSetExtractor: ego_feature_dim=18, features_dim=50
- Policy kwargs helper: ego_feature_dim passthrough
"""

import csv
import os
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from ml.envs.replay_convoy_env import ReplayConvoyEnv
from ml.envs.observation_builder import ObservationBuilder
from ml.models.deep_set_extractor import DeepSetExtractor
from ml.policies.deep_set_policy import create_deep_set_policy_kwargs


# ---------------------------------------------------------------------------
# Helpers — minimal recording fixtures for ReplayConvoyEnv
# ---------------------------------------------------------------------------

def _write_tx_csv(path: Path, n_rows: int = 20, speed: float = 10.0):
    """Write a minimal TX CSV with n_rows at 100ms intervals."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp_local_ms", "msg_timestamp", "vehicle_id",
            "lat", "lon", "speed", "heading", "accel_x", "accel_y",
            "hop_count",
        ])
        writer.writeheader()
        for i in range(n_rows):
            t = 1000 + i * 100
            writer.writerow({
                "timestamp_local_ms": t,
                "msg_timestamp": t,
                "vehicle_id": "V001",
                "lat": 32.0 + i * 0.00001,
                "lon": 34.0,
                "speed": speed,
                "heading": 0.0,
                "accel_x": 0.0,
                "accel_y": -0.5 if i >= 10 else 0.0,  # braking in second half
                "hop_count": 0,
            })


def _write_rx_csv(path: Path, n_rows: int = 20):
    """Write a minimal RX CSV with one peer (V002) ahead of ego."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp_local_ms", "msg_timestamp", "vehicle_id",
            "lat", "lon", "speed", "heading", "accel_x", "accel_y",
            "hop_count",
        ])
        writer.writeheader()
        for i in range(n_rows):
            t = 1000 + i * 100
            writer.writerow({
                "timestamp_local_ms": t,
                "msg_timestamp": t,
                "vehicle_id": "V002",
                "lat": 32.0 + 0.001 + i * 0.00001,  # ahead of ego
                "lon": 34.0,
                "speed": 10.0,
                "heading": 0.0,
                "accel_x": 0.0,
                "accel_y": -3.0 if i >= 12 else 0.0,  # peer brakes later
                "hop_count": 0,
            })


def _make_recording_dir(tmp_path: Path, n_rows: int = 20) -> Path:
    """Create a minimal recording directory for ReplayConvoyEnv."""
    rec_dir = tmp_path / "recordings" / "rec_001"
    rec_dir.mkdir(parents=True)
    _write_tx_csv(rec_dir / "V001_tx.csv", n_rows=n_rows)
    _write_rx_csv(rec_dir / "V001_rx.csv", n_rows=n_rows)
    return tmp_path / "recordings"


# ---------------------------------------------------------------------------
# ReplayConvoyEnv stacking tests
# ---------------------------------------------------------------------------

class TestReplayConvoyEnvStacking:
    """Tests for ego frame stacking in ReplayConvoyEnv."""

    def test_default_stack_1_backward_compat(self, tmp_path):
        """ego_stack_frames=1 produces (6,) ego — unchanged from Run 023."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=1,
        )
        assert env.observation_space["ego"].shape == (6,)
        obs, _ = env.reset()
        assert obs["ego"].shape == (6,)

    def test_stack_3_produces_18_dim_ego(self, tmp_path):
        """ego_stack_frames=3 produces (18,) ego."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        assert env.observation_space["ego"].shape == (18,)
        obs, _ = env.reset()
        assert obs["ego"].shape == (18,)

    def test_reset_fills_all_frames_identical(self, tmp_path):
        """After reset, all 3 frames should be identical copies."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        obs, _ = env.reset()
        ego = obs["ego"]
        # [ego_t, ego_{t-1}, ego_{t-2}] — all same on reset
        frame0 = ego[0:6]
        frame1 = ego[6:12]
        frame2 = ego[12:18]
        np.testing.assert_array_equal(frame0, frame1)
        np.testing.assert_array_equal(frame1, frame2)

    def test_stacking_order_most_recent_first(self, tmp_path):
        """After stepping, obs[0:6] is the most recent frame."""
        rec_dir = _make_recording_dir(tmp_path, n_rows=30)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        obs, _ = env.reset()
        reset_ego = obs["ego"][0:6].copy()

        # Step with zero action — ego state evolves from the recording
        obs1, _, _, _, _ = env.step(np.array([0.0]))
        step1_current = obs1["ego"][0:6].copy()
        step1_prev = obs1["ego"][6:12].copy()

        # The previous frame (6:12) should equal the reset frame
        np.testing.assert_array_equal(step1_prev, reset_ego)
        # Current frame might differ (different timestep in recording)

    def test_after_3_steps_frames_are_historical(self, tmp_path):
        """After 3+ steps, each frame slot contains a distinct historical ego."""
        rec_dir = _make_recording_dir(tmp_path, n_rows=30)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        obs, _ = env.reset()
        collected_egos = [obs["ego"][0:6].copy()]

        for i in range(4):
            obs, _, _, _, _ = env.step(np.array([0.0]))
            collected_egos.append(obs["ego"][0:6].copy())

        # After step 3+, the stacked observation should contain the
        # 3 most recent ego vectors
        final_ego = obs["ego"]
        frame0 = final_ego[0:6]   # most recent = collected_egos[-1]
        frame1 = final_ego[6:12]  # t-1 = collected_egos[-2]
        frame2 = final_ego[12:18] # t-2 = collected_egos[-3]

        np.testing.assert_array_equal(frame0, collected_egos[-1])
        np.testing.assert_array_equal(frame1, collected_egos[-2])
        np.testing.assert_array_equal(frame2, collected_egos[-3])

    def test_peers_and_mask_unchanged(self, tmp_path):
        """Peers and peer_mask shapes are not affected by ego stacking."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        assert env.observation_space["peers"].shape == (8, 6)
        assert env.observation_space["peer_mask"].shape == (8,)
        obs, _ = env.reset()
        assert obs["peers"].shape == (8, 6)
        assert obs["peer_mask"].shape == (8,)

    def test_obs_matches_observation_space(self, tmp_path):
        """Observation must be valid under the declared observation space."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            f"obs ego shape {obs['ego'].shape} vs space {env.observation_space['ego'].shape}"
        )

        obs2, _, _, _, _ = env.step(np.array([0.0]))
        assert env.observation_space.contains(obs2)

    def test_invalid_stack_frames_raises(self, tmp_path):
        """ego_stack_frames < 1 should raise ValueError."""
        rec_dir = _make_recording_dir(tmp_path)
        with pytest.raises(ValueError, match="ego_stack_frames must be >= 1"):
            ReplayConvoyEnv(
                recordings_dir=str(rec_dir),
                augment=False,
                seed=42,
                ego_stack_frames=0,
            )

    def test_stack_2_produces_12_dim(self, tmp_path):
        """ego_stack_frames=2 produces (12,) ego."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=2,
        )
        assert env.observation_space["ego"].shape == (12,)
        obs, _ = env.reset()
        assert obs["ego"].shape == (12,)

    def test_observation_space_bounds_tiled(self, tmp_path):
        """Observation space low/high should be tiled single-frame bounds."""
        rec_dir = _make_recording_dir(tmp_path)
        env = ReplayConvoyEnv(
            recordings_dir=str(rec_dir),
            augment=False,
            seed=42,
            ego_stack_frames=3,
        )
        ego_space = env.observation_space["ego"]
        expected_low = np.tile([0.0, -1.0, 0.0, -1.0, 0.0, 0.0], 3)
        expected_high = np.tile([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], 3)
        np.testing.assert_array_equal(ego_space.low, expected_low.astype(np.float32))
        np.testing.assert_array_equal(ego_space.high, expected_high.astype(np.float32))


# ---------------------------------------------------------------------------
# DeepSetExtractor tests with ego_feature_dim=18
# ---------------------------------------------------------------------------

class TestDeepSetExtractorStacking:
    """Tests for DeepSetExtractor with larger ego dimension."""

    def _make_obs_space(self, ego_dim: int = 6):
        """Build a Dict observation space with variable ego dim."""
        return gym.spaces.Dict({
            "ego": gym.spaces.Box(
                low=-np.ones(ego_dim, dtype=np.float32),
                high=np.ones(ego_dim, dtype=np.float32),
            ),
            "peers": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(8, 6), dtype=np.float32,
            ),
            "peer_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32,
            ),
        })

    def test_default_ego_6_features_dim_38(self):
        """Default: ego_feature_dim=6, embed_dim=32 → features_dim=38."""
        obs_space = self._make_obs_space(6)
        extractor = DeepSetExtractor(obs_space, ego_feature_dim=6, embed_dim=32)
        assert extractor.features_dim == 38

    def test_ego_18_features_dim_50(self):
        """Stacked: ego_feature_dim=18, embed_dim=32 → features_dim=50."""
        obs_space = self._make_obs_space(18)
        extractor = DeepSetExtractor(obs_space, ego_feature_dim=18, embed_dim=32)
        assert extractor.features_dim == 50

    def test_forward_pass_ego_18(self):
        """Forward pass with 18-dim ego produces (batch, 50) features."""
        obs_space = self._make_obs_space(18)
        extractor = DeepSetExtractor(obs_space, ego_feature_dim=18, embed_dim=32)

        batch_size = 4
        obs = {
            "ego": torch.randn(batch_size, 18),
            "peers": torch.randn(batch_size, 8, 6),
            "peer_mask": torch.ones(batch_size, 8),
        }
        features = extractor(obs)
        assert features.shape == (batch_size, 50)

    def test_forward_pass_ego_6_backward_compat(self):
        """Forward pass with default 6-dim ego still works."""
        obs_space = self._make_obs_space(6)
        extractor = DeepSetExtractor(obs_space, ego_feature_dim=6, embed_dim=32)

        batch_size = 4
        obs = {
            "ego": torch.randn(batch_size, 6),
            "peers": torch.randn(batch_size, 8, 6),
            "peer_mask": torch.ones(batch_size, 8),
        }
        features = extractor(obs)
        assert features.shape == (batch_size, 38)

    def test_forward_pass_no_valid_peers(self):
        """With all-zero peer mask, output should still be correct shape."""
        obs_space = self._make_obs_space(18)
        extractor = DeepSetExtractor(obs_space, ego_feature_dim=18, embed_dim=32)

        obs = {
            "ego": torch.randn(2, 18),
            "peers": torch.randn(2, 8, 6),
            "peer_mask": torch.zeros(2, 8),  # no valid peers
        }
        features = extractor(obs)
        assert features.shape == (2, 50)


# ---------------------------------------------------------------------------
# Policy kwargs helper tests
# ---------------------------------------------------------------------------

class TestPolicyKwargsHelper:
    """Tests for create_deep_set_policy_kwargs ego_feature_dim passthrough."""

    def test_default_ego_feature_dim_6(self):
        kwargs = create_deep_set_policy_kwargs()
        assert kwargs["features_extractor_kwargs"]["ego_feature_dim"] == 6

    def test_ego_feature_dim_18(self):
        kwargs = create_deep_set_policy_kwargs(ego_feature_dim=18)
        assert kwargs["features_extractor_kwargs"]["ego_feature_dim"] == 18

    def test_embed_dim_passthrough(self):
        kwargs = create_deep_set_policy_kwargs(peer_embed_dim=64, ego_feature_dim=18)
        assert kwargs["features_extractor_kwargs"]["embed_dim"] == 64
        assert kwargs["features_extractor_kwargs"]["ego_feature_dim"] == 18

    def test_log_std_init_passthrough(self):
        kwargs = create_deep_set_policy_kwargs(log_std_init=-1.0)
        assert kwargs["log_std_init"] == -1.0
