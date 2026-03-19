"""Unit tests for replay fine-tuning config helpers."""

import pytest

import ml.scripts.run_replay_fine_tuning as replay_ft
from ml.scripts.run_replay_fine_tuning import (
    build_replay_reward_config,
    make_linear_schedule,
)


def test_build_replay_reward_config_defaults_to_threshold_only_gate():
    config = build_replay_reward_config(
        ignore_hazard_threshold=0.15,
        ignore_danger_distance=20.0,
        ignore_danger_closing_rate=0.5,
        ignore_require_danger_geometry=False,
    )
    assert config["ignoring_hazard_threshold"] == pytest.approx(0.15)
    assert config["ignoring_require_danger_geometry"] is False
    assert config["ignoring_use_any_braking_peer"] is True
    assert config["early_reaction_threshold"] == pytest.approx(0.01)


def test_build_replay_reward_config_can_enable_geometry_gate():
    config = build_replay_reward_config(
        ignore_hazard_threshold=0.3,
        ignore_danger_distance=25.0,
        ignore_danger_closing_rate=0.3,
        ignore_require_danger_geometry=True,
    )
    assert config["ignoring_hazard_threshold"] == pytest.approx(0.3)
    assert config["ignoring_danger_distance"] == pytest.approx(25.0)
    assert config["ignoring_danger_closing_rate"] == pytest.approx(0.3)
    assert config["ignoring_require_danger_geometry"] is True


def test_linear_schedule_returns_initial_at_start():
    schedule = make_linear_schedule(1e-4, 1e-5)
    assert schedule(1.0) == pytest.approx(1e-4)


def test_linear_schedule_returns_final_at_end():
    schedule = make_linear_schedule(1e-4, 1e-5)
    assert schedule(0.0) == pytest.approx(1e-5)


def test_linear_schedule_interpolates_midpoint():
    schedule = make_linear_schedule(1e-4, 1e-5)
    expected = 1e-5 + (1e-4 - 1e-5) * 0.5
    assert schedule(0.5) == pytest.approx(expected)


def test_make_env_passes_shadow_reward_geometry_flag(monkeypatch):
    captured = {}

    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(replay_ft, "ReplayConvoyEnv", FakeEnv)
    monkeypatch.setattr(replay_ft, "Monitor", lambda env: env)

    env_factory = replay_ft.make_env(
        recordings_dir="ml/data/recordings",
        augment=False,
        seed=42,
        use_recorded_ego=True,
        use_shadow_reward_geometry=True,
        reward_config={"ignoring_require_danger_geometry": True},
    )
    env_factory()

    assert captured["use_recorded_ego"] is True
    assert captured["use_shadow_reward_geometry"] is True
    assert captured["reward_config"] == {"ignoring_require_danger_geometry": True}
