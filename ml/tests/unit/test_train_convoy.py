"""
Unit tests for train_convoy CLI and evaluation utilities (Phase 3).
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.training import train_convoy


class DummyEnv:
    """Minimal env stub for evaluation tests."""

    def __init__(self, episodes):
        self._episodes = episodes
        self._ep_index = 0
        self._step_index = 0

    def reset(self):
        self._step_index = 0
        scenario_id = self._episodes[self._ep_index]["scenario_id"]
        return "obs", {"scenario_id": scenario_id}

    def step(self, _action):
        episode = self._episodes[self._ep_index]
        reward, terminated, truncated = episode["steps"][self._step_index]
        self._step_index += 1

        if terminated or truncated:
            self._ep_index += 1

        return "obs", reward, terminated, truncated, {}

    def close(self):
        pass


def _make_args(**overrides):
    defaults = dict(
        dataset_dir="/tmp/dataset",
        sumo_cfg=None,
        emulator_params="/tmp/emulator.json",
        seed=123,
        eval_episodes=2,
        gui=False,
        run_id="run_123",
        total_timesteps=100,
        learning_rate=1e-4,
        n_steps=32,
        ent_coef=0.01,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_convoy"])

    args = train_convoy.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(train_convoy.__file__)))
    expected = os.path.join(base_dir, "scenarios", "base", "scenario.sumocfg")

    assert args.dataset_dir is None
    assert args.sumo_cfg == expected
    assert args.eval_episodes == 10
    assert args.emulator_params is None
    assert args.seed == 42
    assert args.total_timesteps == 100_000
    assert args.learning_rate == 3e-4
    assert args.n_steps == 2048
    assert args.ent_coef == 0.01
    assert args.output_dir is None
    assert args.run_id is None
    assert args.skip_eval is False
    assert args.gui is False


def test_parse_args_dataset_dir(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_convoy", "--dataset_dir", "/tmp/data"])

    args = train_convoy.parse_args()

    assert args.dataset_dir == "/tmp/data"
    assert args.sumo_cfg is None


def test_parse_args_mutual_exclusion(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_convoy", "--dataset_dir", "/tmp/data", "--sumo_cfg", "/tmp/a.sumocfg"],
    )

    with pytest.raises(SystemExit):
        train_convoy.parse_args()


def test_parse_args_all_hyperparams(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_convoy",
            "--sumo_cfg",
            "/tmp/test.sumocfg",
            "--eval_episodes",
            "5",
            "--emulator_params",
            "/tmp/emu.json",
            "--seed",
            "7",
            "--total_timesteps",
            "500",
            "--learning_rate",
            "0.001",
            "--n_steps",
            "64",
            "--ent_coef",
            "0.02",
            "--output_dir",
            "/tmp/out",
            "--run_id",
            "run_test",
            "--skip_eval",
            "--gui",
        ],
    )

    args = train_convoy.parse_args()

    assert args.sumo_cfg == "/tmp/test.sumocfg"
    assert args.eval_episodes == 5
    assert args.emulator_params == "/tmp/emu.json"
    assert args.seed == 7
    assert args.total_timesteps == 500
    assert args.learning_rate == 0.001
    assert args.n_steps == 64
    assert args.ent_coef == 0.02
    assert args.output_dir == "/tmp/out"
    assert args.run_id == "run_test"
    assert args.skip_eval is True
    assert args.gui is True


def test_evaluate_returns_metrics(monkeypatch):
    episodes = [
        {"scenario_id": "eval_000", "steps": [(1.0, False, False), (2.0, True, False)]},
        {"scenario_id": "eval_001", "steps": [(0.5, False, True)]},
    ]
    dummy_env = DummyEnv(episodes)

    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)

    captured_kwargs = {}

    def _make_env(_env_id, **kwargs):
        captured_kwargs.update(kwargs)
        return dummy_env

    monkeypatch.setattr(train_convoy.gym, "make", _make_env)
    monkeypatch.setattr(train_convoy.PPO, "load", lambda _path: mock_model)

    args = _make_args()
    metrics = train_convoy.evaluate("/tmp/model.zip", args)

    assert metrics["eval_episodes"] == 2
    assert metrics["collisions"] == 1
    assert metrics["truncations"] == 1
    assert metrics["scenarios_evaluated"] == ["eval_000", "eval_001"]

    expected_rewards = [3.0, 0.5]
    expected_lengths = [2, 1]
    assert metrics["avg_reward"] == float(np.mean(expected_rewards))
    assert metrics["std_reward"] == float(np.std(expected_rewards))
    assert metrics["min_reward"] == float(np.min(expected_rewards))
    assert metrics["max_reward"] == float(np.max(expected_rewards))
    assert metrics["avg_length"] == float(np.mean(expected_lengths))

    assert captured_kwargs["hazard_injection"] is False
    assert captured_kwargs["scenario_mode"] == "eval"
    assert captured_kwargs["sumo_cfg"] is None
    assert captured_kwargs["scenario_seed"] == args.seed
    assert captured_kwargs["emulator_params_path"] == args.emulator_params


def test_evaluate_uses_deterministic(monkeypatch):
    episodes = [
        {"scenario_id": "eval_000", "steps": [(1.0, True, False)]},
    ]
    dummy_env = DummyEnv(episodes)

    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)

    monkeypatch.setattr(train_convoy.gym, "make", lambda *_args, **_kwargs: dummy_env)
    monkeypatch.setattr(train_convoy.PPO, "load", lambda _path: mock_model)

    args = _make_args(eval_episodes=1)
    train_convoy.evaluate("/tmp/model.zip", args)

    assert mock_model.predict.called
    assert all(call.kwargs.get("deterministic") is True for call in mock_model.predict.call_args_list)


def test_evaluate_counts_collisions(monkeypatch):
    episodes = [
        {"scenario_id": "eval_000", "steps": [(1.0, True, False)]},
        {"scenario_id": "eval_001", "steps": [(1.0, False, True)]},
    ]
    dummy_env = DummyEnv(episodes)

    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)

    monkeypatch.setattr(train_convoy.gym, "make", lambda *_args, **_kwargs: dummy_env)
    monkeypatch.setattr(train_convoy.PPO, "load", lambda _path: mock_model)

    args = _make_args(eval_episodes=2)
    metrics = train_convoy.evaluate("/tmp/model.zip", args)

    assert metrics["collisions"] == 1
    assert metrics["collision_rate"] == 0.5


def test_save_metrics_creates_file(tmp_path):
    args = _make_args()
    training_metrics = {"training_timesteps": 100}
    eval_metrics = {"avg_reward": 1.23}

    metrics_path = train_convoy.save_metrics(
        output_dir=str(tmp_path),
        training_metrics=training_metrics,
        eval_metrics=eval_metrics,
        args=args,
    )

    assert os.path.exists(metrics_path)


def test_save_metrics_schema(tmp_path):
    args = _make_args()
    training_metrics = {"training_timesteps": 100}
    eval_metrics = {"avg_reward": 1.23}

    metrics_path = train_convoy.save_metrics(
        output_dir=str(tmp_path),
        training_metrics=training_metrics,
        eval_metrics=eval_metrics,
        args=args,
    )

    data = json.loads(open(metrics_path, "r", encoding="utf-8").read())

    assert data["run_id"] == args.run_id
    assert "timestamp" in data
    assert data["config"]["dataset_dir"] == args.dataset_dir
    assert data["config"]["sumo_cfg"] == args.sumo_cfg
    assert data["config"]["emulator_params"] == args.emulator_params
    assert data["config"]["seed"] == args.seed
    assert data["config"]["total_timesteps"] == args.total_timesteps
    assert data["config"]["learning_rate"] == args.learning_rate
    assert data["config"]["n_steps"] == args.n_steps
    assert data["config"]["ent_coef"] == args.ent_coef
    assert data["training"] == training_metrics
    assert data["evaluation"] == eval_metrics
