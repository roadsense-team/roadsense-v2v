"""
Integration test for the Deep Sets training pipeline.
"""
import os

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from ml.policies.deep_set_policy import create_deep_set_policy_kwargs
import ml.envs  # Registers environments


SUMO_AVAILABLE = os.environ.get("SUMO_AVAILABLE", "false").lower() == "true"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_training_pipeline_runs_and_saves(tmp_path, scenario_path):
    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=scenario_path,
        max_steps=50,
        render_mode=None,
    )

    policy_kwargs = create_deep_set_policy_kwargs(peer_embed_dim=32)

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=32,
        batch_size=32,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=100)
        save_path = tmp_path / "deep_sets_test"
        model.save(str(save_path))
    finally:
        env.close()

    assert save_path.with_suffix(".zip").exists()
