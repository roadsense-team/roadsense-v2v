"""
Integration tests for ConvoyEnv with real SUMO.

These tests require:
- SUMO installed and accessible via 'sumo' command
- ml/scenarios/base/scenario.sumocfg exists

Run with: pytest -m integration
"""

import os

import numpy as np
import pytest


SUMO_AVAILABLE = os.system("sumo --version > /dev/null 2>&1") == 0


@pytest.fixture
def env(scenario_path):
    """ConvoyEnv with real SUMO."""
    from envs.convoy_env import ConvoyEnv

    env = ConvoyEnv(
        sumo_cfg=scenario_path,
        hazard_injection=False,
        max_steps=100,
    )
    yield env
    env.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_full_episode_completes_without_crash(env):
    """
    Run one episode to completion (collision or truncation).

    This test verifies:
    - SUMO starts and connects successfully
    - step() runs without exceptions
    - Episode terminates properly (terminated or truncated)
    """
    obs, info = env.reset()
    assert obs is not None
    assert "step" in info

    done = False
    step_count = 0
    max_steps = 100

    while not done and step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    assert done or step_count >= max_steps
    assert step_count > 0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_full_episode_returns_valid_observations(env):
    """
    Every step returns Dict observation with expected shapes.
    """
    obs, _ = env.reset()

    assert set(obs.keys()) == {"ego", "peers", "peer_mask"}
    assert obs["ego"].shape == (4,)
    assert obs["peers"].shape == (8, 6)
    assert obs["peer_mask"].shape == (8,)
    assert obs["ego"].dtype == np.float32
    assert obs["peers"].dtype == np.float32
    assert obs["peer_mask"].dtype == np.float32

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs["ego"].shape == (4,)
        assert obs["peers"].shape == (8, 6)
        assert obs["peer_mask"].shape == (8,)
        assert np.isfinite(obs["ego"]).all(), "Observation contains NaN or Inf"
        assert np.isfinite(obs["peers"]).all(), "Observation contains NaN or Inf"
        assert np.isfinite(obs["peer_mask"]).all(), "Observation contains NaN or Inf"

        if terminated or truncated:
            break


@pytest.mark.integration
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_gym_make_creates_convoy_env(scenario_path):
    """
    gym.make('RoadSense-Convoy-v0') returns ConvoyEnv instance.
    """
    import gymnasium as gym
    import ml.envs

    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=scenario_path,
        max_steps=10,
    )

    try:
        assert env is not None
        from envs.convoy_env import ConvoyEnv
        assert isinstance(env.unwrapped, ConvoyEnv)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_registered_env_has_correct_observation_space(scenario_path):
    """
    observation_space is Dict with expected shapes.
    """
    import gymnasium as gym
    import ml.envs

    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=scenario_path,
        max_steps=10,
    )

    try:
        from gymnasium.spaces import Dict as DictSpace
        assert isinstance(env.observation_space, DictSpace)
        assert env.observation_space["ego"].shape == (4,)
        assert env.observation_space["peers"].shape == (8, 6)
        assert env.observation_space["peer_mask"].shape == (8,)
        assert env.observation_space["ego"].dtype == np.float32
        assert env.observation_space["peers"].dtype == np.float32
        assert env.observation_space["peer_mask"].dtype == np.float32
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_registered_env_has_correct_action_space(scenario_path):
    """
    action_space is Discrete(4).
    """
    import gymnasium as gym
    import ml.envs

    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=scenario_path,
        max_steps=10,
    )

    try:
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 4
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_random_agent_100_episodes_no_crash(scenario_path):
    """
    Run 100 episodes with random actions.
    All must complete without exception.

    This is a stress test for resource management:
    - SUMO start/stop cycles
    - Emulator queue clearing
    - Memory leaks
    """
    from envs.convoy_env import ConvoyEnv

    env = ConvoyEnv(
        sumo_cfg=scenario_path,
        hazard_injection=True,
        max_steps=50,
    )

    try:
        for episode in range(100):
            obs, info = env.reset(seed=episode)
            done = False
            steps = 0

            while not done and steps < 50:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            assert done or steps >= 50, f"Episode {episode} didn't terminate properly"
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_check_env_passes(scenario_path):
    """
    stable-baselines3 check_env() passes.

    check_env() validates:
    - Observation/action space consistency
    - reset() and step() return correct types
    - No common Gymnasium API violations
    """
    from stable_baselines3.common.env_checker import check_env
    from envs.convoy_env import ConvoyEnv

    env = ConvoyEnv(
        sumo_cfg=scenario_path,
        hazard_injection=False,
        max_steps=10,
    )

    try:
        check_env(env, warn=True)
    finally:
        env.close()
