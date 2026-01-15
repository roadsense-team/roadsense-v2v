"""
Unit tests for DeepSetExtractor.
"""
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from models.deep_set_extractor import DeepSetExtractor


def _make_observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "ego": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            "peers": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(8, 6),
                dtype=np.float32,
            ),
            "peer_mask": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32,
            ),
        }
    )


def _make_batch(peer_mask: torch.Tensor, seed: int = 0) -> Dict[str, torch.Tensor]:
    batch_size = peer_mask.shape[0]
    torch.manual_seed(seed)
    ego = torch.randn((batch_size, 4))
    peers = torch.randn((batch_size, 8, 6))
    return {"ego": ego, "peers": peers, "peer_mask": peer_mask}


class _DummyConvoyEnv(gym.Env):
    """Minimal env to verify SB3 integration."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = _make_observation_space()
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._sample_obs()
        return obs, {}

    def step(self, action):
        obs = self._sample_obs()
        return obs, 0.0, False, False, {}

    def _sample_obs(self):
        return {
            "ego": np.zeros((4,), dtype=np.float32),
            "peers": np.zeros((8, 6), dtype=np.float32),
            "peer_mask": np.zeros((8,), dtype=np.float32),
        }


def test_extractor_output_shape():
    observation_space = _make_observation_space()
    extractor = DeepSetExtractor(observation_space)

    peer_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    observations = _make_batch(peer_mask)

    features = extractor(observations)

    assert features.shape == (3, 36)


def test_extractor_handles_zero_peers():
    observation_space = _make_observation_space()
    extractor = DeepSetExtractor(observation_space)

    peer_mask = torch.zeros((2, 8), dtype=torch.float32)
    observations = _make_batch(peer_mask)

    features = extractor(observations)

    assert torch.isfinite(features).all()
    assert torch.allclose(features[:, :32], torch.zeros_like(features[:, :32]))
    assert torch.allclose(features[:, 32:], observations["ego"])


def test_extractor_handles_max_peers():
    observation_space = _make_observation_space()
    extractor = DeepSetExtractor(observation_space)

    peer_mask = torch.ones((4, 8), dtype=torch.float32)
    observations = _make_batch(peer_mask)

    features = extractor(observations)

    assert features.shape == (4, 36)


def test_extractor_permutation_invariant():
    observation_space = _make_observation_space()
    extractor = DeepSetExtractor(observation_space)

    torch.manual_seed(123)
    ego = torch.randn((1, 4))
    peers = torch.randn((1, 8, 6))
    peer_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=torch.float32,
    )

    perm = torch.tensor([3, 1, 0, 2, 4, 5, 6, 7])
    obs_a = {"ego": ego, "peers": peers, "peer_mask": peer_mask}
    obs_b = {
        "ego": ego.clone(),
        "peers": peers[:, perm],
        "peer_mask": peer_mask[:, perm],
    }

    features_a = extractor(obs_a)
    features_b = extractor(obs_b)

    assert torch.allclose(features_a, features_b, atol=1e-6)


def test_extractor_features_dim_is_36():
    observation_space = _make_observation_space()
    extractor = DeepSetExtractor(observation_space)

    assert extractor.features_dim == 36


def test_extractor_works_with_sb3_ppo():
    env = _DummyConvoyEnv()

    policy_kwargs = dict(
        features_extractor_class=DeepSetExtractor,
        features_extractor_kwargs=dict(embed_dim=32),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    assert action.shape == env.action_space.shape
