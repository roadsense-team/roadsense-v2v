"""
Deep Sets feature extractor for variable-size peer observations.
"""
from typing import Dict

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DeepSetExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation spaces with variable peers.

    Implements Deep Sets architecture:
    1. Shared MLP encodes each peer independently
    2. Max pooling aggregates peer embeddings (masked)
    3. Concatenates pooled features with ego state

    Compatible with stable-baselines3 PPO/A2C/SAC.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        peer_feature_dim: int = 6,
        ego_feature_dim: int = 4,
        embed_dim: int = 32,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("DeepSetExtractor requires Dict observation space.")

        expected_keys = {"ego", "peers", "peer_mask"}
        missing = expected_keys - set(observation_space.spaces.keys())
        if missing:
            raise ValueError(
                "DeepSetExtractor missing observation keys: "
                + ", ".join(sorted(missing))
            )

        super().__init__(
            observation_space,
            features_dim=embed_dim + ego_feature_dim,
        )
        self._features_dim = embed_dim + ego_feature_dim

        self.peer_encoder = nn.Sequential(
            nn.Linear(peer_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict with keys:
                - 'ego': (batch, 4) ego features
                - 'peers': (batch, 8, 6) peer features
                - 'peer_mask': (batch, 8) validity mask (1=valid, 0=padded)

        Returns:
            features: (batch, 36) combined features for policy head
        """
        ego = observations["ego"]
        peers = observations["peers"]
        peer_mask = observations["peer_mask"]

        peer_embeddings = self.peer_encoder(peers)

        peer_mask_bool = peer_mask.bool()
        mask_expanded = peer_mask_bool.unsqueeze(-1).expand_as(peer_embeddings)
        peer_embeddings = peer_embeddings.masked_fill(
            ~mask_expanded,
            float("-inf"),
        )

        z, _ = peer_embeddings.max(dim=1)

        all_masked = ~peer_mask_bool.any(dim=1, keepdim=True)
        z = torch.where(all_masked, torch.zeros_like(z), z)

        combined = torch.cat([z, ego], dim=1)
        return combined
