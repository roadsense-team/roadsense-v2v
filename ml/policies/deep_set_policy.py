"""
Policy helpers for Deep Sets integration with SB3.
"""
from typing import Dict, Any

from ml.models.deep_set_extractor import DeepSetExtractor


def create_deep_set_policy_kwargs(peer_embed_dim: int = 32) -> Dict[str, Any]:
    """
    Create SB3 policy kwargs that use the DeepSetExtractor.

    Args:
        peer_embed_dim: Embedding size for peer encoder (default 32).

    Returns:
        Policy kwargs dict for stable-baselines3.
    """
    return dict(
        features_extractor_class=DeepSetExtractor,
        features_extractor_kwargs=dict(embed_dim=peer_embed_dim),
    )
