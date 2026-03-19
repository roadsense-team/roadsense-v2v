"""
Policy helpers for Deep Sets integration with SB3.
"""
from typing import Dict, Any

from ml.models.deep_set_extractor import DeepSetExtractor


def create_deep_set_policy_kwargs(
    peer_embed_dim: int = 32,
    log_std_init: float = -0.5,
    ego_feature_dim: int = 6,
) -> Dict[str, Any]:
    """
    Create SB3 policy kwargs that use the DeepSetExtractor.

    Args:
        peer_embed_dim: Embedding size for peer encoder (default 32).
        log_std_init: Initial log-std for the action distribution (default -0.5,
            gives std~0.6). Prevents std explosion during training.
        ego_feature_dim: Ego observation dimension (default 6).  Set to
            ``ego_stack_frames * 6`` when using temporal ego stacking (Run 025).

    Returns:
        Policy kwargs dict for stable-baselines3.
    """
    return dict(
        features_extractor_class=DeepSetExtractor,
        features_extractor_kwargs=dict(
            embed_dim=peer_embed_dim,
            ego_feature_dim=ego_feature_dim,
        ),
        log_std_init=log_std_init,
    )
