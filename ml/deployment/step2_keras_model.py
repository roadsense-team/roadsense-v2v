"""
Step 2: Define the Keras model that exactly mirrors the SB3 DeepSetExtractor + policy.

The architecture replicates the PyTorch forward pass:
  1. Peer encoder MLP  (6 → 32 → 32, ReLU)  applied to each of MAX_PEERS peers
  2. Zero-mask before max-pool (safe because ReLU outputs are ≥ 0)
  3. Reduce-max across peer axis → (32,)
  4. Concatenate with ego (18,) → (50,)
  5. Policy MLP (50 → 64 → 64 → 1, Tanh)
  6. Clip output to [0, 1]

Why zero-masking ≡ -inf masking (the key insight):
  The original PyTorch code fills invalid peer embeddings with -inf before max-pool.
  Because the peer encoder ends with ReLU (outputs ≥ 0), replacing -inf with 0 is
  equivalent: valid peers are always ≥ 0, invalid peers are forced to 0, and the
  max(valid ≥ 0, zeros) = max of valid peers.  All-zero case (no peers) → zero vector.

This lets us express the full pipeline with standard TFLite ops only.

Inputs:
  ego:       (batch, 18)  — 6 features × 3 stacked frames, already normalized
  peers:     (batch, 8, 6) — per-peer features, already normalized
  peer_mask: (batch, 8)   — 1.0 = valid peer, 0.0 = padding

Output:
  action:    (batch, 1)   — braking signal in [0, 1]

Run from repo root (inside Docker):
  python -m ml.deployment.step2_keras_model
"""
import json
import os

import numpy as np
import tensorflow as tf


class PeerMaxPool(tf.keras.layers.Layer):
    """Reduce-max over the peer axis (axis=1)."""
    def call(self, x):
        return tf.reduce_max(x, axis=1)

    def get_config(self):
        return super().get_config()


class ClipAction(tf.keras.layers.Layer):
    """Clip output to [0, 1]."""
    def call(self, x):
        return tf.clip_by_value(x, 0.0, 1.0)

    def get_config(self):
        return super().get_config()


def build_deepset_policy(
    ego_dim: int = 18,
    peer_feature_dim: int = 6,
    embed_dim: int = 32,
    max_peers: int = 8,
    policy_hidden_1: int = 64,
    policy_hidden_2: int = 64,
    action_dim: int = 1,
) -> "tf.keras.Model":
    """
    Build and return the Keras DeepSet policy model.

    Returns a tf.keras.Model with three inputs and one output.
    """
    # ------------------------------------------------------------------ #
    # Inputs                                                               #
    # ------------------------------------------------------------------ #
    ego_input   = tf.keras.Input(shape=(ego_dim,),              name="ego",       dtype=tf.float32)
    peers_input = tf.keras.Input(shape=(max_peers, peer_feature_dim), name="peers", dtype=tf.float32)
    mask_input  = tf.keras.Input(shape=(max_peers,),            name="peer_mask", dtype=tf.float32)

    # ------------------------------------------------------------------ #
    # Peer encoder  (shared weights across all peers, applied via          #
    # TimeDistributed which is equivalent to Dense applied to last axis)  #
    # ------------------------------------------------------------------ #
    enc1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(embed_dim, activation="relu", name="peer_enc_dense1"),
        name="td_enc1"
    )(peers_input)                                    # (batch, 8, 32)

    enc2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(embed_dim, activation="relu", name="peer_enc_dense2"),
        name="td_enc2"
    )(enc1)                                           # (batch, 8, 32)

    # ------------------------------------------------------------------ #
    # Zero-masking: multiply each peer embedding by its mask flag         #
    # Shape: (batch, 8, 1) * (batch, 8, 32) → (batch, 8, 32)             #
    # ------------------------------------------------------------------ #
    mask_expanded = tf.keras.layers.Reshape(
        (max_peers, 1), name="mask_reshape"
    )(mask_input)                                     # (batch, 8, 1)

    masked_enc = tf.keras.layers.Multiply(name="apply_mask")([enc2, mask_expanded])  # (batch, 8, 32)

    # ------------------------------------------------------------------ #
    # Max-pool across peer dimension → (batch, 32)                        #
    # ------------------------------------------------------------------ #
    peer_pool = PeerMaxPool(name="peer_max_pool")(masked_enc)  # (batch, 32)

    # ------------------------------------------------------------------ #
    # Concatenate peer embedding with ego state                           #
    # ------------------------------------------------------------------ #
    combined = tf.keras.layers.Concatenate(name="concat_ego_peers")([peer_pool, ego_input])
    # (batch, 32 + 18) = (batch, 50)

    # ------------------------------------------------------------------ #
    # Policy MLP: Tanh activations (matching SB3 default)                 #
    # ------------------------------------------------------------------ #
    h1 = tf.keras.layers.Dense(policy_hidden_1, activation="tanh", name="policy_dense1")(combined)
    h2 = tf.keras.layers.Dense(policy_hidden_2, activation="tanh", name="policy_dense2")(h1)
    action_logit = tf.keras.layers.Dense(action_dim, name="action_out")(h2)

    # ------------------------------------------------------------------ #
    # Clip to [0, 1] — deterministic action (mean of the Gaussian)       #
    # ------------------------------------------------------------------ #
    action = ClipAction(name="action_clip")(action_logit)

    model = tf.keras.Model(
        inputs={"ego": ego_input, "peers": peers_input, "peer_mask": mask_input},
        outputs=action,
        name="deepset_policy",
    )
    return model


def load_config(config_path: str = "ml/deployment/artifacts/model_config.json") -> dict:
    with open(config_path) as f:
        return json.load(f)


def print_model_summary(model) -> None:
    model.summary()
    print(f"\nInput shapes:")
    for inp in model.inputs:
        print(f"  {inp.name}: {inp.shape}")
    print(f"Output shape: {model.output.shape}")
    total = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"Trainable parameters: {total:,}")


if __name__ == "__main__":
    cfg = load_config()
    model = build_deepset_policy(
        ego_dim=cfg["ego_dim"],
        peer_feature_dim=cfg["peer_feature_dim"],
        embed_dim=cfg["embed_dim"],
        max_peers=cfg["max_peers"],
        policy_hidden_1=cfg["policy_hidden_1"],
        policy_hidden_2=cfg["policy_hidden_2"],
        action_dim=cfg["action_dim"],
    )
    print_model_summary(model)
