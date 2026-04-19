"""
Step 4: Generate golden reference I/O vectors using the original SB3 model.

These vectors are the ground truth for all subsequent validation stages:
  - Keras model must match within MAE < 1e-4
  - Float TFLite must match within MAE < 1e-4
  - INT8 TFLite must match within MAE < 0.02

Reads:
  ml/results/run_025_replay_v1/model_final.zip  (SB3 model)
  ml/deployment/artifacts/model_config.json

Writes:
  ml/deployment/artifacts/golden_vectors.npz    (inputs + reference outputs)

The golden vectors are generated from:
  1. Fixed synthetic edge cases (zeros, all-mask, single peer, all-peer)
  2. Samples from the real convoy recordings (.npz files)
  3. Random inputs with fixed seed (reproducible)

Run from repo root (inside Docker):
  python -m ml.deployment.step4_gen_golden_vectors
"""
import argparse
import json
import os
import sys
import zipfile
import io
from typing import List, Tuple

import numpy as np
import torch


DEFAULT_MODEL  = "ml/results/run_025_replay_v1/model_final.zip"
DEFAULT_CONFIG = "ml/deployment/artifacts/model_config.json"
DEFAULT_OUT    = "ml/deployment/artifacts/golden_vectors.npz"
DEFAULT_SEED   = 42
N_RANDOM       = 200


def load_sb3_model(model_zip_path: str):
    """Load the SB3 PPO model for inference."""
    sys.path.insert(0, ".")
    import ml.envs  # registers environments
    from stable_baselines3 import PPO
    model = PPO.load(model_zip_path, device="cpu")
    return model


def predict_sb3(model, obs_dict: dict) -> np.ndarray:
    """
    Run deterministic inference with the SB3 model.
    obs_dict keys: 'ego', 'peers', 'peer_mask' — each (batch, ...).
    Returns actions (batch, 1).
    """
    # SB3 predict expects a dict of arrays, one env at a time
    results = []
    batch = obs_dict["ego"].shape[0]
    for i in range(batch):
        single = {
            "ego":       obs_dict["ego"][i:i+1],
            "peers":     obs_dict["peers"][i:i+1],
            "peer_mask": obs_dict["peer_mask"][i:i+1],
        }
        action, _ = model.predict(single, deterministic=True)
        results.append(action.reshape(1, -1))
    return np.concatenate(results, axis=0).astype(np.float32)


def make_synthetic_cases(cfg: dict) -> dict:
    """Fixed edge-case inputs with known semantics."""
    E = cfg["ego_dim"]
    P = cfg["max_peers"]
    F = cfg["peer_feature_dim"]

    cases = []

    # Case 0: All zeros, no peers
    ego = np.zeros((1, E), dtype=np.float32)
    peers = np.zeros((1, P, F), dtype=np.float32)
    mask = np.zeros((1, P), dtype=np.float32)
    cases.append((ego, peers, mask))

    # Case 1: Ego moving at moderate speed, no peers
    ego = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    peers = np.zeros((1, P, F), dtype=np.float32)
    mask = np.zeros((1, P), dtype=np.float32)
    cases.append((ego, peers, mask))

    # Case 2: One peer hard-braking (rel_x=0.3, accel=-0.8), braking_received=1.0
    ego = np.array([[0.5, 0.0, 0.125, -0.8, 1.0, 0.0,
                     0.5, 0.0, 0.125, -0.8, 1.0, 0.0,
                     0.5, 0.0, 0.125, -0.8, 1.0, 0.0]], dtype=np.float32)
    peers = np.zeros((1, P, F), dtype=np.float32)
    peers[0, 0] = [0.3, 0.0, -0.3, 0.0, -0.8, 0.1]  # close peer, braking hard
    mask = np.zeros((1, P), dtype=np.float32)
    mask[0, 0] = 1.0
    cases.append((ego, peers, mask))

    # Case 3: Max peers (all 8 valid), all peers cruising
    ego = np.array([[0.5, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.5, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.5, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    peers = np.tile(
        np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32),
        (1, P, 1)
    )
    for i in range(P):
        peers[0, i, 0] = 0.1 + i * 0.05  # spread peers along x
    mask = np.ones((1, P), dtype=np.float32)
    cases.append((ego, peers, mask))

    # Case 4: Emergency — ego fast, single peer closing at high speed
    ego = np.array([[0.9, 0.0, 0.125, -0.5, 0.0, 0.8,
                     0.9, 0.0, 0.125, -0.5, 0.0, 0.8,
                     0.9, 0.0, 0.125, -0.5, 0.0, 0.8]], dtype=np.float32)
    peers = np.zeros((1, P, F), dtype=np.float32)
    peers[0, 0] = [0.15, 0.0, -0.5, 0.0, -1.0, 0.0]  # very close, max braking
    mask = np.zeros((1, P), dtype=np.float32)
    mask[0, 0] = 1.0
    cases.append((ego, peers, mask))

    # Case 5: braking_received active but no peers visible
    ego = np.array([[0.6, 0.0, 0.0, 0.0, 0.8, 0.0,
                     0.6, 0.0, 0.0, 0.0, 0.8, 0.0,
                     0.6, 0.0, 0.0, 0.0, 0.8, 0.0]], dtype=np.float32)
    peers = np.zeros((1, P, F), dtype=np.float32)
    mask = np.zeros((1, P), dtype=np.float32)
    cases.append((ego, peers, mask))

    ego_all   = np.concatenate([c[0] for c in cases], axis=0)
    peers_all = np.concatenate([c[1] for c in cases], axis=0)
    mask_all  = np.concatenate([c[2] for c in cases], axis=0)
    return {"ego": ego_all, "peers": peers_all, "peer_mask": mask_all}


def make_random_cases(cfg: dict, n: int, seed: int) -> dict:
    """Random inputs with fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    E, P, F = cfg["ego_dim"], cfg["max_peers"], cfg["peer_feature_dim"]

    ego = rng.uniform(-1.0, 1.0, (n, E)).astype(np.float32)
    # Ego features must stay roughly in [-1, 1]; clamp
    ego = np.clip(ego, -1.0, 1.0)
    # braking_received (idx 4, 10, 16) must be in [0, 1]
    for br_idx in [4, 10, 16]:
        ego[:, br_idx] = rng.uniform(0.0, 1.0, n).astype(np.float32)

    peers = rng.uniform(-1.0, 1.0, (n, P, F)).astype(np.float32)
    peers[:, :, 5] = rng.uniform(0.0, 1.0, (n, P)).astype(np.float32)  # age in [0,1]

    # Random number of valid peers per sample
    n_valid = rng.integers(0, P + 1, n)
    mask = np.zeros((n, P), dtype=np.float32)
    for i, nv in enumerate(n_valid):
        mask[i, :nv] = 1.0

    return {"ego": ego, "peers": peers, "peer_mask": mask}


def load_real_cases(cfg: dict, max_samples: int = 100) -> dict:
    """Load samples from convoy_observations.npz if available."""
    npz_candidates = [
        "ml/data/convoy_analysis/convoy_observations.npz",
        "ml/data/convoy_analysis_extra/convoy_observations.npz",
        "ml/data/convoy_analysis_site/convoy_observations.npz",
    ]
    for path in npz_candidates:
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                print(f"  Loaded real observations from: {path}")
                print(f"  Keys: {list(data.keys())}")
                # These files may not have the exact observation format we need.
                # If they do, extract; otherwise skip.
                if "ego" in data and "peers" in data and "peer_mask" in data:
                    actual_ego_dim = data["ego"].shape[1]
                    if actual_ego_dim != cfg["ego_dim"]:
                        print(f"  Skipping {path}: ego_dim={actual_ego_dim}, need {cfg['ego_dim']} (old recording)")
                        continue
                    n = min(max_samples, data["ego"].shape[0])
                    return {
                        "ego":       data["ego"][:n].astype(np.float32),
                        "peers":     data["peers"][:n].astype(np.float32),
                        "peer_mask": data["peer_mask"][:n].astype(np.float32),
                    }
            except Exception as e:
                print(f"  Skipping {path}: {e}")
    return {}


def generate_golden_vectors(
    model_path: str,
    config_path: str,
    out_path: str,
    seed: int = DEFAULT_SEED,
    n_random: int = N_RANDOM,
) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    print("Loading SB3 model for golden vector generation...")
    model = load_sb3_model(model_path)

    # ---------------------------------------------------------------------- #
    # Collect input batches                                                   #
    # ---------------------------------------------------------------------- #
    batches = []
    labels  = []

    synth = make_synthetic_cases(cfg)
    batches.append(synth)
    labels += [f"synthetic_{i}" for i in range(synth["ego"].shape[0])]
    print(f"  Synthetic cases: {synth['ego'].shape[0]}")

    rand = make_random_cases(cfg, n_random, seed)
    batches.append(rand)
    labels += [f"random_{i}" for i in range(n_random)]
    print(f"  Random cases: {n_random}")

    real = load_real_cases(cfg)
    if real:
        batches.append(real)
        labels += [f"real_{i}" for i in range(real["ego"].shape[0])]
        print(f"  Real recording cases: {real['ego'].shape[0]}")

    # Merge all
    all_ego   = np.concatenate([b["ego"]       for b in batches], axis=0)
    all_peers = np.concatenate([b["peers"]      for b in batches], axis=0)
    all_mask  = np.concatenate([b["peer_mask"]  for b in batches], axis=0)

    print(f"\nTotal golden vectors: {all_ego.shape[0]}")

    # ---------------------------------------------------------------------- #
    # Run SB3 model (ground truth)                                           #
    # ---------------------------------------------------------------------- #
    print("Running SB3 model for reference outputs...")
    obs = {"ego": all_ego, "peers": all_peers, "peer_mask": all_mask}
    ref_actions = predict_sb3(model, obs)
    print(f"  Reference actions: min={ref_actions.min():.4f}  max={ref_actions.max():.4f}  "
          f"mean={ref_actions.mean():.4f}")

    # ---------------------------------------------------------------------- #
    # Save                                                                    #
    # ---------------------------------------------------------------------- #
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        ego=all_ego,
        peers=all_peers,
        peer_mask=all_mask,
        ref_actions=ref_actions,
        labels=np.array(labels),
    )
    print(f"\nGolden vectors saved: {out_path}")
    print(f"  ego:        {all_ego.shape}")
    print(f"  peers:      {all_peers.shape}")
    print(f"  peer_mask:  {all_mask.shape}")
    print(f"  ref_actions:{ref_actions.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--config",  default=DEFAULT_CONFIG)
    parser.add_argument("--out",     default=DEFAULT_OUT)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_rand",  type=int, default=N_RANDOM)
    args = parser.parse_args()
    generate_golden_vectors(args.model, args.config, args.out, args.seed, args.n_rand)


if __name__ == "__main__":
    main()
