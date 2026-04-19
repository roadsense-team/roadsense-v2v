"""
Step 1: Export weights from SB3 PyTorch model to numpy .npz + JSON manifest.

Reads: ml/results/run_025_replay_v1/model_final.zip  (SB3 zip)
Writes:
  ml/deployment/artifacts/weights.npz          (all layer weights)
  ml/deployment/artifacts/model_config.json    (architecture config)

Run from repo root (inside Docker):
  python -m ml.deployment.step1_export_weights
"""
import argparse
import json
import os
import sys
import zipfile
import io

import numpy as np
import torch


DEFAULT_MODEL_PATH = "ml/results/run_025_replay_v1/model_final.zip"
DEFAULT_OUT_DIR = "ml/deployment/artifacts"


def load_sb3_policy_state(model_zip_path: str) -> dict:
    """Extract policy.pth state_dict from an SB3 zip."""
    with zipfile.ZipFile(model_zip_path, "r") as z:
        with z.open("policy.pth") as f:
            state = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    return state


def load_sb3_model_config(model_zip_path: str) -> dict:
    """Extract the JSON data blob from an SB3 zip."""
    with zipfile.ZipFile(model_zip_path, "r") as z:
        with z.open("data") as f:
            return json.load(f)


def export_weights(model_zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {model_zip_path}")
    state = load_sb3_policy_state(model_zip_path)

    print("All keys in policy.pth:")
    for k, v in state.items():
        shape = v.shape if hasattr(v, "shape") else type(v)
        print(f"  {k:60s}  {shape}")

    # ------------------------------------------------------------------ #
    # Extract the layers we need for inference (actor path only).         #
    # SB3 MultiInputActorCriticPolicy with DeepSetExtractor lays out as:  #
    #   features_extractor.peer_encoder.0.*   -> peer MLP layer 1         #
    #   features_extractor.peer_encoder.2.*   -> peer MLP layer 2         #
    #   mlp_extractor.policy_net.0.*          -> actor MLP layer 1        #
    #   mlp_extractor.policy_net.2.*          -> actor MLP layer 2        #
    #   action_net.*                          -> output linear             #
    # ------------------------------------------------------------------ #

    required = {
        "peer_enc_w1":  "features_extractor.peer_encoder.0.weight",   # (32, 6)
        "peer_enc_b1":  "features_extractor.peer_encoder.0.bias",     # (32,)
        "peer_enc_w2":  "features_extractor.peer_encoder.2.weight",   # (32, 32)
        "peer_enc_b2":  "features_extractor.peer_encoder.2.bias",     # (32,)
        "policy_w1":    "mlp_extractor.policy_net.0.weight",          # (64, 50)
        "policy_b1":    "mlp_extractor.policy_net.0.bias",            # (64,)
        "policy_w2":    "mlp_extractor.policy_net.2.weight",          # (64, 64)
        "policy_b2":    "mlp_extractor.policy_net.2.bias",            # (64,)
        "action_w":     "action_net.weight",                          # (1, 64)
        "action_b":     "action_net.bias",                            # (1,)
    }

    weights = {}
    missing = []
    for friendly, key in required.items():
        if key in state:
            weights[friendly] = state[key].numpy().astype(np.float32)
            print(f"  Exported {friendly:20s} from {key}  shape={weights[friendly].shape}")
        else:
            missing.append(key)

    if missing:
        print("\nERROR: The following keys were not found in policy.pth:")
        for k in missing:
            print(f"  {k}")
        print("\nAll available keys:")
        for k in state.keys():
            print(f"  {k}")
        sys.exit(1)

    npz_path = os.path.join(out_dir, "weights.npz")
    np.savez(npz_path, **weights)
    print(f"\nWeights saved: {npz_path}")

    # ------------------------------------------------------------------ #
    # Write architecture config so downstream scripts are self-contained  #
    # ------------------------------------------------------------------ #
    config = {
        "model_zip": model_zip_path,
        "ego_dim": int(weights["policy_w1"].shape[1] - weights["peer_enc_w2"].shape[0]),
        "peer_feature_dim": int(weights["peer_enc_w1"].shape[1]),
        "embed_dim": int(weights["peer_enc_w2"].shape[0]),
        "max_peers": 8,
        "policy_hidden_1": int(weights["policy_w1"].shape[0]),
        "policy_hidden_2": int(weights["policy_w2"].shape[0]),
        "action_dim": int(weights["action_w"].shape[0]),
        "action_low": 0.0,
        "action_high": 1.0,
        "peer_enc_activation": "relu",
        "policy_activation": "tanh",
    }
    cfg_path = os.path.join(out_dir, "model_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved:  {cfg_path}")
    print(f"\nArchitecture summary:")
    print(f"  Peer encoder:  ({config['peer_feature_dim']}) → {config['embed_dim']} → {config['embed_dim']}  [ReLU]")
    print(f"  Max-pool over {config['max_peers']} peers → ({config['embed_dim']},)")
    print(f"  Ego input:     ({config['ego_dim']},)")
    print(f"  Concat:        ({config['embed_dim'] + config['ego_dim']},)")
    print(f"  Policy MLP:    → {config['policy_hidden_1']} → {config['policy_hidden_2']} → {config['action_dim']}  [Tanh, clip to [0,1]]")


def main():
    parser = argparse.ArgumentParser(description="Export SB3 model weights to numpy")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    export_weights(args.model, args.out_dir)


if __name__ == "__main__":
    main()
