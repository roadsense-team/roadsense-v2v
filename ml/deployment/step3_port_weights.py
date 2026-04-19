"""
Step 3: Load exported numpy weights into the Keras model and verify.

Reads:
  ml/deployment/artifacts/weights.npz       (from step 1)
  ml/deployment/artifacts/model_config.json  (from step 1)

Writes:
  ml/deployment/artifacts/model_float.keras  (saved Keras model)

Weight layout differences between PyTorch and Keras/TensorFlow:
  PyTorch Linear: weight shape is (out_features, in_features)  →  needs transpose
  Keras Dense:    kernel shape is (in_features, out_features)

Run from repo root (inside Docker):
  python -m ml.deployment.step3_port_weights
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from ml.deployment.step2_keras_model import build_deepset_policy, load_config


DEFAULT_WEIGHTS  = "ml/deployment/artifacts/weights.npz"
DEFAULT_CONFIG   = "ml/deployment/artifacts/model_config.json"
DEFAULT_OUT      = "ml/deployment/artifacts/model_float_savedmodel"


def port_weights(weights_npz: str, config_path: str, out_path: str) -> tf.keras.Model:
    cfg = load_config(config_path)
    w   = np.load(weights_npz)

    # ---------------------------------------------------------------------- #
    # Build the Keras model                                                   #
    # ---------------------------------------------------------------------- #
    model = build_deepset_policy(
        ego_dim=cfg["ego_dim"],
        peer_feature_dim=cfg["peer_feature_dim"],
        embed_dim=cfg["embed_dim"],
        max_peers=cfg["max_peers"],
        policy_hidden_1=cfg["policy_hidden_1"],
        policy_hidden_2=cfg["policy_hidden_2"],
        action_dim=cfg["action_dim"],
    )

    # ---------------------------------------------------------------------- #
    # Map numpy arrays to Keras layer weights.                               #
    # PyTorch Linear.weight is (out, in) → must transpose to (in, out).      #
    # ---------------------------------------------------------------------- #
    assignments = {
        # TimeDistributed wraps Dense; access the inner layer
        "td_enc1":             (w["peer_enc_w1"].T, w["peer_enc_b1"]),  # kernel (6,32), bias (32,)
        "td_enc2":             (w["peer_enc_w2"].T, w["peer_enc_b2"]),  # kernel (32,32), bias (32,)
        "policy_dense1":       (w["policy_w1"].T,   w["policy_b1"]),    # kernel (50,64), bias (64,)
        "policy_dense2":       (w["policy_w2"].T,   w["policy_b2"]),    # kernel (64,64), bias (64,)
        "action_out":          (w["action_w"].T,    w["action_b"]),     # kernel (64,1), bias (1,)
    }

    # ---------------------------------------------------------------------- #
    # Apply weights layer by layer                                           #
    # ---------------------------------------------------------------------- #
    for layer in model.layers:
        lname = layer.name

        # TimeDistributed: inner layer is accessed via layer.layer
        if lname in ("td_enc1", "td_enc2"):
            inner = layer.layer  # the Dense inside TimeDistributed
            kernel, bias = assignments[lname]
            inner.set_weights([kernel, bias])
            print(f"  Set {lname} (TimeDistributed):  kernel={kernel.shape}  bias={bias.shape}")

        elif lname in ("policy_dense1", "policy_dense2", "action_out"):
            kernel, bias = assignments[lname]
            layer.set_weights([kernel, bias])
            print(f"  Set {lname}:  kernel={kernel.shape}  bias={bias.shape}")

    # ---------------------------------------------------------------------- #
    # Shape sanity checks                                                     #
    # ---------------------------------------------------------------------- #
    _assert_shapes(model, cfg)

    # ---------------------------------------------------------------------- #
    # Save                                                                    #
    # ---------------------------------------------------------------------- #
    # Save as TF SavedModel (directory).  This format serializes the concrete
    # computation graph — no custom layer class registration needed at load time.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tf.saved_model.save(model, out_path)
    print(f"\nSavedModel written: {out_path}/")
    return model


def _assert_shapes(model: tf.keras.Model, cfg: dict) -> None:
    """Quick sanity check: run a zero input through the model."""
    batch = 2
    dummy = {
        "ego":       np.zeros((batch, cfg["ego_dim"]), dtype=np.float32),
        "peers":     np.zeros((batch, cfg["max_peers"], cfg["peer_feature_dim"]), dtype=np.float32),
        "peer_mask": np.zeros((batch, cfg["max_peers"]), dtype=np.float32),
    }
    out = model.predict(dummy, verbose=0)
    assert out.shape == (batch, cfg["action_dim"]), f"Unexpected output shape: {out.shape}"
    print(f"  Shape check passed: input OK → output {out.shape}")

    # With all zeros (no peers, ego=0) output should be in [0, 1]
    assert np.all(out >= 0.0) and np.all(out <= 1.0), f"Output out of [0,1]: {out}"
    print(f"  Range check passed: output in [0, 1]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default=DEFAULT_WEIGHTS)
    parser.add_argument("--config",   default=DEFAULT_CONFIG)
    parser.add_argument("--out",      default=DEFAULT_OUT)
    args = parser.parse_args()
    port_weights(args.weights, args.config, args.out)


if __name__ == "__main__":
    main()
