"""
Step 8: Generate a golden-vector C test file for ESP32.

This creates a self-contained C++ file that encodes the INT8 model's
golden vectors as C arrays.  The ESP32 runs through each vector,
calls ModelRunner::infer(), and checks the output is within tolerance.

Reads:
  ml/deployment/artifacts/golden_vectors.npz
  ml/deployment/artifacts/model_int8_dr.tflite  (dynamic-range, validated by step6)

Writes:
  hardware/test/test_model_runner/golden_vectors.h

Run from repo root:
  python -m ml.deployment.step8_esp32_golden_test
"""
import argparse
import os
import numpy as np
import tensorflow as tf

DEFAULT_GOLDEN  = "ml/deployment/artifacts/golden_vectors.npz"
DEFAULT_INT8    = "ml/deployment/artifacts/model_int8_dr.tflite"
DEFAULT_OUT     = "hardware/test/test_model_runner/golden_vectors.h"

# Tolerance for ESP32 hardware test
ESP32_TOLERANCE = 0.05   # |esp32_out - int8_tflite_expected| < 0.05


def run_tflite_batch(tflite_path: str, inputs: dict) -> np.ndarray:
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    # Match inputs by shape — robust to any index ordering across model variants.
    shape_to_key = {(1, 18): "ego", (1, 8): "peer_mask", (1, 8, 6): "peers"}
    input_map = {det["index"]: shape_to_key[tuple(det["shape"].tolist())] for det in inp}

    results = []
    n = inputs["ego"].shape[0]
    for i in range(n):
        for idx, key in input_map.items():
            interp.set_tensor(idx, inputs[key][i:i+1])
        interp.invoke()
        results.append(interp.get_tensor(out[0]["index"]).copy().reshape(1, 1))
    return np.concatenate(results).astype(np.float32)


def to_c_float_array(arr: np.ndarray, name: str) -> str:
    """Emit a C array declaration with correct multi-dimensional brackets."""
    flat = arr.flatten()
    vals = ", ".join(f"{v:.8f}f" for v in flat)
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const float {name}{dims} = {{{vals}}};"


def generate_test_header(golden_npz: str, int8_path: str, out_path: str) -> None:
    data = np.load(golden_npz, allow_pickle=True)
    inputs = {
        "ego":       data["ego"].astype(np.float32),
        "peers":     data["peers"].astype(np.float32),
        "peer_mask": data["peer_mask"].astype(np.float32),
    }
    labels = data.get("labels", [str(i) for i in range(inputs["ego"].shape[0])])

    print(f"Running INT8 TFLite to get expected outputs for {inputs['ego'].shape[0]} samples...")
    expected = run_tflite_batch(int8_path, inputs)

    # Take only the first N samples (synthetic + some random)
    N = min(50, inputs["ego"].shape[0])
    ego_n   = inputs["ego"][:N]
    peers_n = inputs["peers"][:N]
    mask_n  = inputs["peer_mask"][:N]
    exp_n   = expected[:N].reshape(N)   # 1-D: shape (N,)

    header = f"""// GENERATED FILE — DO NOT EDIT
// Source: ml/deployment/step8_esp32_golden_test.py
// Contains {N} test vectors for on-device validation.
// Tolerance: |output - expected| < {ESP32_TOLERANCE}f

#pragma once

static constexpr int kTestVectors    = {N};
static constexpr float kTolerance    = {ESP32_TOLERANCE}f;
// kEgoDim / kMaxPeers / kPeerFeatureDim defined in ModelRunner.h

// ego[kTestVectors][kEgoDim]
{to_c_float_array(ego_n, 'kTestEgo')}

// peers[kTestVectors][kMaxPeers][kPeerFeatureDim]
{to_c_float_array(peers_n, 'kTestPeers')}

// peer_mask[kTestVectors][kMaxPeers]
{to_c_float_array(mask_n, 'kTestMask')}

// expected_action[kTestVectors]
{to_c_float_array(exp_n, 'kTestExpected')}
"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header)
    print(f"Golden test header written: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--int8",   default=DEFAULT_INT8)
    parser.add_argument("--out",    default=DEFAULT_OUT)
    args = parser.parse_args()
    generate_test_header(args.golden, args.int8, args.out)


if __name__ == "__main__":
    main()
