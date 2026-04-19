"""
Step 5: Convert the float Keras model to INT8 TFLite.

Two outputs:
  1. Float32 TFLite  (sanity check against Keras)
  2. INT8 quantized TFLite  (for ESP32)

Reads:
  ml/deployment/artifacts/model_float.keras
  ml/deployment/artifacts/golden_vectors.npz  (used as representative dataset)

Writes:
  ml/deployment/artifacts/model_float.tflite
  ml/deployment/artifacts/model_int8.tflite

INT8 quantization notes:
  - Full integer quantization: all ops quantized to INT8
  - Input/output tensors kept as FLOAT32 (easier ESP32 integration)
  - Representative dataset: 200+ samples from golden_vectors.npz
  - This is Post-Training Quantization (PTQ), no retraining needed

Run from repo root (inside Docker):
  python -m ml.deployment.step5_quantize_tflite
"""
import argparse
import os
from typing import Generator

import numpy as np
import tensorflow as tf


DEFAULT_SAVEDMODEL  = "ml/deployment/artifacts/model_float_savedmodel"
DEFAULT_WEIGHTS     = "ml/deployment/artifacts/weights.npz"
DEFAULT_CONFIG      = "ml/deployment/artifacts/model_config.json"
DEFAULT_GOLDEN      = "ml/deployment/artifacts/golden_vectors.npz"
DEFAULT_OUT_FLOAT   = "ml/deployment/artifacts/model_float.tflite"
DEFAULT_OUT_INT8    = "ml/deployment/artifacts/model_int8.tflite"


def _build_model_in_memory(
    weights_npz: str = DEFAULT_WEIGHTS,
    config_path: str = DEFAULT_CONFIG,
) -> "tf.keras.Model":
    """
    Build the Keras model and load weights fully in-process.

    Used for INT8 quantization: from_keras_model() freezes ResourceVariables
    into Const nodes before handing to the converter, avoiding the
    READ_VARIABLE error that occurs when converting from a SavedModel during
    calibrated quantization.
    """
    import json
    from ml.deployment.step2_keras_model import build_deepset_policy

    with open(config_path) as f:
        cfg = json.load(f)

    model = build_deepset_policy(
        ego_dim=cfg["ego_dim"],
        peer_feature_dim=cfg["peer_feature_dim"],
        embed_dim=cfg["embed_dim"],
        max_peers=cfg["max_peers"],
        policy_hidden_1=cfg["policy_hidden_1"],
        policy_hidden_2=cfg["policy_hidden_2"],
        action_dim=cfg["action_dim"],
    )

    w = np.load(weights_npz)
    for layer in model.layers:
        name = layer.name
        if name == "td_enc1":
            layer.layer.set_weights([w["peer_enc_w1"].T, w["peer_enc_b1"]])
        elif name == "td_enc2":
            layer.layer.set_weights([w["peer_enc_w2"].T, w["peer_enc_b2"]])
        elif name == "policy_dense1":
            layer.set_weights([w["policy_w1"].T, w["policy_b1"]])
        elif name == "policy_dense2":
            layer.set_weights([w["policy_w2"].T, w["policy_b2"]])
        elif name == "action_out":
            layer.set_weights([w["action_w"].T, w["action_b"]])
    return model


def _input_order_from_tflite(float_tflite_path: str) -> list:
    """
    Read the float TFLite's input details and return the correct feed order
    as a list of keys ('ego', 'peers', 'peer_mask'), matched by shape.

    Expected shapes (batch=1):
      ego       → [1, 18]
      peers     → [1, 8, 6]
      peer_mask → [1, 8]
    """
    with open(float_tflite_path, "rb") as f:
        content = f.read()
    interp = tf.lite.Interpreter(model_content=content)
    interp.allocate_tensors()

    shape_to_key = {
        (1, 18):    "ego",
        (1, 8, 6):  "peers",
        (1, 8):     "peer_mask",
    }
    order = []
    for det in interp.get_input_details():
        key = shape_to_key.get(tuple(det["shape"].tolist()))
        if key is None:
            raise ValueError(
                f"Unknown input shape {det['shape'].tolist()} in float TFLite. "
                f"Expected one of {list(shape_to_key.keys())}"
            )
        order.append(key)
        print(f"  TFLite input[{det['index']}] shape={det['shape'].tolist()} → '{key}'")
    return order


def _input_order_from_concrete(concrete_func) -> list:
    """Derive input feed order by converting the ConcreteFunction to a float
    TFLite temporarily and inspecting its input tensor shapes. This matches the
    exact graph that the converter/calibrator will see when quantizing.
    """
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = conv.convert()
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    shape_to_key = {
        (1, 18):    "ego",
        (1, 8, 6):  "peers",
        (1, 8):     "peer_mask",
    }
    order = []
    for det in interp.get_input_details():
        key = shape_to_key.get(tuple(det["shape"].tolist()))
        if key is None:
            raise ValueError(
                f"Unknown input shape {det['shape'].tolist()} in concrete TFLite. "
                f"Expected one of {list(shape_to_key.keys())}"
            )
        order.append(key)
        print(f"  TFLite input[{det['index']}] shape={det['shape'].tolist()} → '{key}'")
    return order


def make_representative_dataset(golden_npz: str,
                                 float_tflite_path: str = DEFAULT_OUT_FLOAT) -> Generator:
    """
    Yield calibration samples for INT8 quantization.
    Reads input order from the already-converted float TFLite so the feed
    order always matches the model's actual input indices.
    """
    order = _input_order_from_tflite(float_tflite_path)
    data  = np.load(golden_npz)
    n     = data["ego"].shape[0]
    for i in range(n):
        yield [data[key][i:i+1].astype(np.float32) for key in order]


def convert_float_tflite(savedmodel_path: str, out_path: str) -> bytes:
    """Convert to float32 TFLite (no quantization)."""
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Float TFLite saved: {out_path}  ({size_kb:.1f} KB)")
    return tflite_model


def convert_int8_tflite(
    savedmodel_path: str,
    golden_npz: str,
    out_path: str,
    weights_npz: str = DEFAULT_WEIGHTS,
    config_path: str = DEFAULT_CONFIG,
) -> bytes:
    """
    Convert to INT8 TFLite via ConcreteFunction.

    Why this approach:
      - from_saved_model  → READ_VARIABLE error during calibration (mutable vars)
      - from_keras_model  → calls Keras2 internal _get_save_spec, absent in Keras3
      - from_concrete_functions → traces the model, freezing all variables into
        Const nodes before the converter/calibrator ever sees them. Works with
        any Keras version and avoids all SavedModel variable issues.

    Input/output kept as FLOAT32 (no scale/zero-point handling needed on ESP32).
    If float I/O is unsupported, a JSON sidecar documents the quant params.
    """
    import json as _json

    print("  Building model in-process and tracing to ConcreteFunction...")
    model = _build_model_in_memory(weights_npz, config_path)

    # Trace with explicit input signatures — this freezes all variable values
    # into the graph as constants, making the result safe for TFLite calibration.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 18],    dtype=tf.float32, name="ego"),
        tf.TensorSpec(shape=[1, 8, 6],  dtype=tf.float32, name="peers"),
        tf.TensorSpec(shape=[1, 8],     dtype=tf.float32, name="peer_mask"),
    ])
    def _infer(ego, peers, peer_mask):
        return model({"ego": ego, "peers": peers, "peer_mask": peer_mask},
                     training=False)

    concrete_func = _infer.get_concrete_function()

    # Freeze variables into constants to avoid READ_VARIABLE ops during
    # calibration/quantization.
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )
        const_func = convert_variables_to_constants_v2(concrete_func)
        frozen_funcs = [const_func]
    except Exception as e:  # Fallback if internal API path changes
        print(f"[warn] convert_variables_to_constants_v2 failed: {e}. Proceeding without explicit freezing.")
        frozen_funcs = [concrete_func]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        frozen_funcs, trackable_obj=model
    )

    # Quick sanity check for representative dataset shapes
    def _assert_golden_shapes(path: str) -> None:
        arr = np.load(path)
        s_ego = tuple(arr["ego"].shape[1:])
        s_peers = tuple(arr["peers"].shape[1:])
        s_mask = tuple(arr["peer_mask"].shape[1:])
        if s_ego != (18,) or s_peers != (8, 6) or s_mask != (8,):
            raise ValueError(
                f"golden_vectors.npz has unexpected shapes: ego={s_ego}, peers={s_peers}, peer_mask={s_mask}. "
                f"Expected ego=(18,), peers=(8,6), peer_mask=(8,)."
            )

    _assert_golden_shapes(golden_npz)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Prefer name-based feeding to avoid any ambiguity.
    # Yield a dict keyed by signature input names used in the tf.function.
    def _rep_dataset():
        arr = np.load(golden_npz)
        n = arr["ego"].shape[0]
        for i in range(n):
            yield {
                "ego":       arr["ego"][i:i+1].astype(np.float32),
                "peers":     arr["peers"][i:i+1].astype(np.float32),
                "peer_mask": arr["peer_mask"][i:i+1].astype(np.float32),
            }

    converter.representative_dataset = _rep_dataset
    # Use new quantizer if available to improve calibration robustness.
    if hasattr(converter, "experimental_new_quantizer"):
        converter.experimental_new_quantizer = True
    # Disable resource variable handling to prefer frozen Consts
    if hasattr(converter, "experimental_enable_resource_variables"):
        converter.experimental_enable_resource_variables = False
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,      # fallback for any op that can't be INT8
    ]
    # Do NOT set inference_input_type / inference_output_type explicitly.
    # With DEFAULT optimizations + representative dataset, TFLite defaults to
    # float32 I/O automatically, avoiding dequantize-op insertion that causes
    # the RESHAPE shape-mismatch bug (18 != 8).
    tflite_model = converter.convert()
    io_type = "float32 I/O (INT8 ops)"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"INT8 TFLite saved: {out_path}  ({size_kb:.1f} KB)  [{io_type}]")
    return tflite_model


def _save_quant_params(tflite_model: bytes, tflite_path: str) -> None:
    """Save input/output quantization params to a sidecar JSON."""
    import json
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    info = {
        "inputs": [],
        "outputs": [],
    }
    for t in interp.get_input_details():
        info["inputs"].append({
            "name":       t["name"],
            "dtype":      str(t["dtype"]),
            "shape":      t["shape"].tolist(),
            "scale":      float(t["quantization"][0]),
            "zero_point": int(t["quantization"][1]),
        })
    for t in interp.get_output_details():
        info["outputs"].append({
            "name":       t["name"],
            "dtype":      str(t["dtype"]),
            "shape":      t["shape"].tolist(),
            "scale":      float(t["quantization"][0]),
            "zero_point": int(t["quantization"][1]),
        })
    json_path = tflite_path.replace(".tflite", "_quant_params.json")
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Quantization params: {json_path}")


def print_tensor_details(tflite_path: str) -> None:
    """Print all tensor names, shapes, and quantization params."""
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    print(f"\nTensor details for: {tflite_path}")
    print("  INPUTS:")
    for t in interp.get_input_details():
        print(f"    [{t['index']}] {t['name']:40s}  {t['dtype']}  {t['shape'].tolist()}"
              f"  scale={t['quantization'][0]:.6f}  zp={t['quantization'][1]}")
    print("  OUTPUTS:")
    for t in interp.get_output_details():
        print(f"    [{t['index']}] {t['name']:40s}  {t['dtype']}  {t['shape'].tolist()}"
              f"  scale={t['quantization'][0]:.6f}  zp={t['quantization'][1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_SAVEDMODEL)
    parser.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    parser.add_argument("--config",     default=DEFAULT_CONFIG)
    parser.add_argument("--golden",     default=DEFAULT_GOLDEN)
    parser.add_argument("--out_float",  default=DEFAULT_OUT_FLOAT)
    parser.add_argument("--out_int8",   default=DEFAULT_OUT_INT8)
    args = parser.parse_args()

    print("=== Converting to Float TFLite ===")
    convert_float_tflite(args.model, args.out_float)
    print_tensor_details(args.out_float)

    print("\n=== Converting to INT8 TFLite ===")
    convert_int8_tflite(args.model, args.golden, args.out_int8,
                        weights_npz=args.weights, config_path=args.config)
    print_tensor_details(args.out_int8)


if __name__ == "__main__":
    main()
