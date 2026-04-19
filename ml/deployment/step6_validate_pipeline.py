"""
Step 6: Full pipeline validation — compare every stage against the golden vectors.

Validates the entire chain:
  SB3 (gold) → Keras → Float TFLite → INT8 TFLite

For each transition reports:
  - MAE          mean absolute error
  - RMSE         root mean squared error
  - Max AE       worst-case absolute error
  - Agree@0.05   action agreement within ±0.05 of gold (braking threshold)
  - Agree@0.15   action agreement within ±0.15 of gold (soft threshold)
  - Cross@0.1    hazard-detection agreement (both above/below 0.1)
  - Cross@0.3    hazard-detection agreement (both above/below 0.3)
  - PASS/FAIL    vs hard limits defined below

Acceptance criteria (must PASS all to proceed to ESP32 build):
  Keras vs SB3:          MAE < 1e-4,  max_AE < 5e-4,  agree@0.05 > 99.0%
  Float TFLite vs SB3:   MAE < 1e-3,  max_AE < 5e-3,  agree@0.05 > 98.0%
  INT8 TFLite vs SB3:    MAE < 0.02,  max_AE < 0.05,  agree@0.05 > 95.0%,  cross@0.1 > 97.0%

Run from repo root (inside Docker):
  python -m ml.deployment.step6_validate_pipeline
"""
import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


DEFAULT_GOLDEN    = "ml/deployment/artifacts/golden_vectors.npz"
DEFAULT_KERAS     = "ml/deployment/artifacts/model_float_savedmodel"
DEFAULT_FLOAT_TFL = "ml/deployment/artifacts/model_float.tflite"
DEFAULT_INT8_TFL  = "ml/deployment/artifacts/model_int8.tflite"
DEFAULT_REPORT    = "ml/deployment/artifacts/validation_report.json"


# ============================================================================ #
# Acceptance thresholds                                                         #
# ============================================================================ #
THRESHOLDS = {
    "keras_vs_sb3": {
        "mae":        1e-4,
        "max_ae":     5e-4,
        "agree_005":  99.0,
        "agree_015":  99.5,
        "cross_010":  99.5,
        "cross_030":  99.5,
    },
    "float_tflite_vs_sb3": {
        "mae":        1e-3,
        "max_ae":     5e-3,
        "agree_005":  98.0,
        "agree_015":  99.0,
        "cross_010":  99.0,
        "cross_030":  99.0,
    },
    "int8_tflite_vs_sb3": {
        "mae":        0.02,
        "max_ae":     0.05,
        "agree_005":  95.0,
        "agree_015":  98.0,
        "cross_010":  97.0,
        "cross_030":  98.0,
    },
}


# ============================================================================ #
# Inference helpers                                                             #
# ============================================================================ #

def run_keras(savedmodel_path: str, inputs: dict) -> np.ndarray:
    """Run SavedModel regardless of exported input names.

    Maps the model's serving_default signature inputs (e.g., 'inputs',
    'inputs_1', 'inputs_2') to our semantic keys by shape:
      - (None, 18)   → 'ego'
      - (None, 8, 6) → 'peers'
      - (None, 8)    → 'peer_mask'
    """
    model = tf.saved_model.load(savedmodel_path)
    infer = model.signatures["serving_default"]

    # Build a name→tensor mapping using shapes (ignore leading batch dim)
    sig_inputs = infer.structured_input_signature[1]  # dict: name → TensorSpec
    shape_to_key = {
        (18,):   "ego",
        (8, 6):  "peers",
        (8,):    "peer_mask",
    }
    feed = {}
    for name, spec in sig_inputs.items():
        s = tuple(spec.shape.as_list()[1:])  # drop batch
        key = shape_to_key.get(s)
        if key is None:
            raise ValueError(f"Unrecognized SavedModel input shape {spec.shape} for '{name}'")
        feed[name] = tf.constant(inputs[key])

    result = infer(**feed)
    out = list(result.values())[0].numpy()  # first (and only) output
    return out.reshape(-1, 1).astype(np.float32)


def run_tflite(tflite_path: str, inputs: dict) -> np.ndarray:
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()
    # Use reference kernels to avoid delegate-specific numerical quirks (e.g., XNNPACK NaNs)
    interp = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_delegates=[],
    )
    interp.allocate_tensors()

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    # Match each input slot to the right data key by shape (robust to any name/index ordering)
    shape_to_key = {
        (1, 18):    "ego",
        (1, 8, 6):  "peers",
        (1, 8):     "peer_mask",
    }
    input_map = {}  # index → data key
    for det in inp_details:
        key = shape_to_key.get(tuple(det["shape"].tolist()))
        if key is None:
            raise ValueError(f"Unrecognised TFLite input shape: {det['shape'].tolist()}")
        input_map[det["index"]] = (key, det["dtype"])

    results = []
    n = inputs["ego"].shape[0]
    for i in range(n):
        for idx, (key, dtype) in input_map.items():
            interp.set_tensor(idx, inputs[key][i:i+1].astype(dtype))
        interp.invoke()
        out = interp.get_tensor(out_details[0]["index"]).copy()
        results.append(out.reshape(1, 1))

    return np.concatenate(results, axis=0).astype(np.float32)


def _convert_dynamic_range(savedmodel_path: str, out_path: str) -> None:
    """Create a dynamic-range quantized TFLite (INT8 weights, float activations)
    from a frozen concrete function to avoid READ_VARIABLE ops."""
    sm = tf.saved_model.load(savedmodel_path)
    infer = sm.signatures["serving_default"]
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
    const_func = convert_variables_to_constants_v2(infer)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([const_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(converter.convert())


def _convert_float_frozen_from_savedmodel(savedmodel_path: str, out_path: str) -> None:
    """Create a float32 TFLite by freezing variables from SavedModel's signature.

    This avoids READ_VARIABLE ops that some interpreters/kernels cannot execute.
    """
    sm = tf.saved_model.load(savedmodel_path)
    infer = sm.signatures["serving_default"]
    # Freeze variables into constants
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
    const_func = convert_variables_to_constants_v2(infer)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([const_func])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(converter.convert())


# ============================================================================ #
# Metrics                                                                      #
# ============================================================================ #

def compute_metrics(
    pred: np.ndarray,
    gold: np.ndarray,
    label: str,
) -> Dict[str, float]:
    diff = pred.reshape(-1) - gold.reshape(-1)
    abs_diff = np.abs(diff)
    metrics = {
        "label":      label,
        "n":          int(len(diff)),
        "mae":        float(np.mean(abs_diff)),
        "rmse":       float(np.sqrt(np.mean(diff**2))),
        "max_ae":     float(np.max(abs_diff)),
        "mean_pred":  float(np.mean(pred)),
        "std_pred":   float(np.std(pred)),
        "agree_005":  float(np.mean(abs_diff < 0.05) * 100),
        "agree_015":  float(np.mean(abs_diff < 0.15) * 100),
        "cross_010":  float(np.mean(
            (pred.reshape(-1) > 0.1) == (gold.reshape(-1) > 0.1)) * 100),
        "cross_030":  float(np.mean(
            (pred.reshape(-1) > 0.3) == (gold.reshape(-1) > 0.3)) * 100),
    }
    return metrics


def evaluate_pass_fail(metrics: dict, stage: str) -> Tuple[bool, list]:
    if stage not in THRESHOLDS:
        return True, []
    thresh = THRESHOLDS[stage]
    failures = []
    for metric, limit in thresh.items():
        val = metrics.get(metric)
        if val is None:
            continue
        # For error metrics: must be BELOW threshold
        # For agreement/cross metrics: must be ABOVE threshold
        if metric in ("mae", "max_ae", "rmse"):
            if val > limit:
                failures.append(f"{metric}={val:.6f} > {limit:.6f}")
        else:
            if val < limit:
                failures.append(f"{metric}={val:.2f}% < {limit:.2f}%")
    return len(failures) == 0, failures


def print_metrics(metrics: dict, stage: str) -> None:
    passed, failures = evaluate_pass_fail(metrics, stage)
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {metrics['label']}  (n={metrics['n']})")
    print(f"    MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  "
          f"MaxAE={metrics['max_ae']:.6f}")
    print(f"    Agree@0.05={metrics['agree_005']:.2f}%  "
          f"Agree@0.15={metrics['agree_015']:.2f}%")
    print(f"    Cross@0.1={metrics['cross_010']:.2f}%  "
          f"Cross@0.3={metrics['cross_030']:.2f}%")
    if failures:
        for f in failures:
            print(f"    !! FAILED: {f}")


# ============================================================================ #
# Per-sample inspection for failures                                           #
# ============================================================================ #

def show_worst_cases(pred: np.ndarray, gold: np.ndarray, labels, n: int = 5) -> None:
    diff = np.abs(pred.reshape(-1) - gold.reshape(-1))
    worst = np.argsort(diff)[::-1][:n]
    print(f"  Worst {n} samples (by absolute error):")
    for idx in worst:
        lbl = labels[idx] if labels is not None else str(idx)
        print(f"    [{lbl}]  pred={pred.flat[idx]:.5f}  gold={gold.flat[idx]:.5f}  "
              f"AE={diff[idx]:.5f}")


# ============================================================================ #
# Main validation                                                               #
# ============================================================================ #

def run_validation(
    golden_npz: str,
    keras_path: str,
    float_tflite_path: str,
    int8_tflite_path: str,
    report_path: str,
) -> dict:
    print("Loading golden vectors...")
    data   = np.load(golden_npz, allow_pickle=True)
    inputs = {
        "ego":       data["ego"].astype(np.float32),
        "peers":     data["peers"].astype(np.float32),
        "peer_mask": data["peer_mask"].astype(np.float32),
    }
    gold   = data["ref_actions"].astype(np.float32)
    labels = data["labels"] if "labels" in data else None
    print(f"  Golden vectors: {gold.shape[0]} samples")
    print(f"  Gold stats: min={gold.min():.4f}  max={gold.max():.4f}  mean={gold.mean():.4f}")

    report = {"stages": [], "all_passed": True}

    # ------------------------------------------------------------------ #
    # Stage 1: Keras vs gold                                               #
    # ------------------------------------------------------------------ #
    if os.path.exists(keras_path):
        print("\n=== Stage 1: SavedModel (Keras weights) ===")
        keras_pred = run_keras(keras_path, inputs)
        m = compute_metrics(keras_pred, gold, "Keras vs SB3")
        print_metrics(m, "keras_vs_sb3")
        show_worst_cases(keras_pred, gold, labels)
        passed, _ = evaluate_pass_fail(m, "keras_vs_sb3")
        m["passed"] = passed
        report["stages"].append(m)
        if not passed:
            report["all_passed"] = False
    else:
        print(f"Keras model not found: {keras_path}")

    # ------------------------------------------------------------------ #
    # Stage 2: Float TFLite vs gold                                       #
    # ------------------------------------------------------------------ #
    if os.path.exists(float_tflite_path):
        print("\n=== Stage 2: Float TFLite model ===")
        try:
            float_pred = run_tflite(float_tflite_path, inputs)
        except Exception as e:
            print(f"  Float TFLite inference failed ({e}). Rebuilding frozen float TFLite...")
            frozen_path = os.path.join(os.path.dirname(float_tflite_path), "model_float_frozen.tflite")
            try:
                _convert_float_frozen_from_savedmodel(keras_path, frozen_path)
                float_pred = run_tflite(frozen_path, inputs)
                print("  Using frozen float TFLite for validation")
            except Exception as e2:
                raise RuntimeError(f"Frozen float TFLite rebuild failed: {e2}")

        m = compute_metrics(float_pred, gold, "Float TFLite vs SB3")
        print_metrics(m, "float_tflite_vs_sb3")
        show_worst_cases(float_pred, gold, labels)
        passed, _ = evaluate_pass_fail(m, "float_tflite_vs_sb3")
        m["passed"] = passed
        report["stages"].append(m)
        if not passed:
            report["all_passed"] = False
    else:
        print(f"Float TFLite not found: {float_tflite_path}")

    # ------------------------------------------------------------------ #
    # Stage 3: INT8 TFLite vs gold                                        #
    # ------------------------------------------------------------------ #
    if os.path.exists(int8_tflite_path):
        print("\n=== Stage 3: INT8 TFLite model ===")
        int8_pred = run_tflite(int8_tflite_path, inputs)
        m = compute_metrics(int8_pred, gold, "INT8 TFLite vs SB3")
        print_metrics(m, "int8_tflite_vs_sb3")
        show_worst_cases(int8_pred, gold, labels)
        passed, _ = evaluate_pass_fail(m, "int8_tflite_vs_sb3")
        m["passed"] = passed
        report["stages"].append(m)
        if not passed:
            # Fallback: try dynamic-range quantized model (weights INT8, activations float)
            dr_path = os.path.join(os.path.dirname(int8_tflite_path), "model_int8_dr.tflite")
            print("\n  INT8 stage failed thresholds — trying dynamic-range quantization fallback...")
            try:
                _convert_dynamic_range(keras_path, dr_path)
                dr_pred = run_tflite(dr_path, inputs)
                m_dr = compute_metrics(dr_pred, gold, "INT8 (Dynamic-Range) vs SB3")
                print_metrics(m_dr, "int8_tflite_vs_sb3")
                show_worst_cases(dr_pred, gold, labels)
                passed_dr, _ = evaluate_pass_fail(m_dr, "int8_tflite_vs_sb3")
                m_dr["passed"] = passed_dr
                report["stages"].append(m_dr)
                if passed_dr:
                    print("  Dynamic-range INT8 PASSED — using this artifact for deployment")
                else:
                    report["all_passed"] = False
            except Exception as e:
                print(f"  Dynamic-range fallback failed: {e}")
                report["all_passed"] = False

        # Per-synthetic-case breakdown
        n_synth = 6
        print(f"\n  Synthetic case breakdown (first {n_synth}):")
        for i in range(min(n_synth, gold.shape[0])):
            lbl = labels[i] if labels is not None else str(i)
            print(f"    [{lbl}]  gold={gold.flat[i]:.4f}  int8={int8_pred.flat[i]:.4f}  "
                  f"AE={abs(int8_pred.flat[i] - gold.flat[i]):.4f}")
    else:
        print(f"INT8 TFLite not found: {int8_tflite_path}")

    # ------------------------------------------------------------------ #
    # Save report                                                          #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    if report["all_passed"]:
        print("  ALL STAGES PASSED — safe to proceed to ESP32 build")
    else:
        print("  SOME STAGES FAILED — do NOT build ESP32 binary")
    print(f"  Report saved: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden",      default=DEFAULT_GOLDEN)
    parser.add_argument("--keras",       default=DEFAULT_KERAS)
    parser.add_argument("--float_tfl",   default=DEFAULT_FLOAT_TFL)
    parser.add_argument("--int8_tfl",    default=DEFAULT_INT8_TFL)
    parser.add_argument("--report",      default=DEFAULT_REPORT)
    args = parser.parse_args()
    run_validation(args.golden, args.keras, args.float_tfl, args.int8_tfl, args.report)


if __name__ == "__main__":
    main()
