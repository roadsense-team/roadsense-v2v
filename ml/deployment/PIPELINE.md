# ESP32 Deployment Pipeline

Converts the trained SB3 PPO model (PyTorch) → INT8 TFLite → ESP32 firmware.

## Quick reference

```
python -m ml.deployment.step1_export_weights
python -m ml.deployment.step2_keras_model         # (verify only — no output file)
python -m ml.deployment.step3_port_weights
python -m ml.deployment.step4_gen_golden_vectors
python -m ml.deployment.step5_quantize_tflite
python -m ml.deployment.step6_validate_pipeline   # must ALL PASS before continuing
python -m ml.deployment.step7_export_c_array
python -m ml.deployment.step8_esp32_golden_test
# then: pio run -e <env> --target upload && pio device monitor
```

All commands run from the repo root **inside Docker**:
```bash
./ml/run_docker.sh  # or: docker run --rm -v $(pwd):/work:Z roadsense-ml:latest bash
```

---

## Phase 1 — Audit results

### What exists

| Artifact | Location | Status |
|---|---|---|
| Trained SB3 model (latest) | `ml/results/run_025_replay_v1/model_final.zip` | ✅ |
| DeepSetExtractor (PyTorch) | `ml/models/deep_set_extractor.py` | ✅ |
| ObservationBuilder (preprocessing) | `ml/envs/observation_builder.py` | ✅ |
| ConeFilter (C++) | `hardware/src/inference/ConeFilter.cpp` | ✅ |
| Validation data (recordings) | `ml/data/convoy_analysis*/convoy_observations.npz` | ✅ |

### What is missing (created by this pipeline)

| Artifact | Created by | Location |
|---|---|---|
| Numpy weight export | step1 | `ml/deployment/artifacts/weights.npz` |
| Architecture config | step1 | `ml/deployment/artifacts/model_config.json` |
| Float Keras model | step3 | `ml/deployment/artifacts/model_float.keras` |
| Golden I/O vectors | step4 | `ml/deployment/artifacts/golden_vectors.npz` |
| Float TFLite | step5 | `ml/deployment/artifacts/model_float.tflite` |
| INT8 TFLite | step5 | `ml/deployment/artifacts/model_int8.tflite` |
| Validation report | step6 | `ml/deployment/artifacts/validation_report.json` |
| C array header | step7 | `hardware/src/inference/model_data.h` |
| C array source | step7 | `hardware/src/inference/model_data.cc` |
| ESP32 golden test | step8 | `hardware/test/test_model_runner/golden_vectors.h` |
| ModelRunner.h | already created | `hardware/src/inference/ModelRunner.h` |
| ModelRunner.cpp | already created | `hardware/src/inference/ModelRunner.cpp` |

---

## Phase 2 — Deployment path chosen

**Path: Direct weight transfer → Keras → Post-Training INT8 quantization → TFLite Micro**

Rejected alternatives:
- **ONNX export**: PyTorch→ONNX→TFLite conversion chain is fragile for custom ops (masked max-pool). Not needed when we can directly port weights to Keras.
- **Quantization-Aware Training (QAT)**: Would require retraining in TF/Keras from scratch. The model already generalizes well; PTQ is sufficient.
- **Distillation (true teacher-student)**: The Keras model IS the original model (same weights, same architecture). Distillation is not needed.
- **Custom C++ without TFLM**: Viable (model is tiny), but TFLM gives us verified INT8 math and easier debugging.

**Why PTQ is sufficient:**
- Model has only ~8,800 parameters
- Output is 1D braking signal — quantization error averages out
- Acceptance criterion (MAE < 0.02) is loose compared to 8-bit quantization error (~0.004 typical)

**Key architectural insight:**
The peer encoder uses ReLU×2, so all peer embeddings are ≥ 0.
Setting invalid peers to 0 (for masked max-pool) is exactly equivalent to -inf masking.
This means the full model expresses in standard TFLite ops with no custom kernels.

---

## Phase 3 — Model architecture (exact)

```
Input: ego (18,)         ← 6 features × 3 stacked frames
Input: peers (8, 6)      ← up to 8 peers, 6 features each
Input: peer_mask (8,)    ← 1.0=valid, 0.0=padding

peers → TimeDistributed(Dense(32, relu))  → (8, 32)
      → TimeDistributed(Dense(32, relu))  → (8, 32)
      → * peer_mask.reshape(8,1)          → (8, 32)   [zero invalid peers]
      → reduce_max(axis=1)               → (32,)     [best peer per feature]

concat([peer_pool (32,), ego (18,)])      → (50,)

→ Dense(64, tanh)
→ Dense(64, tanh)
→ Dense(1)
→ clip_by_value(0, 1)                    → action (1,)
```

**Observation features (already normalized by ObservationBuilder):**

Ego (6 per frame, × 3 frames = 18 total):
```
ego[0]  = speed / 30.0                     ∈ [0, 1]
ego[1]  = acceleration / 10.0             ∈ [-1, 1]
ego[2]  = visible_peer_count / 8.0        ∈ [0, 1]
ego[3]  = min_peer_accel / 10.0           ∈ [-1, 0]
ego[4]  = braking_received_decay          ∈ [0, 1]
ego[5]  = max_closing_speed / 30.0        ∈ [0, 1]
```

Peers (6 features each):
```
peer[0] = rel_x / 100.0    (forward distance, normalized)
peer[1] = rel_y / 100.0    (lateral offset, normalized)
peer[2] = rel_speed / 30.0 (negative = peer slower than ego)
peer[3] = rel_heading / 180.0
peer[4] = peer_accel / 10.0
peer[5] = peer_age_ms / 500.0
```

---

## Phase 4 — Acceptance criteria

| Stage | MAE | Max AE | Agree@0.05 | Cross@0.1 |
|---|---|---|---|---|
| Keras vs SB3 | < 1e-4 | < 5e-4 | > 99.0% | > 99.5% |
| Float TFLite vs SB3 | < 1e-3 | < 5e-3 | > 98.0% | > 99.0% |
| INT8 TFLite vs SB3 | < 0.02 | < 0.05 | > 95.0% | > 97.0% |

**If INT8 fails:** the most common cause is insufficient calibration data.
- Run step4 with `--n_rand 500`
- Re-run step5 to requantize

**Cross@0.1 / Cross@0.3 explained:**
Measures whether both models agree on whether the braking signal crosses a threshold.
This is what matters operationally — a false negative (missing a hazard) is catastrophic,
a false positive (braking when safe) is merely uncomfortable.

---

## Phase 5 — ESP32 memory budget

| Item | Flash | RAM |
|---|---|---|
| Model flatbuffer | ~9 KB | 0 (in .rodata) |
| TFLM tensor arena | 0 | 12 KB |
| TFLM code | ~60 KB | ~2 KB |
| ModelRunner object | 0 | <1 KB |
| **Total** | **~69 KB** | **~15 KB** |

ESP32 has: 4 MB flash, 520 KB SRAM.  Well within budget.

---

## Phase 6 — End-to-end checklist

### Step 0: Prerequisites (run once)
```bash
pip install tensorflow>=2.13  stable-baselines3  torch  numpy
```

### Step 1: Export weights
```bash
python -m ml.deployment.step1_export_weights \
  --model ml/results/run_025_replay_v1/model_final.zip
```
**Expected output files:**
- `ml/deployment/artifacts/weights.npz`
- `ml/deployment/artifacts/model_config.json`

**Blocker if missing:** torch or stable-baselines3 not installed.

---

### Step 3: Port weights to Keras
```bash
python -m ml.deployment.step3_port_weights
```
**Expected output:** `ml/deployment/artifacts/model_float.keras`

**Check:** Script prints "Shape check passed" and "Range check passed".

**Blocker if fails:** Key names in policy.pth don't match expected names.
Run step1 first to print all key names, then edit `required` dict in step1.

---

### Step 4: Generate golden vectors
```bash
python -m ml.deployment.step4_gen_golden_vectors
```
**Expected output:** `ml/deployment/artifacts/golden_vectors.npz`

Contains 206+ I/O pairs from SB3 model (ground truth).

---

### Step 5: Quantize to TFLite
```bash
python -m ml.deployment.step5_quantize_tflite
```
**Expected output files:**
- `ml/deployment/artifacts/model_float.tflite`
- `ml/deployment/artifacts/model_int8.tflite`
- `ml/deployment/artifacts/model_int8_quant_params.json`  (if INT8 I/O)

**Blocker if fails:** TFLite converter can't handle `tf.reduce_max` with axis arg.
Fix: replace Lambda layer in step2 with `tf.keras.layers.GlobalMaxPooling1D` equivalent.

---

### Step 6: Validate pipeline (GATE)
```bash
python -m ml.deployment.step6_validate_pipeline
```
**Expected output:** `ml/deployment/artifacts/validation_report.json`

**MUST read "ALL STAGES PASSED" before continuing.**

**If INT8 fails MAE:**
1. Increase calibration data: rerun step4 with `--n_rand 500`
2. Rerun step5
3. Rerun step6

**If Keras fails MAE > 1e-4:**
Weight transposition error. Check step3 — PyTorch weight is (out, in), Keras is (in, out).

---

### Step 7: Export C array
```bash
python -m ml.deployment.step7_export_c_array
```
**Expected output files:**
- `hardware/src/inference/model_data.h`
- `hardware/src/inference/model_data.cc`

---

### Step 8: Generate ESP32 test
```bash
python -m ml.deployment.step8_esp32_golden_test
```
**Expected output:** `hardware/test/test_model_runner/golden_vectors.h`

---

### Step 9: Build and flash ESP32
1. Add TFLM library to `platformio.ini`:
```ini
[env:esp32_model]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps =
    tensorflow/lite-micro@^1.0.0
build_src_filter =
    +<inference/ModelRunner.cpp>
    +<inference/model_data.cc>
    +<inference/ConeFilter.cpp>
```

2. Build:
```bash
pio run -e esp32_model
```

3. Flash:
```bash
pio run -e esp32_model --target upload
```

4. Monitor:
```bash
pio device monitor --baud 115200
```

---

### Step 10: On-device validation
Add the test env to platformio.ini and flash the test binary:
```ini
[env:test_model_runner]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps = tensorflow/lite-micro@^1.0.0
build_src_filter =
    +<../test/test_model_runner/test_main.cpp>
    +<inference/ModelRunner.cpp>
    +<inference/model_data.cc>
```

Expected Serial output:
```
=== ModelRunner Golden Vector Test ===
[ModelRunner] Ready. Arena used: XXXX / 12288 bytes
--- Results: 50 / 50 passed ---
ALL TESTS PASSED
```

**If a vector fails:** compare the index to golden_vectors.h and re-check the
input tensor order in ModelRunner.cpp (indices 0/1/2).

---

## Troubleshooting

### "AllocateTensors() failed"
Arena too small. Increase `kTensorArenaSize` in ModelRunner.h by 2 KB steps.

### "Schema version mismatch"
TFLM library version doesn't match the TFLite flatbuffer schema.
Update both the Python `tensorflow` version and the Arduino lib to matching versions.

### "Invoke() failed" (kTfLiteError)
Missing op in resolver. Use Netron (netron.app) to open model_int8.tflite,
check all op types, and add any missing op to `resolver_` in ModelRunner.cpp.

### INT8 MAE fails validation
Not enough calibration data. Rerun step4 with `--n_rand 500` and step5.

### Wrong tensor input order
ModelRunner.cpp hard-codes `input(0)=ego, input(1)=peers, input(2)=peer_mask`.
Verify this matches your model by running step5 and reading the
`print_tensor_details()` output.  Update indices if needed.
