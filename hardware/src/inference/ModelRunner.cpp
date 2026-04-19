/**
 * ModelRunner.cpp
 *
 * See ModelRunner.h for interface documentation.
 *
 * Op resolver ops required by this model:
 *   FULLY_CONNECTED   (all Dense layers)
 *   MUL               (mask * embedding)
 *   CONCATENATION     (concat ego + peer_pool)
 *   RESHAPE           (mask reshape)
 *   REDUCE_MAX        (peer max-pool)
 *   TANH              (policy MLP activation)
 *   RELU              (peer encoder activation)
 *   MINIMUM           (action clip_by_value upper: min(x, 1))
 *   SHAPE             (mask reshape — present in DR flatbuffer)
 *   MAXIMUM           (action clip_by_value lower: max(x, 0))
 *   PACK              (occasionally inserted around reshape/slice flows)
 *
 * If the converter fused RELU into FULLY_CONNECTED, you may not need
 * a separate RELU op — check the flatbuffer with Netron if you get
 * kTfLiteError on begin().
 */

#include "ModelRunner.h"
#include <Arduino.h>  // Serial

bool ModelRunner::begin() {
    // InitializeTarget() was added in newer TFLM — not present in TensorFlowLite_ESP32 v1.0
    // tflite::InitializeTarget();

    // ------------------------------------------------------------------ //
    // Register ops used by this model.                                    //
    // If begin() returns false, use Netron to inspect the flatbuffer and  //
    // add any missing ops here.                                           //
    // ------------------------------------------------------------------ //
    resolver_.AddFullyConnected();
    resolver_.AddMul();
    resolver_.AddConcatenation();
    resolver_.AddReshape();
    resolver_.AddReduceMax();
    resolver_.AddTanh();
    resolver_.AddRelu();
    resolver_.AddMinimum();  // for clip upper bound
    resolver_.AddShape();         // used by DR model for mask reshape
    resolver_.AddMaximum();       // for clip lower bound (clip_by_value → MAXIMUM + MINIMUM)
    resolver_.AddStridedSlice();  // present in DR flatbuffer
    resolver_.AddPack();          // present in DR flatbuffer
    resolver_.AddTranspose();     // present in DR flatbuffer

    model_ = tflite::GetModel(g_model_data);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ModelRunner] Schema version mismatch!");
        return false;
    }

    // Placement-new into scratch buffer (no heap allocation).
    // TensorFlowLite_ESP32 v1.0 API requires explicit nullptr for
    // ErrorReporter, ResourceVariables, and Profiler (no default args).
    interpreter_ = new (interp_buf_) tflite::MicroInterpreter(
        model_, resolver_, tensor_arena_, kTensorArenaSize,
        &error_reporter_, nullptr, nullptr);

    TfLiteStatus status = interpreter_->AllocateTensors();
    if (status != kTfLiteOk) {
        Serial.println("[ModelRunner] AllocateTensors() failed!");
        last_status_ = status;
        return false;
    }

    // ------------------------------------------------------------------ //
    // Bind inputs by shape (robust to export order):
    //   ego=[1,18], peer_mask=[1,8], peers=[1,8,6]
    // ------------------------------------------------------------------ //
    input_ego_ = nullptr;
    input_mask_ = nullptr;
    input_peers_ = nullptr;
    output_ = interpreter_->output(0);

    for (int i = 0; i < 3; ++i) {
        TfLiteTensor* tin = interpreter_->input(i);
        if (!tin || !tin->dims) continue;
        const int nd = tin->dims->size;
        if (nd == 2 && tin->dims->data[0] == 1 && tin->dims->data[1] == kEgoDim) {
            input_ego_ = tin;
        } else if (nd == 2 && tin->dims->data[0] == 1 && tin->dims->data[1] == kMaxPeers) {
            input_mask_ = tin;
        } else if (nd == 3 && tin->dims->data[0] == 1 && tin->dims->data[1] == kMaxPeers && tin->dims->data[2] == kPeerFeatureDim) {
            input_peers_ = tin;
        }
    }

    if (!input_ego_ || !input_mask_ || !input_peers_) {
        Serial.println("[ModelRunner] Failed to bind inputs by shape (ego/mask/peers)");
        return false;
    }

    // Sanity check dimensions
    if (input_ego_->dims->size != 2 ||
        input_ego_->dims->data[0] != 1 ||
        input_ego_->dims->data[1] != kEgoDim) {
        Serial.printf("[ModelRunner] ego dim mismatch: expected [1][%d]\n", kEgoDim);
        return false;
    }
    if (input_peers_->dims->size != 3 ||
        input_peers_->dims->data[0] != 1 ||
        input_peers_->dims->data[1] != kMaxPeers ||
        input_peers_->dims->data[2] != kPeerFeatureDim) {
        Serial.printf("[ModelRunner] peers dim mismatch: expected [1][%d][%d]\n",
                      kMaxPeers, kPeerFeatureDim);
        return false;
    }

    Serial.printf("[ModelRunner] Ready. Arena used: %u / %u bytes\n",
                  (unsigned)interpreter_->arena_used_bytes(),
                  (unsigned)kTensorArenaSize);
    return true;
}


float ModelRunner::infer(const float ego[kEgoDim],
                         const float peers[kMaxPeers][kPeerFeatureDim],
                         const float peer_mask[kMaxPeers]) {
    // ------------------------------------------------------------------ //
    // Copy inputs into TFLM tensors.                                      //
    // Tensors are float32 (inference_input_type=float32 from step 5).     //
    // ------------------------------------------------------------------ //
    memcpy(input_ego_->data.f, ego, kEgoDim * sizeof(float));
    memcpy(input_peers_->data.f, peers, kMaxPeers * kPeerFeatureDim * sizeof(float));
    memcpy(input_mask_->data.f, peer_mask, kMaxPeers * sizeof(float));

    // ------------------------------------------------------------------ //
    // Run inference                                                        //
    // ------------------------------------------------------------------ //
    TfLiteStatus status = interpreter_->Invoke();
    last_status_ = status;
    if (status != kTfLiteOk) {
        Serial.println("[ModelRunner] Invoke() failed!");
        return -1.0f;
    }

    // ------------------------------------------------------------------ //
    // Read output (already clipped to [0,1] by the Lambda layer)          //
    // ------------------------------------------------------------------ //
    float action = output_->data.f[0];

    // Defensive clamp in case quantization pushed slightly out of range
    if (action < 0.0f) action = 0.0f;
    if (action > 1.0f) action = 1.0f;

    return action;
}


void ModelRunner::printArenaUsed() const {
    if (interpreter_) {
        Serial.printf("[ModelRunner] Arena used: %u / %u bytes (%.1f%%)\n",
                      (unsigned)interpreter_->arena_used_bytes(),
                      (unsigned)kTensorArenaSize,
                      100.0f * interpreter_->arena_used_bytes() / kTensorArenaSize);
    }
}
