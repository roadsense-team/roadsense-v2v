/**
 * ModelRunner.h
 *
 * TFLite Micro inference wrapper for the RoadSense DeepSet braking policy.
 *
 * Architecture recap (from model_data.h):
 *   ego      [1][18] float32  — 6 features × 3 stacked frames
 *   peers    [1][8][6] float32
 *   peer_mask[1][8] float32
 *   → action [1][1] float32  braking ∈ [0, 1]
 *
 * Usage:
 *   ModelRunner runner;
 *   runner.begin();                 // call once in setup()
 *
 *   // Fill inputs (already normalized, see ObservationBuilder)
 *   float ego[18] = {...};
 *   float peers[8][6] = {...};
 *   float mask[8] = {...};
 *   float action = runner.infer(ego, peers, mask);
 *
 * Memory:
 *   TFLM arena: 64 KB in BSS (see kTensorArenaSize below).
 *   Model flash: ~29 KB in .rodata  (model_int8_dr.tflite, dynamic-range INT8).
 *   Total: ~93 KB.
 */
#pragma once

#include <cstdint>
#include <cstring>

// TFLite Micro headers — available in esp32-tflite-micro Arduino library
// or via idf-component-manager (tensorflow/lite-micro)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

static constexpr int kEgoDim          = 18;
static constexpr int kMaxPeers        = 8;
static constexpr int kPeerFeatureDim  = 6;
static constexpr int kActionDim       = 1;
// Arena size: 64 KB for DR model (float32 activations, peer encoder intermediates).
static constexpr int kTensorArenaSize = 64 * 1024;

class ModelRunner {
public:
    ModelRunner() = default;

    /**
     * Initialize TFLM and allocate tensors.
     * Call once from setup().
     * Returns true on success, false on error.
     */
    bool begin();

    /**
     * Run inference.
     *
     * @param ego        Normalized ego features, length kEgoDim (18).
     * @param peers      Normalized peer features [kMaxPeers][kPeerFeatureDim].
     *                   Invalid peers should be zero-padded.
     * @param peer_mask  1.0f for valid peers, 0.0f for padding, length kMaxPeers.
     * @return           Braking signal in [0, 1], or -1.0f on error.
     */
    float infer(const float ego[kEgoDim],
                const float peers[kMaxPeers][kPeerFeatureDim],
                const float peer_mask[kMaxPeers]);

    /**
     * Return the last TFLM status code (kTfLiteOk == 0 on success).
     */
    int lastStatus() const { return last_status_; }

    /**
     * Print arena high-water mark to Serial (call after first inference).
     */
    void printArenaUsed() const;

private:
    alignas(16) uint8_t tensor_arena_[kTensorArenaSize];

    tflite::MicroErrorReporter     error_reporter_;
    const tflite::Model*           model_       = nullptr;
    tflite::MicroInterpreter*      interpreter_ = nullptr;
    tflite::MicroMutableOpResolver<16> resolver_;

    TfLiteTensor* input_ego_   = nullptr;
    TfLiteTensor* input_peers_ = nullptr;
    TfLiteTensor* input_mask_  = nullptr;
    TfLiteTensor* output_      = nullptr;

    int last_status_ = 0;

    // Scratch buffer for interpreter (no heap allocation)
    uint8_t interp_buf_[sizeof(tflite::MicroInterpreter)];
};
