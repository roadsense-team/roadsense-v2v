#!/bin/bash
# =============================================================================
# RoadSense Training Run 025 - Temporal Ego Stack (2M)
# =============================================================================
# Paste this into EC2 User Data when launching from the roadsense-training AMI.
#
# BEFORE LAUNCHING - customize these variables:
#   RUN_ID        - Unique run identifier
#   GITHUB_PAT    - Your GitHub Personal Access Token
#   TOTAL_STEPS   - Training timesteps (default: 2000000)
#   S3_BUCKET     - S3 bucket for results
#
# Run 025 — Temporal ego frame stacking to break the discrimination ceiling.
#
#   Why Run 025 exists:
#     - Run 024 (all variants v3-v6) hit a discrimination ceiling:
#       the model cannot distinguish hazard onset from stale braking
#       decay using a single-timestep observation.
#     - v3: 72% det / 68% FP | v5: 32% / 16.6% — Pareto stalled.
#     - Root cause: single-frame ego lacks temporal context.
#
#   Run 025 changes:
#     1. ego_stack_frames=3: ego observation stacks last 3 frames.
#        [ego_t, ego_{t-1}, ego_{t-2}] → 18-dim (was 6-dim).
#     2. DeepSetExtractor features_dim: 38→50 (32 embed + 18 ego).
#     3. Fresh SUMO training — no weight transfer from Run 023
#        (obs dim change breaks checkpoint compatibility).
#
#   Everything else unchanged from Run 023:
#     - base_real road and route grounding
#     - Heading-free observation (ego heading still removed)
#     - Hazard decel randomization: uniform [3.0, 10.0] m/s²
#     - Resolved-hazard episodes: 40% resolve after 2-5s
#     - Reward: decay-scaled penalty/bonus (obs/reward aligned)
#     - Architecture: Deep Sets
#     - Hyperparams: LR=1e-4, n_steps=4096, ent_coef=0.0, log_std_init=-0.5
#     - VecNormalize(norm_obs=False, norm_reward=True)
#     - BRAKING_DURATION 0.5-1.5s (keeps signal sharp)
#     - Dataset: base_real, seed=42, 25 train + 40 eval
#
#   Diagnostic success targets:
#     - SUMO V2V reaction >= Run 023 (~85%)
#     - 0% collisions
#     - After replay fine-tune: >40% detection AND <15% FP on Recording #2
# =============================================================================
exec > /var/log/training-run.log 2>&1

# ===================== CUSTOMIZE THESE =====================
RUN_ID="cloud_prod_025"
GITHUB_PAT="<YOUR_PAT_HERE>"
TOTAL_STEPS=2000000
S3_BUCKET="saferide-training-results"
EGO_STACK_FRAMES=3
# ===========================================================

export AWS_DEFAULT_REGION=il-central-1
export AWS_REGION="$AWS_DEFAULT_REGION"
WORK_DIR="/home/ubuntu/work"
DATASET_DIR="ml/scenarios/datasets/dataset_v14_run025"
EMULATOR_PARAMS="ml/espnow_emulator/emulator_params_measured.json"

# -------------------------------------------------------------------
# CLEANUP TRAP: S3 upload + shutdown ALWAYS run, even if training fails
# -------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Cleanup: uploading results and shutting down ==="

    # Upload whatever results exist (model, checkpoints, metrics)
    if [ -d "$WORK_DIR/results" ]; then
        echo "Uploading results to S3..."
        aws s3 cp "$WORK_DIR/results" "s3://$S3_BUCKET/$RUN_ID" --recursive --region "$AWS_DEFAULT_REGION" || \
            echo "WARNING: S3 upload failed. Results are still on disk at $WORK_DIR/results"
    else
        echo "WARNING: No results directory found at $WORK_DIR/results"
    fi

    # Also upload the training log itself for debugging
    aws s3 cp /var/log/training-run.log "s3://$S3_BUCKET/$RUN_ID/training-run.log" --region "$AWS_DEFAULT_REGION" || \
        echo "WARNING: Could not upload training log to S3"

    echo "Finished: $(date -u)"
    echo "=== Shutting down ==="
    shutdown -h now
}
trap cleanup EXIT
# -------------------------------------------------------------------

echo "=== RoadSense Training Run: $RUN_ID ==="
echo "Started: $(date -u)"

# Fail fast on errors during setup (steps 0-3)
set -euo pipefail

# 0. Fix known AMI quirks (git ownership + script permissions)
git config --global --add safe.directory "$WORK_DIR"
chmod +x "$WORK_DIR/ml/run_docker.sh"

# 1. Pull latest code
echo "[1/7] Pulling latest code..."
cd "$WORK_DIR"
git remote set-url origin "https://${GITHUB_PAT}@github.com/roadsense-team/roadsense-v2v.git"
git pull origin master
# Clean PAT from memory-resident remote
git remote set-url origin https://github.com/roadsense-team/roadsense-v2v.git
# Re-fix permissions after pull (git may reset them)
chmod +x "$WORK_DIR/ml/run_docker.sh"

# 2. Rebuild Docker image if Dockerfile changed
echo "[2/7] Rebuilding Docker image (cached - should be fast)..."
cd "$WORK_DIR/ml"
docker build -t roadsense-ml:latest .

# 3. Generate dataset from base_real (same canonical params as every run)
echo "[3/7] Generating dataset from base_real..."
cd "$WORK_DIR"
./ml/run_docker.sh generate \
    --base_dir ml/scenarios/base_real \
    --output_dir "$DATASET_DIR" \
    --seed 42 \
    --train_count 25 \
    --eval_count 40 \
    --peer_drop_prob 0.4 \
    --eval_peer_drop_prob 0.0 \
    --min_peers 1 \
    --emulator_params "$EMULATOR_PARAMS"

# 4. Enforce deterministic eval peer counts (8x each n=1-5)
echo "[4/7] Enforcing eval peer counts (8x each n=1-5)..."
./ml/run_docker.sh python3 ml/scripts/fix_eval_peer_counts.py \
    --dataset_dir "$DATASET_DIR" \
    --target_counts 1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5

# 5. Verify dataset structure
echo "[5/7] Verifying dataset structure..."
./ml/run_docker.sh python3 -c "
import json, sys, pathlib
d = pathlib.Path('$DATASET_DIR')
m = json.loads((d / 'manifest.json').read_text())
train = m.get('train_scenarios', [])
evl = m.get('eval_scenarios', [])
print(f'Train scenarios: {len(train)}')
print(f'Eval scenarios:  {len(evl)}')
if len(train) < 20:
    print('ERROR: Too few train scenarios'); sys.exit(1)
if len(evl) < 40:
    print('ERROR: Too few eval scenarios'); sys.exit(1)
print('Dataset verification PASSED')
"

# 6. Train
# NOTE: eval uses fixed_step mode (--eval_hazard_step 200) for
# reproducible comparison with earlier runs.  Training uses state_bucket
# mode (configured in HazardInjector default).
# Run 025: --ego_stack_frames 3 for temporal ego observation stacking.
echo "[6/7] Starting training ($TOTAL_STEPS steps, ego_stack=$EGO_STACK_FRAMES)..."
set +e
./ml/run_docker.sh train \
    --dataset_dir "$DATASET_DIR" \
    --emulator_params "$EMULATOR_PARAMS" \
    --total_timesteps "$TOTAL_STEPS" \
    --run_id "$RUN_ID" \
    --output_dir /work/results \
    --ego_stack_frames "$EGO_STACK_FRAMES" \
    --eval_force_hazard \
    --eval_use_deterministic_matrix \
    --eval_matrix_peer_counts "1,2,3,4,5" \
    --eval_matrix_episodes_per_bucket 10
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "WARNING: Training exited with code $TRAIN_EXIT"
else
    echo "Training completed successfully (exit code 0)"
fi

echo "[7/7] Training done. Cleanup trap will handle S3 upload and shutdown."
