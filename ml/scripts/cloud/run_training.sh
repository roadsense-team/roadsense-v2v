#!/bin/bash
# =============================================================================
# RoadSense Training Run 016 - braking_received Observation Feature
# =============================================================================
# Paste this into EC2 User Data when launching from the roadsense-training AMI.
#
# BEFORE LAUNCHING - customize these variables:
#   RUN_ID        - Unique run identifier
#   GITHUB_PAT    - Your GitHub Personal Access Token
#   TOTAL_STEPS   - Training timesteps (default: 10000000)
#   S3_BUCKET     - S3 bucket for results
#
# Run 016 — Fix the dead critic that killed Runs 014 and 015:
#
#   Root cause of Run 014/015 failure:
#     Gradual slowDown produces min_peer_accel ~ -0.3 to -0.5 (normalized),
#     overlapping with normal traffic. But PENALTY_IGNORING_HAZARD creates a
#     ~2000 point return gap. The critic sees same-looking states with wildly
#     different returns → explained_variance = 0 for the entire run.
#
#   Changes in this run:
#     1. Added braking_received binary feature at ego[5] (6-dim ego, up from 5).
#        1.0 when any peer's V2V-reported accel <= -2.5 m/s², 0.0 otherwise.
#        Gives the critic a clean binary partition for hazard vs calm states.
#     2. braking_received is LATCHED per episode — once set, stays 1.0 for the
#        rest of the episode. Matches the reward's latched penalty condition.
#        Eliminates state/reward aliasing where ego[5]=0 but penalty still fires.
#     3. DeepSetExtractor features_dim updated: 37 → 38 (32 embed + 6 ego).
#     4. H5 validation script fixed to compute and pass braking_received.
#
#   Retained from Run 014/015:
#     - Latch _hazard_source_braking_latched in convoy_env.py (~99% coverage)
#     - Gradual hazard injection (slowDown 2-4s, domain randomized)
#     - CF override, Deep Sets, HAZARD_PROBABILITY=0.5
#     - base_real: sigma=0.0, speedFactor=1.0, speedDev=0, 25m spacing
#     - Hazard-gated reward terms (PENALTY_IGNORING_HAZARD=-5.0,
#       REWARD_EARLY_REACTION=+2.0, BRAKING_ACCEL_THRESHOLD=-2.5)
#     - Hyperparams: LR=1e-4, n_steps=4096, ent_coef=0.0, log_std_init=-0.5
#     - Dataset generation params (base_real, seed=42, 25 train + 40 eval)
#
#   Key diagnostic to watch:
#     - explained_variance: MUST rise above 0 by 200k steps.
#       If it stays at 0 with braking_received, there is a deeper issue.
#     - V2V reaction at 1M checkpoint: first alive/dead signal.
#
#   Pass criteria:
#     1. SUMO eval: >90% V2V reaction (target 100%, Run 011 baseline)
#     2. avg_reward > -200 (Run 011 was -86.62)
#     3. explained_variance > 0.5 by 5M steps
#     4. H5 re-validation: model action > 0.1 within 3s of peer braking onset
#     5. False positive rate < 5% in Extra Recording
# =============================================================================
exec > /var/log/training-run.log 2>&1

# ===================== CUSTOMIZE THESE =====================
RUN_ID="cloud_prod_016"
GITHUB_PAT="<YOUR_PAT_HERE>"
TOTAL_STEPS=10000000
S3_BUCKET="saferide-training-results"
# ===========================================================

export AWS_DEFAULT_REGION=il-central-1
export AWS_REGION="$AWS_DEFAULT_REGION"
WORK_DIR="/home/ubuntu/work"
DATASET_DIR="ml/scenarios/datasets/dataset_v8"
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

# 3. Generate dataset_v8 from base_real (same params as v6/v7)
echo "[3/7] Generating dataset_v8 from base_real..."
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
echo "[6/7] Starting training ($TOTAL_STEPS steps)..."
set +e
./ml/run_docker.sh train \
    --dataset_dir "$DATASET_DIR" \
    --emulator_params "$EMULATOR_PARAMS" \
    --total_timesteps "$TOTAL_STEPS" \
    --run_id "$RUN_ID" \
    --output_dir /work/results \
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
