#!/bin/bash
# =============================================================================
# RoadSense Training Run 018 - Signal Strength + Reward Normalization
# =============================================================================
# Paste this into EC2 User Data when launching from the roadsense-training AMI.
#
# BEFORE LAUNCHING - customize these variables:
#   RUN_ID        - Unique run identifier
#   GITHUB_PAT    - Your GitHub Personal Access Token
#   TOTAL_STEPS   - Training timesteps (default: 2000000 for diagnostic run)
#   S3_BUCKET     - S3 bucket for results
#
# Run 018 — Fix the dead critic (explained_variance=0 in Runs 014-017):
#
#   Root cause (confirmed by 5 consecutive failed runs):
#     Gradual slowDown (2-4s) produces peer accel of ~-4.6 m/s^2 (normalized
#     -0.46), which OVERLAPS with normal car-following adjustments (-0.3 to
#     -0.5). The critic cannot distinguish hazard from normal driving in the
#     observation space. explained_variance=0 is the mathematically correct
#     answer: identical-looking states genuinely map to different returns.
#
#     Run 011 (last success, explained_variance=0.956) used setSpeed(0) which
#     produced an instant -10.0 m/s^2 signal — unmistakable in the observation.
#
#   Changes in this run (2 fixes):
#     1. FASTER SLOWDOWN: BRAKING_DURATION 2-4s -> 0.5-1.5s.
#        At 13.9 m/s convoy speed: produces -9.3 to -27.8 m/s^2 (clamped to
#        -10.0 by VehicleState), matching Run 011's signal strength.
#        Still gradual (slowDown, not setSpeed) for sim-to-real validity.
#        Real emergency braking at -8.6 m/s^2 takes ~1.6s from 50 km/h.
#     2. REWARD NORMALIZATION: VecNormalize(norm_obs=False, norm_reward=True).
#        Raw returns span -2500 to +2000 with no normalization.
#        Value function now learns from unit-scale targets.
#
#   Also includes (from Codex):
#     3. Warmup contamination fix: braking_received latch cannot be set during
#        pre-episode warmup (update_braking_latch=False + explicit reset).
#
#   Retained from Run 017 (all 5 structural fixes still active):
#     - Reward aligned to observation braking_received latch (Fix 1)
#     - Progress feature ego[6] = step/max_steps, 7-dim ego (Fix 2)
#     - Fixed hazard timing: HAZARD_PROBABILITY=1.0, step=200 (Fix 3)
#     - Speed-gated penalty: suppressed when ego_speed <= 0.5 m/s (Fix 4)
#     - Instrumentation: obs/reward divergence, stop-suppression (Fix 5)
#     - braking_received remains PRE-CONE
#     - CF override, Deep Sets, features_dim=39
#     - base_real: sigma=0.0, speedFactor=1.0, speedDev=0, 25m spacing
#     - Hazard-gated reward terms (PENALTY_IGNORING_HAZARD=-5.0,
#       REWARD_EARLY_REACTION=+2.0, BRAKING_ACCEL_THRESHOLD=-2.5)
#     - Hyperparams: LR=1e-4, n_steps=4096, ent_coef=0.0, log_std_init=-0.5
#     - Dataset generation params (base_real, seed=42, 25 train + 40 eval)
#
#   Key diagnostic to watch:
#     - explained_variance: >0.1 by 100k, >0.3 by 300k.
#       If it stays near 0 / numerical noise, kill the run early.
#     - V2V reaction at 1M checkpoint: target >=70% in deterministic eval.
#
#   Pass criteria:
#     1. explained_variance wakes up clearly before 300k
#     2. SUMO eval at 1M: >=70% V2V reaction
#     3. No regression in collision rate
#     4. If alive at 1M, continue to 2M and decide whether to extend later
#
#   Kill criteria (same as Run 017):
#     - explained_variance=0 at 300k
#     - V2V reaction=0% at first checkpoint
# =============================================================================
exec > /var/log/training-run.log 2>&1

# ===================== CUSTOMIZE THESE =====================
RUN_ID="cloud_prod_018"
GITHUB_PAT="<YOUR_PAT_HERE>"
TOTAL_STEPS=2000000
S3_BUCKET="saferide-training-results"
# ===========================================================

export AWS_DEFAULT_REGION=il-central-1
export AWS_REGION="$AWS_DEFAULT_REGION"
WORK_DIR="/home/ubuntu/work"
DATASET_DIR="ml/scenarios/datasets/dataset_v9"
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

# 3. Generate dataset_v9 from base_real (same canonical params)
echo "[3/7] Generating dataset_v9 from base_real..."
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
    --eval_force_hazard \
    --eval_hazard_step 200 \
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
