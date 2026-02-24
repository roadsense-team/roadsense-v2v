#!/bin/bash
# =============================================================================
# RoadSense Training Run - User-Data Template
# =============================================================================
# Paste this into EC2 User Data when launching from the roadsense-training AMI.
#
# BEFORE LAUNCHING - customize these variables:
#   RUN_ID        - Unique run identifier (e.g., cloud_prod_002)
#   GITHUB_PAT    - Your GitHub Personal Access Token
#   TOTAL_STEPS   - Training timesteps (default: 10000000)
#   S3_BUCKET     - S3 bucket for results
#
# The generate command below uses the CANONICAL flags from CLAUDE.md.
# Only change: --base_dir, --output_dir, --seed, --train_count, --eval_count.
# =============================================================================
exec > /var/log/training-run.log 2>&1

# ===================== CUSTOMIZE THESE =====================
RUN_ID="cloud_prod_002"
GITHUB_PAT="<YOUR_PAT_HERE>"
TOTAL_STEPS=10000000
S3_BUCKET="saferide-training-results"
# ===========================================================

export AWS_DEFAULT_REGION=il-central-1
WORK_DIR="/home/ubuntu/work"

# -------------------------------------------------------------------
# CLEANUP TRAP: S3 upload + shutdown ALWAYS run, even if training fails
# -------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Cleanup: uploading results and shutting down ==="

    # Upload whatever results exist (model, checkpoints, metrics)
    if [ -d "$WORK_DIR/results" ]; then
        echo "Uploading results to S3..."
        aws s3 cp "$WORK_DIR/results" "s3://$S3_BUCKET/$RUN_ID" --recursive || \
            echo "WARNING: S3 upload failed. Results are still on disk at $WORK_DIR/results"
    else
        echo "WARNING: No results directory found."
    fi

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
echo "[1/5] Pulling latest code..."
cd "$WORK_DIR"
git remote set-url origin "https://${GITHUB_PAT}@github.com/roadsense-team/roadsense-v2v.git"
git pull origin master
# Clean PAT from memory-resident remote
git remote set-url origin https://github.com/roadsense-team/roadsense-v2v.git
# Re-fix permissions after pull (git may reset them)
chmod +x "$WORK_DIR/ml/run_docker.sh"

# 2. Rebuild Docker image if Dockerfile changed (usually instant from cache)
echo "[2/5] Rebuilding Docker image (cached - should be fast)..."
cd "$WORK_DIR/ml"
docker build -t roadsense-ml:latest .

# 3. Generate dataset
echo "[3/5] Generating dataset_v2..."
cd "$WORK_DIR"
./ml/run_docker.sh generate \
    --base_dir ml/scenarios/base \
    --output_dir ml/scenarios/datasets/dataset_v2/base \
    --seed 42 \
    --train_count 16 \
    --eval_count 4 \
    --route_randomize_non_ego \
    --route_include_v001 \
    --peer_drop_prob 0.3 \
    --min_peers 1 \
    --emulator_params ml/espnow_emulator/emulator_params_measured.json

# 4. Train (disable set -e so cleanup trap runs even if training exits non-zero)
echo "[4/5] Starting training ($TOTAL_STEPS steps)..."
set +e
./ml/run_docker.sh train \
    --dataset_dir ml/scenarios/datasets/dataset_v2/base \
    --emulator_params ml/espnow_emulator/emulator_params_measured.json \
    --total_timesteps "$TOTAL_STEPS" \
    --run_id "$RUN_ID" \
    --output_dir /work/results
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "WARNING: Training exited with code $TRAIN_EXIT"
fi

echo "[5/5] Training done. Cleanup trap will handle S3 upload and shutdown."
# cleanup() runs automatically via EXIT trap
