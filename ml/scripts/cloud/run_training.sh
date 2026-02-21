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
set -euo pipefail
exec > /var/log/training-run.log 2>&1

# ===================== CUSTOMIZE THESE =====================
RUN_ID="cloud_prod_002"
GITHUB_PAT="<YOUR_PAT_HERE>"
TOTAL_STEPS=10000000
S3_BUCKET="saferide-training-results"
# ===========================================================

export AWS_DEFAULT_REGION=il-central-1
WORK_DIR="/home/ubuntu/work"

echo "=== RoadSense Training Run: $RUN_ID ==="
echo "Started: $(date -u)"

# 1. Pull latest code
echo "[1/5] Pulling latest code..."
cd "$WORK_DIR"
git remote set-url origin "https://${GITHUB_PAT}@github.com/roadsense-team/roadsense-v2v.git"
git pull origin master
# Clean PAT from memory-resident remote
git remote set-url origin https://github.com/roadsense-team/roadsense-v2v.git

# 2. Rebuild Docker image if Dockerfile changed (usually instant from cache)
echo "[2/5] Rebuilding Docker image (cached - should be fast)..."
cd "$WORK_DIR/ml"
docker build -t roadsense-ml:latest .

# 3. Generate dataset
echo "[3/5] Generating dataset_v2..."
cd "$WORK_DIR"
./ml/run_docker.sh generate \
    --base_dir ml/scenarios/base \
    --output_dir ml/scenarios/datasets/dataset_v2/base_01 \
    --seed 42 \
    --train_count 16 \
    --eval_count 4 \
    --route_randomize_non_ego \
    --route_include_v001 \
    --peer_drop_prob 0.3 \
    --min_peers 1 \
    --emulator_params ml/espnow_emulator/emulator_params_measured.json

# 4. Train
echo "[4/5] Starting training ($TOTAL_STEPS steps)..."
./ml/run_docker.sh train \
    --dataset_dir ml/scenarios/datasets/dataset_v2/base_01 \
    --emulator_params ml/espnow_emulator/emulator_params_measured.json \
    --total_timesteps "$TOTAL_STEPS" \
    --run_id "$RUN_ID" \
    --output_dir /work/results

# 5. Upload results to S3
echo "[5/5] Uploading results to S3..."
aws s3 cp "$WORK_DIR/results" "s3://$S3_BUCKET/$RUN_ID" --recursive

echo "=== Training Complete: $RUN_ID ==="
echo "Finished: $(date -u)"
echo "Results: s3://$S3_BUCKET/$RUN_ID"

# Self-terminate
shutdown -h now
