#!/bin/bash
# =============================================================================
# RoadSense Training AMI Setup Script
# =============================================================================
# Run this ONCE on a fresh Ubuntu 22.04 EC2 instance via SSH.
# After completion, create an AMI snapshot from the AWS console.
#
# Usage:
#   1. Launch c6i.xlarge from Ubuntu 22.04 LTS in il-central-1
#   2. SSH in: ssh -i key.pem ubuntu@<IP>
#   3. Run:    bash setup_ami.sh <GITHUB_PAT>
#   4. Verify output, then create AMI from console
# =============================================================================

set -euo pipefail

PAT="${1:?Usage: bash setup_ami.sh <GITHUB_PAT>}"
REPO_URL="https://${PAT}@github.com/roadsense-team/roadsense-v2v.git"
WORK_DIR="/home/ubuntu/work"

echo "========================================="
echo " RoadSense AMI Setup"
echo "========================================="

# 1. System packages
echo "[1/5] Installing system packages..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends docker.io awscli git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu

# 2. Clone repo
echo "[2/5] Cloning repository..."
if [ -d "$WORK_DIR" ]; then
    echo "  -> $WORK_DIR already exists, pulling latest..."
    cd "$WORK_DIR" && git pull origin master
else
    git clone "$REPO_URL" "$WORK_DIR"
fi

# 3. Build Docker image (the slow part we're baking in)
echo "[3/5] Building Docker image (this takes 5-8 minutes)..."
cd "$WORK_DIR/ml"
sudo docker build -t roadsense-ml:latest .

# 4. Clean PAT from git remote (SECURITY: don't bake credentials into AMI)
echo "[4/5] Cleaning credentials from git remote..."
cd "$WORK_DIR"
git remote set-url origin https://github.com/roadsense-team/roadsense-v2v.git

# 5. Verify
echo "[5/5] Verifying installation..."
echo ""
echo "--- Docker image ---"
sudo docker images | grep roadsense-ml
echo ""
echo "--- SUMO version ---"
sudo docker run --rm roadsense-ml:latest sumo --version
echo ""
echo "--- ML stack ---"
sudo docker run --rm roadsense-ml:latest python -c "import gymnasium; import stable_baselines3; print('ML stack: OK')"
echo ""

echo "========================================="
echo " AMI Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Go to AWS Console -> EC2 -> Instances"
echo "  2. Select this instance -> Actions -> Image -> Create Image"
echo "  3. Name: roadsense-training-v1"
echo "  4. Wait for AMI to become 'available' (~5-10 min)"
echo "  5. Terminate this builder instance"
echo ""
echo "Or via CLI:"
echo "  aws ec2 create-image --instance-id <ID> --name roadsense-training-v1 --region il-central-1"
