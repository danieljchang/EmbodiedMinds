#!/bin/bash
# Deploy code to AWS and start training with BERT implementation (batch size 8)

set -e

# Configuration
EC2_IP="18.217.28.52"
EC2_USER="ec2-user"
KEY_PATH="/Users/abhivakil/Desktop/11777.pem"
PROJECT_DIR="~/EmbodiedMinds"

echo "=========================================="
echo "Deploying to AWS EC2"
echo "=========================================="
echo "Instance: $EC2_USER@$EC2_IP"
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Sync code to EC2 (excluding large files and git)
echo "📦 Syncing code to EC2..."
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='venv/' \
    --exclude='*.egg-info/' \
    -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
    . "$EC2_USER@$EC2_IP:$PROJECT_DIR/"

echo ""
echo "✓ Code synced!"
echo ""

# Step 2: Create training script for EC2
echo "📝 Creating training script..."
cat > /tmp/run_training_aws.sh << 'TRAINING_SCRIPT'
#!/bin/bash
# Training script to run on EC2

set -e

cd ~/EmbodiedMinds

echo "=========================================="
echo "Starting Training on AWS EC2"
echo "=========================================="

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Verify CUDA
echo "CUDA Status:"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Create necessary directories
mkdir -p logs checkpoints

# Run training with BERT implementation (no VLM)
echo "Starting training with BERT encoder (batch size 8)..."
echo "Configuration:"
echo "  - Batch Size: 8"
echo "  - Text Encoder: BERT (frozen)"
echo "  - 3D Preprocessing: Enabled"
echo "  - Early Stopping: Enabled (patience=5)"
echo ""

# Run training (no VLM adapter = BERT mode)
PYTHONPATH=. python3 src/encoders/text_encoder.py \
    --data-root ./data/EB-Man_trajectory_dataset \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 0.001 \
    2>&1 | tee logs/training_aws.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

# Find best checkpoint
BEST_CHECKPOINT="checkpoints/agent_best.pt"
if [ -f "$BEST_CHECKPOINT" ]; then
    echo "Best checkpoint: $BEST_CHECKPOINT"
    
    # Run evaluation
    echo ""
    echo "=========================================="
    echo "Running Trajectory Evaluation"
    echo "=========================================="
    
    PYTHONPATH=. python3 evaluate_trajectory.py \
        --checkpoint "$BEST_CHECKPOINT" \
        --data-root ./data/EB-Man_trajectory_dataset \
        --max-episodes 100 \
        --success-threshold 0.8 \
        --log-dir ./logs \
        --s3-bucket 11777-h1 \
        --s3-prefix metrics/ \
        --dataset-type single_step \
        2>&1 | tee logs/evaluation_aws.log
    
    echo ""
    echo "Evaluation complete!"
    
    # Upload to S3
    echo ""
    echo "Uploading results to S3..."
    aws s3 sync ./logs s3://11777-h1/metrics/ --region us-east-2 --exclude "*.log" || echo "⚠️  S3 upload failed"
    aws s3 sync ./checkpoints s3://11777-h1/checkpoints/ --region us-east-2 || echo "⚠️  S3 upload failed"
fi

echo ""
echo "All done! Check logs/training_aws.log for details."
echo ""
TRAINING_SCRIPT

# Upload training script to EC2
echo "📤 Uploading training script..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no /tmp/run_training_aws.sh "$EC2_USER@$EC2_IP:$PROJECT_DIR/"
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" "chmod +x $PROJECT_DIR/run_training_aws.sh"

echo "✓ Training script uploaded!"
echo ""

# Step 3: Start training in tmux session
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""
echo "🚀 Launching training in tmux session..."
echo ""

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" << 'REMOTE_CMD'
cd ~/EmbodiedMinds

# Kill existing training session if any
tmux kill-session -t training 2>/dev/null || true

# Start new tmux session with training
tmux new-session -d -s training
tmux send-keys -t training "./run_training_aws.sh" C-m

echo "✓ Training started in tmux session 'training'"
echo ""
echo "To monitor progress:"
echo "  ssh -i /Users/abhivakil/Desktop/11777.pem ec2-user@18.217.28.52"
echo "  tmux attach -t training    # Attach to training session"
echo "  tail -f ~/EmbodiedMinds/logs/training_aws.log    # View logs"
echo ""
echo "To detach from tmux: Ctrl+B, then D"
REMOTE_CMD

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Training is now running on AWS in the background."
echo ""
echo "📊 To monitor:"
echo "  ssh -i $KEY_PATH $EC2_USER@$EC2_IP"
echo "  tmux attach -t training"
echo ""
echo "📁 Files will be saved to:"
echo "  - Logs: ~/EmbodiedMinds/logs/"
echo "  - Checkpoints: ~/EmbodiedMinds/checkpoints/"
echo "  - S3: s3://11777-h1/metrics/ and s3://11777-h1/checkpoints/"
echo ""

