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
