#!/bin/bash
# Training script with comprehensive metrics tracking

set -e

# Configuration
DATA_ROOT="./data/EB-Man_trajectory_dataset"
BATCH_SIZE=16  # Increased from 8 (tested: GPU can handle up to 32, using 16 for safety)
EPOCHS=50
LR=1e-4
EARLY_STOPPING_PATIENCE=5  # Stop if no improvement for 5 epochs
EARLY_STOPPING_MIN_DELTA=0.001  # Minimum improvement threshold
LOG_DIR="./logs"
CHECKPOINT_DIR="./checkpoints"
S3_BUCKET="11777-h1"
S3_PREFIX="metrics/"

echo "=========================================="
echo "Starting Training with Metrics Tracking"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Log directory: $LOG_DIR"
echo "=========================================="

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# Start training
echo ""
echo "Starting training with early stopping..."
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE epochs"
echo "Early stopping min delta: $EARLY_STOPPING_MIN_DELTA"
cd ~/EmbodiedMinds
PYTHONPATH=. python3 src/encoders/text_encoder.py \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
    2>&1 | tee "$LOG_DIR/training.log"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

# Find best checkpoint (preferred) or latest checkpoint
BEST_CHECKPOINT="$CHECKPOINT_DIR/agent_best.pt"
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/agent_epoch*.pt 2>/dev/null | head -1)

if [ -f "$BEST_CHECKPOINT" ]; then
    CHECKPOINT_TO_USE="$BEST_CHECKPOINT"
    echo "Using best checkpoint: $CHECKPOINT_TO_USE"
elif [ -n "$LATEST_CHECKPOINT" ]; then
    CHECKPOINT_TO_USE="$LATEST_CHECKPOINT"
    echo "Using latest checkpoint: $CHECKPOINT_TO_USE"
else
    echo "⚠️  No checkpoint found!"
    exit 1
fi
echo ""

# Run comprehensive trajectory evaluation
echo "=========================================="
echo "Running Trajectory Evaluation"
echo "=========================================="

PYTHONPATH=. PYTHONPATH=. python3 evaluate_trajectory.py \
    --checkpoint "$CHECKPOINT_TO_USE" \
    --data-root "$DATA_ROOT" \
    --max-episodes 100 \
    --success-threshold 0.8 \
    --log-dir "$LOG_DIR" \
    --s3-bucket "$S3_BUCKET" \
    --s3-prefix "$S3_PREFIX" \
    --dataset-type single_step \
    2>&1 | tee "$LOG_DIR/evaluation.log"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="

# Display summary
echo ""
echo "Metrics Summary:"
echo "----------------"
if [ -f "$LOG_DIR"/metrics_summary_*.csv ]; then
    cat "$LOG_DIR"/metrics_summary_*.csv | head -10
fi

echo ""
echo "All metrics saved to: $LOG_DIR"
echo "Checkpoint saved to: $LATEST_CHECKPOINT"

# Upload to S3 if configured
if [ -n "$S3_BUCKET" ]; then
    echo ""
    echo "Uploading metrics to S3..."
    aws s3 sync "$LOG_DIR" "s3://$S3_BUCKET/$S3_PREFIX" --region us-east-2 --exclude "*.log" || echo "⚠️  S3 upload failed (check IAM permissions)"
fi

echo ""
echo "=========================================="
echo "All Done!"
echo "=========================================="

