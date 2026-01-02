# Metrics Documentation

Complete guide to metrics tracking, loss functions, and predictions.

## Overview

The system tracks comprehensive task-level metrics including Task Success Rate, Subgoal Success Rate, Planner Steps, Environment Steps, and Error Analysis. All metrics are automatically saved during and after training.

## Tracked Metrics

### 1. Task Success Rate
**Definition**: The percentage of tasks where the agent successfully completes the full instruction.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.task_success_rate`
- CSV: `logs/metrics_summary_*.csv` → "Task Success Rate" row
- Format: Float (0.0 to 1.0)

### 2. Subgoal Success Rate
**Definition**: For high-level tasks, the fraction of intermediate goals completed correctly.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.avg_subgoal_success_rate`
- CSV: `logs/metrics_summary_*.csv` → "Avg Subgoal Success Rate" row
- Format: Float (0.0 to 1.0)

### 3. Planner Steps
**Definition**: The number of model inferences (planning calls) required to produce a complete executable plan.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.avg_planner_steps`
- CSV: `logs/metrics_summary_*.csv` → "Avg Planner Steps" row
- Format: Integer

### 4. Environment Steps
**Definition**: The number of interactions the agent performs within the environment while executing a task.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.avg_environment_steps`
- CSV: `logs/metrics_summary_*.csv` → "Avg Environment Steps" row
- Format: Integer

### 5. Error Analysis
**Definition**: Qualitative categorization of failures (e.g., perception errors, reasoning errors, or planning errors).

**Error Types**:
- `perception`: Errors in position (x, y, z coordinates)
- `orientation`: Errors in rotation (roll, pitch, yaw)
- `gripper`: Errors in gripper state
- `planning`: Multiple dimension errors (planning failure)
- `invalid_target`: Invalid target values

**Storage**:
- Location: `logs/metrics_*.json` → `summary.error_analysis`
- CSV: `logs/metrics_summary_*.csv` → "Error Type" and "Count" columns

## Loss Function

### Type: **CrossEntropyLoss** (Multi-class Classification)

The model uses **CrossEntropyLoss** for each of the 7 action dimensions independently.

**Location**: `src/heads/output_heads.py`

**Key Features**:
1. **Per-Dimension Loss**: Each of the 7 action dimensions (x, y, z, roll, pitch, yaw, gripper) has its own classification head
2. **Ignore Invalid Targets**: Uses `ignore_index=-1` to ignore invalid/missing targets
3. **Target Validation**: Automatically clamps out-of-range targets to -1 (invalid)
4. **Averaged Loss**: Returns the average loss across all 7 dimensions

**Action Dimensions and Bins**:
- **x, y, z**: 101 bins each (range: 0-100)
- **roll, pitch, yaw**: 121 bins each (range: 0-120)
- **gripper**: 2 bins (0=closed, 1=open)

## Metrics Storage

### Local Storage (EC2 Instance)

All metrics are stored in the `logs/` directory:

```
~/EmbodiedMinds/
├── logs/
│   ├── training_metrics_epoch_*.json      # Per-epoch training metrics
│   ├── predictions_epoch_*.npz            # Predictions vs targets
│   ├── metrics_YYYYMMDD_HHMMSS.json       # Comprehensive task metrics
│   ├── metrics_summary_YYYYMMDD_HHMMSS.csv # Summary CSV
│   └── trajectory_evaluation_*.json       # Detailed evaluation
```

### S3 Storage (Optional)

If configured, metrics are automatically uploaded to S3:

```
s3://11777-h1/metrics/
├── metrics_YYYYMMDD_HHMMSS.json
├── metrics_summary_YYYYMMDD_HHMMSS.csv
└── trajectory_evaluation_*.json
```

## When Metrics Are Saved

### During Training (Per Epoch)

**Files**:
- `logs/training_metrics_epoch_{N}.json` - Basic accuracy metrics
- `logs/predictions_epoch_{N}.npz` - Predictions vs targets

**Contains**:
- Train loss, validation loss
- Per-dimension accuracy (x, y, z, roll, pitch, yaw, gripper)
- Average accuracy
- Predictions and targets arrays

**Note**: The 5 comprehensive metrics are NOT saved during training (they require full trajectory evaluation).

### After Training (Automatic)

**Files**:
- `logs/metrics_YYYYMMDD_HHMMSS.json` - Full metrics with all 5 comprehensive metrics
- `logs/metrics_summary_YYYYMMDD_HHMMSS.csv` - Summary CSV
- `logs/trajectory_evaluation_*.json` - Detailed evaluation results

**Saved automatically** by `train_with_metrics.sh` which runs `evaluate_trajectory.py` after training completes.

## Viewing Metrics

### Quick View (CSV Summary)
```bash
cat logs/metrics_summary_*.csv
```

### Detailed View (JSON)
```bash
cat logs/metrics_*.json | python3 -m json.tool | less
```

### View Predictions
```bash
# View latest predictions
python3 view_predictions.py

# View specific epoch
python3 view_predictions.py --epoch 5

# View more samples
python3 view_predictions.py --epoch 0 --num-samples 50
```

### Verify All Metrics
```bash
python3 verify_metrics.py
```

### Extract Specific Metric
```python
import json
import glob

files = glob.glob('logs/metrics_*.json')
if files:
    with open(files[-1]) as f:
        data = json.load(f)
        s = data['summary']
        print(f"Task Success Rate: {s['task_success_rate']:.2%}")
        print(f"Subgoal Success Rate: {s['avg_subgoal_success_rate']:.2%}")
        print(f"Planner Steps: {s['avg_planner_steps']:.2f}")
        print(f"Environment Steps: {s['avg_environment_steps']:.2f}")
        print(f"Errors: {s['error_analysis']}")
```

## Prediction Format

### Action Vector (7 dimensions)

Each prediction/target is a 7-dimensional vector:

```python
[x, y, z, roll, pitch, yaw, gripper]
```

**Example**:
```python
predicted = [45, 32, 78, 119, 55, 98, 1]
expected  = [45, 32, 78, 120, 55, 98, 1]
#           ✓   ✓   ✓   ✗    ✓   ✓   ✓
```

### Dimension Meanings

1. **x, y, z** (0-100): 3D position coordinates
   - Normalized to [0, 100] range
   - Each bin represents ~1% of the workspace

2. **roll, pitch, yaw** (0-120): Rotation angles
   - Each bin represents 3 degrees (120 bins × 3° = 360°)
   - Normalized to [0, 120] range

3. **gripper** (0-1): Gripper state
   - 0 = Closed
   - 1 = Open

## Troubleshooting

### Metrics not saving?
- Check `logs/` directory exists and is writable
- Verify disk space: `df -h`

### S3 upload failing?
- Check IAM role has S3 write permissions
- Verify bucket name: `11777-h1`
- Check region: `us-east-2`

### Missing metrics?
- Ensure evaluation script completed successfully
- Check for errors in console output
- Verify dataset loaded correctly

## Summary

✅ **All 5 requested metrics ARE being tracked and saved**
✅ **Saved automatically after training completes**
✅ **Available in both JSON and CSV formats**
✅ **Uploaded to S3 automatically**

**Note**: The comprehensive metrics are calculated during trajectory evaluation (after training), not during training validation. This is because they require full task trajectory evaluation, not just single-step predictions.

