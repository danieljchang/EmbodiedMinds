# AWS Training & Evaluation Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Infrastructure Setup](#aws-infrastructure-setup)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Training Steps](#training-steps)
6. [Evaluation Steps](#evaluation-steps)
7. [Monitoring & Logging](#monitoring--logging)
8. [Cost Optimization](#cost-optimization)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Setup (Before AWS)
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- SSH key pair for EC2 access
- Basic understanding of EC2, S3, and CloudWatch

### Required Knowledge
- PyTorch and deep learning basics
- Linux command line
- Git version control

---

## AWS Infrastructure Setup

### 1. EC2 Instance Selection

#### Recommended Instance Types

**For Development/Testing:**
- **Instance Type:** `g4dn.xlarge` or `g4dn.2xlarge`
- **GPU:** 1x NVIDIA T4 (16GB)
- **vCPU:** 4-8
- **Memory:** 16-32 GB
- **Cost:** ~$0.50-1.00/hour
- **Use Case:** Initial testing, debugging, small experiments

**For Training:**
- **Instance Type:** `g5.xlarge` or `g5.2xlarge` (recommended)
- **GPU:** 1x NVIDIA A10G (24GB)
- **vCPU:** 4-8
- **Memory:** 16-32 GB
- **Cost:** ~$1.00-2.00/hour
- **Use Case:** Full training runs, production training

**For Large-Scale Training:**
- **Instance Type:** `p3.2xlarge` or `p4d.24xlarge`
- **GPU:** 1x V100 (16GB) or 8x A100 (40GB)
- **vCPU:** 8-96
- **Memory:** 61-1152 GB
- **Cost:** ~$3.00-32.00/hour
- **Use Case:** Large datasets, hyperparameter tuning

#### Instance Configuration Checklist
```bash
# Minimum Requirements
- GPU: NVIDIA GPU with CUDA support
- CUDA: Version 11.8 or 12.1
- Storage: 100GB+ (for data, models, checkpoints)
- Network: High bandwidth for data transfer
```

### 2. Create EC2 Instance

#### Step-by-Step AWS Console Setup

1. **Launch EC2 Instance:**
   - Go to EC2 Dashboard → Launch Instance
   - Name: `embodied-minds-training`
   - AMI: **Deep Learning AMI (Ubuntu)** - Latest version
     - Search: "Deep Learning AMI GPU PyTorch"
     - Select: `Deep Learning AMI GPU PyTorch 2.x (Ubuntu 20.04)`
   - Instance Type: `g5.xlarge` (or your choice)
   - Key Pair: Select or create new SSH key pair
   - Network Settings: Allow SSH (port 22) from your IP
   - Storage: 100GB gp3 SSD (expandable)

2. **Configure Security Group:**
   ```
   Inbound Rules:
   - SSH (22): Your IP
   - Custom TCP (8888): Your IP (for Jupyter)
   - Custom TCP (6006): Your IP (for TensorBoard)
   ```

3. **Launch and Note:**
   - Public IP address
   - Instance ID
   - Key pair file location

#### Alternative: AWS CLI Setup

```bash
# Create security group
aws ec2 create-security-group \
    --group-name embodied-minds-sg \
    --description "Security group for training"

# Add SSH rule
aws ec2 authorize-security-group-ingress \
    --group-name embodied-minds-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0  # Restrict to your IP in production

# Launch instance
aws ec2 run-instances \
    --image-id ami-xxxxx \  # Deep Learning AMI ID
    --instance-type g5.xlarge \
    --key-name your-key-pair \
    --security-groups embodied-minds-sg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'
```

### 3. S3 Bucket Setup

#### Create S3 Buckets

```bash
# Training data bucket
aws s3 mb s3://embodied-minds-data --region us-east-1

# Model checkpoints bucket
aws s3 mb s3://embodied-minds-checkpoints --region us-east-1

# Results/logs bucket
aws s3 mb s3://embodied-minds-results --region us-east-1
```

#### S3 Bucket Structure
```
embodied-minds-data/
├── raw/
│   ├── train/
│   ├── val/
│   └── test/
└── processed/

embodied-minds-checkpoints/
├── models/
│   ├── epoch_0.pt
│   ├── epoch_10.pt
│   └── best_model.pt
└── configs/

embodied-minds-results/
├── logs/
├── tensorboard/
└── evaluations/
```

### 4. IAM Roles & Permissions

#### Create IAM Role for EC2

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::embodied-minds-data/*",
        "arn:aws:s3:::embodied-minds-checkpoints/*",
        "arn:aws:s3:::embodied-minds-results/*",
        "arn:aws:s3:::embodied-minds-data",
        "arn:aws:s3:::embodied-minds-checkpoints",
        "arn:aws:s3:::embodied-minds-results"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "cloudwatch:GetMetricStatistics"
      ],
      "Resource": "*"
    }
  ]
}
```

Attach role to EC2 instance:
```bash
aws ec2 associate-iam-instance-profile \
    --instance-id i-xxxxx \
    --iam-instance-profile Name=embodied-minds-role
```

---

## Environment Setup

### 1. Connect to EC2 Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Verify GPU
nvidia-smi

# Should show NVIDIA GPU information
```

### 2. Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone repository
git clone https://github.com/0xlel0uch/EmbodiedMinds.git
cd EmbodiedMinds

# Checkout your branch
git checkout Abhi_vakil_completed_code
```

### 3. Install Dependencies

#### System Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install additional tools
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    screen
```

#### Python Environment

```bash
# Create virtual environment (if not using conda)
python3 -m venv venv
source venv/bin/activate

# Or use conda (recommended, comes with Deep Learning AMI)
conda create -n embodied python=3.9
conda activate embodied
```

#### Install Python Packages

```bash
# Navigate to project directory
cd ~/EmbodiedMinds

# Install core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install ultralytics  # YOLOv8
pip install timm         # MiDaS support
pip install transformers # BERT
pip install numpy
pip install opencv-python-headless
pip install pillow
pip install tqdm
pip install tensorboard  # For logging
pip install wandb        # Optional: Weights & Biases
pip install boto3        # AWS SDK
pip install s3fs         # S3 file system

# Install project in development mode
pip install -e .
```

#### Verify Installation

```bash
# Test CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Test imports
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python -c "import torch.hub; print('MiDaS OK')"
```

### 4. Download Pre-trained Models

```bash
# YOLOv8 will auto-download on first use
# MiDaS will auto-download on first use
# BERT will auto-download on first use

# Pre-download to avoid delays during training
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "import torch.hub; torch.hub.load('intel-isl/MiDaS', 'DPT_Large')"
```

---

## Data Preparation

### 1. Upload Data to S3

#### From Local Machine

```bash
# Upload training data
aws s3 sync ./data/train s3://embodied-minds-data/raw/train/

# Upload validation data
aws s3 sync ./data/val s3://embodied-minds-data/raw/val/

# Upload test data (if available)
aws s3 sync ./data/test s3://embodied-minds-data/raw/test/
```

#### From EC2 Instance

```bash
# Download data from S3
aws s3 sync s3://embodied-minds-data/raw/train/ ~/EmbodiedMinds/data/train/
aws s3 sync s3://embodied-minds-data/raw/val/ ~/EmbodiedMinds/data/val/
```

### 2. Data Structure

Ensure your data follows this structure:
```
data/
├── train/
│   ├── example_001.json
│   ├── example_002.json
│   └── ...
├── val/
│   ├── example_001.json
│   └── ...
└── test/  # Optional
    └── ...
```

### 3. Verify Data

```bash
# Check data structure
python -c "
from data_loader import EmbodiedDataset
ds = EmbodiedDataset(data_root='./data', debug=True)
print(f'Dataset size: {len(ds)}')
item = ds[0]
print(f'Keys: {item.keys()}')
"
```

---

## Training Steps

### 1. Create Training Script

Create `train_aws.py`:

```python
#!/usr/bin/env python3
"""
Training script optimized for AWS EC2
"""
import os
import sys
import argparse
import torch
import boto3
from pathlib import Path
from src.encoders.text_encoder import train

def upload_checkpoint_to_s3(local_path, s3_bucket, s3_key):
    """Upload checkpoint to S3"""
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, s3_bucket, s3_key)
        print(f"✓ Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"✗ Failed to upload: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train EmbodiedMinds model on AWS')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--s3-checkpoint-bucket', type=str,
                        default='embodied-minds-checkpoints',
                        help='S3 bucket for checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train model
    print("Starting training...")
    train(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_3d_preprocessing=True
    )
    
    # Upload final checkpoint
    checkpoint_files = list(Path("checkpoints").glob("*.pt"))
    for ckpt_file in checkpoint_files:
        s3_key = f"models/{ckpt_file.name}"
        upload_checkpoint_to_s3(
            str(ckpt_file),
            args.s3_checkpoint_bucket,
            s3_key
        )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
```

### 2. Run Training

#### Basic Training

```bash
# Activate environment
conda activate embodied  # or source venv/bin/activate

# Run training
python train_aws.py \
    --data-root ./data \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --s3-checkpoint-bucket embodied-minds-checkpoints
```

#### Training with Screen/Tmux (Recommended)

```bash
# Start screen session
screen -S training

# Run training
python train_aws.py --batch-size 8 --epochs 50

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

#### Training with NoHup

```bash
# Run in background
nohup python train_aws.py \
    --batch-size 8 \
    --epochs 50 \
    > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### 3. Monitor Training

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=./logs --port=6006 --host=0.0.0.0

# Access from local machine
# SSH tunnel: ssh -i key.pem -L 6006:localhost:6006 ubuntu@<EC2_IP>
# Browser: http://localhost:6006
```

#### Weights & Biases (Optional)

```bash
# Login to wandb
wandb login

# Training will automatically log to wandb
```

#### CloudWatch Metrics

```python
# Add to training script
import boto3
cloudwatch = boto3.client('cloudwatch')

def log_metric(metric_name, value, epoch):
    cloudwatch.put_metric_data(
        Namespace='EmbodiedMinds/Training',
        MetricData=[{
            'MetricName': metric_name,
            'Value': value,
            'Unit': 'None',
            'Dimensions': [
                {'Name': 'Epoch', 'Value': str(epoch)}
            ]
        }]
    )
```

---

## Evaluation Steps

### 1. Create Evaluation Script

Create `evaluate_aws.py`:

```python
#!/usr/bin/env python3
"""
Evaluation script for trained models
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from data_loader import build_dataloader
from src.models.agent_model import AgentModel
from src.heads.output_heads import OutputHeads

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            instructions = batch['instructions']
            demo_3d_objects = batch['demo_3d_objects']
            current_3d_objects = batch['current_3d_objects']
            demo_actions = batch.get('demo_actions', None)
            targets = batch['targets'].to(device)
            
            # Forward pass
            logits = model.forward(
                instructions,
                demo_3d_objects,
                current_3d_objects,
                demo_actions
            )
            
            # Compute loss
            loss = model.heads.loss(logits, targets)
            total_loss += loss.item()
            
            # Get predictions
            preds = model.heads.predict(logits).cpu()
            all_predictions.append(preds)
            all_targets.append(targets.cpu())
            
            num_batches += 1
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Accuracy per dimension
    mask = (all_targets != -1)
    accuracies = []
    for i in range(7):
        valid = mask[:, i]
        if valid.sum() > 0:
            acc = (all_predictions[valid, i] == all_targets[valid, i]).float().mean()
            accuracies.append(acc.item())
        else:
            accuracies.append(0.0)
    
    avg_loss = total_loss / num_batches
    avg_accuracy = np.mean(accuracies)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'per_dim_accuracy': accuracies,
        'predictions': all_predictions,
        'targets': all_targets
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate EmbodiedMinds model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    bins = checkpoint.get('bins', [101, 101, 101, 121, 121, 121, 2])
    model = AgentModel(bins=bins, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    # Load data
    dataloader = build_dataloader(
        batch_size=args.batch_size,
        debug=False,
        data_root=args.data_root,
        num_workers=0,
        use_3d_preprocessing=True,
        device=device
    )
    
    # Evaluate
    print("Evaluating...")
    results = evaluate_model(model, dataloader, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Average Loss: {results['loss']:.4f}")
    print(f"Average Accuracy: {results['accuracy']:.4f}")
    print("\nPer-Dimension Accuracy:")
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i, (name, acc) in enumerate(zip(dim_names, results['per_dim_accuracy'])):
        print(f"  {name}: {acc:.4f}")
    print("="*50)
    
    # Save results
    results_file = f"evaluation_results_{args.split}.pt"
    torch.save(results, results_file)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
```

### 2. Run Evaluation

```bash
# Download checkpoint from S3
aws s3 cp s3://embodied-minds-checkpoints/models/agent_epoch50.pt ./checkpoints/

# Run evaluation
python evaluate_aws.py \
    --checkpoint ./checkpoints/agent_epoch50.pt \
    --data-root ./data \
    --batch-size 8 \
    --split val
```

### 3. Upload Results

```bash
# Upload evaluation results to S3
aws s3 cp evaluation_results_val.pt \
    s3://embodied-minds-results/evaluations/
```

---

## Monitoring & Logging

### 1. System Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor network
iftop
```

### 2. Training Logs

```bash
# View training logs
tail -f training.log

# Search for errors
grep -i error training.log

# Monitor loss
grep "loss" training.log | tail -20
```

### 3. CloudWatch Integration

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure (interactive)
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

---

## Cost Optimization

### 1. Spot Instances

```bash
# Request spot instance (60-90% cheaper)
aws ec2 request-spot-instances \
    --spot-price "0.50" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://spot-specification.json
```

### 2. Auto-Stop on Completion

```python
# Add to training script
import boto3

def stop_instance_on_completion():
    """Stop EC2 instance when training completes"""
    instance_id = requests.get('http://169.254.169.254/latest/meta-data/instance-id').text
    ec2 = boto3.client('ec2')
    ec2.stop_instances(InstanceIds=[instance_id])
    print("Instance stopped")
```

### 3. Cost Estimation

**Example Training Costs:**
- **g5.xlarge:** $1.00/hour
- **Training time:** 24 hours
- **Total cost:** ~$24
- **With spot:** ~$6-12

### 4. S3 Storage Costs

- **Data storage:** ~$0.023/GB/month
- **Checkpoints:** ~$0.023/GB/month
- **Data transfer:** First 100GB free, then $0.09/GB

---

## Setup Status

### Current Instance
- **Public IP:** 3.139.95.113
- **Username:** ec2-user
- **GPU:** NVIDIA A10G (23.68 GB)
- **CUDA:** 12.1
- **S3 Bucket:** `11777-h1`
- **Region:** `us-east-2`

### Quick Connect
```bash
ssh -i /Users/abhivakil/Desktop/11777.pem ec2-user@3.139.95.113
cd ~/EmbodiedMinds
```

### S3 Setup
The instance should have an IAM role attached with S3 access. Verify:
```bash
aws sts get-caller-identity
aws s3 ls s3://11777-h1/ --region us-east-2
```

## Training Optimization

### Current Settings
- **Batch Size:** 16 (optimized from 8)
- **Early Stopping:** Enabled (patience=5)
- **Expected Time:** ~1-2.5 hours with early stopping

### GPU Memory
- **Total:** 23.68 GB
- **Used (batch_size=16):** ~2.6-3 GB
- **Available:** ~21 GB free
- **Can safely use:** batch_size up to 32 if needed

### Speed Improvements
- **batch_size=8:** ~30-40 min/epoch
- **batch_size=16:** ~15-20 min/epoch (2x faster)
- **With early stopping:** Typically stops after 10-15 epochs

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# Reduce batch size
python train_aws.py --batch-size 4

# Use gradient accumulation
# (modify training script)
```

#### 2. Slow Data Loading

```bash
# Pre-process and cache data
# Use faster storage (NVMe)
# Increase num_workers (if CPU allows)
```

#### 3. Connection Timeout

```bash
# Use screen/tmux for persistent sessions
screen -S training
# Or use nohup
```

#### 4. S3 Upload Failures

```bash
# Check IAM permissions
aws s3 ls s3://embodied-minds-checkpoints/

# Retry with multipart upload
aws configure set default.s3.multipart_threshold 64MB
```

---

## Quick Start Checklist

- [ ] Create EC2 instance (g5.xlarge recommended)
- [ ] Configure security groups
- [ ] Create S3 buckets
- [ ] Set up IAM roles
- [ ] SSH into instance
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Upload data to S3
- [ ] Download data to EC2
- [ ] Verify GPU access
- [ ] Run training script
- [ ] Monitor training progress
- [ ] Evaluate model
- [ ] Download results
- [ ] Stop/terminate instance

---

## Next Steps After Training

1. **Model Analysis:**
   - Review training curves
   - Analyze per-dimension accuracy
   - Identify failure cases

2. **Hyperparameter Tuning:**
   - Learning rate scheduling
   - Batch size optimization
   - Architecture modifications

3. **Deployment:**
   - Export model for inference
   - Create inference API
   - Deploy to production

4. **Documentation:**
   - Document best practices
   - Create inference guide
   - Share results

---

## Support & Resources

- **AWS Documentation:** https://docs.aws.amazon.com/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **Project Repository:** https://github.com/0xlel0uch/EmbodiedMinds
- **Issues:** Create GitHub issue for bugs/questions

---

**Last Updated:** November 2024  
**Maintained By:** Abhi Vakil

