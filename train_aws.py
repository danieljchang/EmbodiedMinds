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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
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

