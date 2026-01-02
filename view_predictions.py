#!/usr/bin/env python3
"""
View predictions vs targets from training or evaluation.

Usage:
    python3 view_predictions.py --epoch 0
    python3 view_predictions.py --file logs/predictions_epoch_0.npz
    python3 view_predictions.py --file evaluation_results_val.pt
"""
import argparse
import numpy as np
import torch
from pathlib import Path

def load_predictions_from_npz(filepath):
    """Load predictions from .npz file"""
    data = np.load(filepath, allow_pickle=True)
    predictions = data['predictions']
    targets = data['targets']
    instructions = data.get('instructions', [])
    return predictions, targets, instructions

def load_predictions_from_pt(filepath):
    """Load predictions from .pt file (evaluation results)"""
    data = torch.load(filepath, map_location='cpu')
    predictions = data['predictions'].numpy()
    targets = data['targets'].numpy()
    return predictions, targets, []

def view_predictions(predictions, targets, instructions=None, num_samples=20, dim_names=None):
    """Display predictions vs targets"""
    if dim_names is None:
        dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    num_samples = min(num_samples, len(predictions))
    
    print("="*80)
    print(f"Predictions vs Targets (showing {num_samples} samples)")
    print("="*80)
    print()
    
    for i in range(num_samples):
        pred = predictions[i]
        tgt = targets[i]
        
        if instructions and i < len(instructions):
            print(f"Sample {i+1}: {instructions[i][:60]}...")
        else:
            print(f"Sample {i+1}:")
        
        print(f"  {'Dimension':<10} {'Predicted':<12} {'Expected':<12} {'Match':<8} {'Error':<10}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
        
        for dim_idx, dim_name in enumerate(dim_names):
            p = int(pred[dim_idx])
            t = int(tgt[dim_idx])
            match = "✓" if p == t else "✗"
            error = abs(p - t) if t != -1 else "N/A"
            
            if t == -1:
                print(f"  {dim_name:<10} {p:<12} {'N/A':<12} {'-':<8} {'-':<10}")
            else:
                print(f"  {dim_name:<10} {p:<12} {t:<12} {match:<8} {error:<10}")
        
        print()
    
    # Summary statistics
    print("="*80)
    print("Summary Statistics")
    print("="*80)
    
    mask = (targets != -1)
    for dim_idx, dim_name in enumerate(dim_names):
        valid = mask[:, dim_idx]
        if valid.sum() > 0:
            correct = (predictions[valid, dim_idx] == targets[valid, dim_idx]).sum()
            total = valid.sum()
            accuracy = correct / total
            errors = np.abs(predictions[valid, dim_idx] - targets[valid, dim_idx])
            mean_error = errors.mean()
            max_error = errors.max()
            
            print(f"{dim_name}:")
            print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
            print(f"  Mean Error: {mean_error:.2f}")
            print(f"  Max Error: {max_error}")
            print()

def main():
    parser = argparse.ArgumentParser(description='View predictions vs targets')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch number to view predictions from')
    parser.add_argument('--file', type=str, default=None,
                       help='Path to predictions file (.npz or .pt)')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to display')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Determine file path
    if args.file:
        filepath = Path(args.file)
    elif args.epoch is not None:
        filepath = Path(args.log_dir) / f"predictions_epoch_{args.epoch}.npz"
    else:
        # Find latest predictions file
        log_dir = Path(args.log_dir)
        prediction_files = sorted(log_dir.glob("predictions_epoch_*.npz"))
        if not prediction_files:
            print("No prediction files found!")
            print(f"Looked in: {log_dir}")
            return
        filepath = prediction_files[-1]
        print(f"Using latest predictions file: {filepath}")
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    # Load predictions
    print(f"Loading predictions from: {filepath}")
    if filepath.suffix == '.npz':
        predictions, targets, instructions = load_predictions_from_npz(filepath)
    elif filepath.suffix == '.pt':
        predictions, targets, instructions = load_predictions_from_pt(filepath)
    else:
        print(f"Unsupported file format: {filepath.suffix}")
        return
    
    print(f"Loaded {len(predictions)} predictions")
    print()
    
    # Display
    view_predictions(predictions, targets, instructions, num_samples=args.num_samples)

if __name__ == "__main__":
    main()

