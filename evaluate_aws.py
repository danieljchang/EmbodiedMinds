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

