#!/usr/bin/env python3
"""
Comprehensive trajectory evaluation script.

Evaluates model on complete task trajectories and computes:
1. Task Success Rate
2. Subgoal Success Rate
3. Planner Steps
4. Environment Steps
5. Error Analysis
"""
import argparse
import torch
import json
from pathlib import Path
from data_loader import build_dataloader, EmbodiedDataset
from src.models.agent_model import AgentModel
from src.utils.trajectory_evaluator import TrajectoryEvaluator
from src.utils.task_metrics import TaskMetricsTracker


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on task trajectories')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./data/EB-Man_trajectory_dataset',
                        help='Path to data directory')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to evaluate (None = all)')
    parser.add_argument('--success-threshold', type=float, default=0.8,
                        help='Threshold for considering task successful (0-1)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory to save metrics')
    parser.add_argument('--s3-bucket', type=str, default=None,
                        help='S3 bucket for uploading metrics (optional)')
    parser.add_argument('--s3-prefix', type=str, default='metrics/',
                        help='S3 prefix for metrics files')
    parser.add_argument('--dataset-type', type=str, default='single_step',
                        choices=['single_step', 'multi_step'],
                        help='Dataset type to use')
    
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
    model.eval()
    
    # Load dataset (use test split for evaluation)
    print(f"Loading dataset from: {args.data_root} (test split)")
    dataset = EmbodiedDataset(
        data_root=args.data_root,
        debug=False,
        dataset_type=args.dataset_type,
        split="test",  # Use held-out test set (10% of data)
        seed=42
    )
    print(f"Dataset loaded: {len(dataset)} test episodes")
    
    # Initialize metrics tracker
    metrics_tracker = TaskMetricsTracker(
        log_dir=args.log_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )
    
    # Initialize evaluator
    evaluator = TrajectoryEvaluator(
        model=model,
        dataset=dataset,
        device=device,
        metrics_tracker=metrics_tracker
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Starting Trajectory Evaluation")
    print("="*60)
    
    results = evaluator.evaluate_all(
        max_episodes=args.max_episodes,
        success_threshold=args.success_threshold
    )
    
    # Print summary
    summary = results.get('summary', {})
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if summary:
        print(f"Total Tasks Evaluated: {summary.get('total_tasks', 0)}")
        print(f"Successful Tasks: {summary.get('successful_tasks', 0)}")
        print(f"Failed Tasks: {summary.get('failed_tasks', 0)}")
        print(f"\nTask Success Rate: {summary.get('task_success_rate', 0.0):.2%}")
        print(f"Avg Subgoal Success Rate: {summary.get('avg_subgoal_success_rate', 0.0):.2%}")
        print(f"Avg Planner Steps: {summary.get('avg_planner_steps', 0.0):.2f}")
        print(f"Avg Environment Steps: {summary.get('avg_environment_steps', 0.0):.2f}")
        print(f"\nTotal Errors: {summary.get('total_errors', 0)}")
        
        error_analysis = summary.get('error_analysis', {})
        if error_analysis:
            print("\nError Analysis:")
            for error_type, count in error_analysis.items():
                print(f"  {error_type}: {count}")
    else:
        print("No tasks were successfully evaluated.")
        print("Check error messages above for details.")
    
    print("="*60)
    
    # Save metrics
    print("\nSaving metrics...")
    metrics_file = metrics_tracker.save_metrics(upload_to_s3=(args.s3_bucket is not None))
    csv_file = metrics_tracker.save_summary_csv()
    
    # Save detailed results
    results_file = Path(args.log_dir) / f"trajectory_evaluation_{Path(args.checkpoint).stem}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

