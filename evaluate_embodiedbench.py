#!/usr/bin/env python3
"""
EmbodiedBench-style evaluation script.

Computes metrics matching the EmbodiedBench paper:
1. Task Success Rate (primary metric)
   - Overall
   - Per capability subset: Base, Common Sense, Complex, Visual, Spatial, Long-Horizon
2. Subgoal Success Rate (for high-level tasks)
3. Average Planner Steps
4. Average Environment Steps
5. Error Analysis
"""
import argparse
import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd

from data_loader import build_dataloader, EmbodiedDataset, collate_fn_3d
from src.models.agent_model import AgentModel
from src.utils.task_metrics import TaskMetricsTracker, categorize_error


def extract_capability_subset(img_path: str) -> str:
    """
    Extract capability subset from image path.
    
    Path format: "images/claude-3-5-sonnet-20241022/base/episode_1/step_0.png"
    or: "claude-3-5-sonnet-20241022/base/episode_1/step_0.png"
    Returns: "base", "common_sense", "complex", "visual", "spatial", or "unknown"
    """
    if not img_path:
        return "unknown"
    
    parts = img_path.split('/')
    # Find the capability subset (usually after model name)
    # Path structure: [images/]model_name/subset/episode_X/step_Y.png
    for i, part in enumerate(parts):
        part_lower = part.lower()
        # Check if this part is a known capability subset
        if part_lower in ['base', 'common_sense', 'complex', 'visual', 'spatial']:
            return part_lower
        # Handle variations
        if 'common' in part_lower or 'sense' in part_lower:
            return 'common_sense'
        if 'long' in part_lower or 'horizon' in part_lower:
            return 'long_horizon'
    
    return "unknown"


def compute_task_success(prediction: torch.Tensor, target: torch.Tensor, 
                         threshold: float = 0.8) -> bool:
    """
    Compute task success based on prediction accuracy.
    
    Args:
        prediction: (7,) tensor of predicted action
        target: (7,) tensor of target action
        threshold: Success threshold (0-1), default 0.8 means 80% of dimensions correct
    
    Returns:
        True if task is successful
    """
    valid_mask = (target != -1)
    if not valid_mask.any():
        return False
    
    correct = (prediction == target) & valid_mask
    accuracy = correct.sum().item() / valid_mask.sum().item()
    return accuracy >= threshold


def evaluate_embodiedbench_style(
    model: AgentModel,
    dataset: EmbodiedDataset,
    device: str = "cuda",
    success_threshold: float = 0.8,
    max_episodes: Optional[int] = None
) -> Dict:
    """
    Evaluate model in EmbodiedBench style.
    
    Returns:
        Dictionary with metrics matching EmbodiedBench paper format
    """
    model.eval()
    
    # Metrics storage
    results = {
        'overall': {
            'total_tasks': 0,
            'successful_tasks': 0,
            'task_success_rate': 0.0,
            'total_planner_steps': 0,
            'total_environment_steps': 0,
            'avg_planner_steps': 0.0,
            'avg_environment_steps': 0.0,
        },
        'by_capability': defaultdict(lambda: {
            'total_tasks': 0,
            'successful_tasks': 0,
            'task_success_rate': 0.0,
            'total_planner_steps': 0,
            'total_environment_steps': 0,
        }),
        'subgoal_success': {
            'total_subgoals': 0,
            'successful_subgoals': 0,
            'subgoal_success_rate': 0.0,
        },
        'error_analysis': defaultdict(int),
        'episode_results': [],
    }
    
    num_episodes = min(len(dataset), max_episodes) if max_episodes else len(dataset)
    print(f"\nEvaluating {num_episodes} episodes...")
    
    with torch.no_grad():
        for episode_idx in range(num_episodes):
            if (episode_idx + 1) % 50 == 0:
                print(f"  Progress: {episode_idx + 1}/{num_episodes}")
            
            try:
                # Get episode data
                item = dataset[episode_idx]
                instruction = item['instruction']
                episode_id = item.get('meta_path', f'episode_{episode_idx}')
                
                # Extract capability subset from dataset entry
                capability_subset = "unknown"
                if episode_idx < len(dataset.data):
                    raw_entry = dataset.data[episode_idx]
                    # First try eval_set field (most reliable)
                    eval_set = raw_entry.get('eval_set', '')
                    if eval_set:
                        capability_subset = eval_set.lower()
                    else:
                        # Fallback: extract from image path
                        if 'trajectory' in raw_entry:
                            traj = raw_entry['trajectory']
                            if traj and len(traj) > 0:
                                plan = traj[0].get('executable_plan', {})
                                img_path = plan.get('img_path', '')
                                if not img_path:
                                    # Try input_image_path
                                    img_path = traj[0].get('input_image_path', '')
                                capability_subset = extract_capability_subset(img_path)
                
                # Process data using collate_fn_3d
                batch = [item]
                processed_batch = collate_fn_3d(batch, device=device)
                
                instructions = processed_batch['instructions']
                demo_3d_objects = processed_batch['demo_3d_objects']
                current_3d_objects = processed_batch['current_3d_objects']
                demo_actions_list = processed_batch.get('demo_actions', None)
                targets = processed_batch.get('targets', None)
                
                # Run model inference
                logits = model.forward(
                    instructions,
                    demo_3d_objects,
                    current_3d_objects,
                    demo_actions_list
                )
                
                # Get predictions
                predictions = model.heads.predict(logits)
                pred = predictions[0]  # Single prediction
                
                # Get target (last action from trajectory)
                if targets is not None:
                    target = targets[0].to(device)
                else:
                    # Fallback: use last demo action
                    demo_actions = item.get('demo_actions', [])
                    if demo_actions and len(demo_actions) > 0:
                        demo_actions_tensor = demo_actions[0]
                        if demo_actions_tensor.shape[0] > 0:
                            valid = (demo_actions_tensor != -1).all(dim=1)
                            valid_idxs = valid.nonzero(as_tuple=False)
                            if len(valid_idxs) > 0:
                                target = demo_actions_tensor[valid_idxs[-1].item()].to(device)
                            else:
                                target = torch.full((7,), -1, dtype=torch.long, device=device)
                        else:
                            target = torch.full((7,), -1, dtype=torch.long, device=device)
                    else:
                        target = torch.full((7,), -1, dtype=torch.long, device=device)
                
                # Compute task success
                task_success = compute_task_success(pred, target, success_threshold)
                
                # Count environment steps (number of actions in trajectory)
                demo_actions = item.get('demo_actions', [])
                if demo_actions and len(demo_actions) > 0:
                    demo_actions_tensor = demo_actions[0]
                    environment_steps = demo_actions_tensor.shape[0] if demo_actions_tensor.shape[0] > 0 else 0
                else:
                    environment_steps = 0
                
                planner_steps = 1  # Single inference per episode for now
                
                # Update overall metrics
                results['overall']['total_tasks'] += 1
                if task_success:
                    results['overall']['successful_tasks'] += 1
                results['overall']['total_planner_steps'] += planner_steps
                results['overall']['total_environment_steps'] += environment_steps
                
                # Update capability-specific metrics
                results['by_capability'][capability_subset]['total_tasks'] += 1
                if task_success:
                    results['by_capability'][capability_subset]['successful_tasks'] += 1
                results['by_capability'][capability_subset]['total_planner_steps'] += planner_steps
                results['by_capability'][capability_subset]['total_environment_steps'] += environment_steps
                
                # Subgoal success (simplified: treat each step as a subgoal)
                valid_mask = (target != -1)
                if valid_mask.any():
                    correct_dims = (pred == target) & valid_mask
                    num_correct = correct_dims.sum().item()
                    num_valid = valid_mask.sum().item()
                    subgoal_success = num_correct / num_valid if num_valid > 0 else 0.0
                    
                    results['subgoal_success']['total_subgoals'] += num_valid
                    results['subgoal_success']['successful_subgoals'] += num_correct
                
                # Error analysis
                if not task_success:
                    error_type = categorize_error(pred, target, instruction, 0)
                    results['error_analysis'][error_type] += 1
                
                # Store episode result
                results['episode_results'].append({
                    'episode_id': episode_id,
                    'capability_subset': capability_subset,
                    'instruction': instruction,
                    'success': task_success,
                    'planner_steps': planner_steps,
                    'environment_steps': environment_steps,
                })
                
            except Exception as e:
                print(f"  Error evaluating episode {episode_idx}: {e}")
                continue
    
    # Compute final rates
    overall = results['overall']
    if overall['total_tasks'] > 0:
        overall['task_success_rate'] = overall['successful_tasks'] / overall['total_tasks']
        overall['avg_planner_steps'] = overall['total_planner_steps'] / overall['total_tasks']
        overall['avg_environment_steps'] = overall['total_environment_steps'] / overall['total_tasks']
    
    for subset, metrics in results['by_capability'].items():
        if metrics['total_tasks'] > 0:
            metrics['task_success_rate'] = metrics['successful_tasks'] / metrics['total_tasks']
            metrics['avg_planner_steps'] = metrics['total_planner_steps'] / metrics['total_tasks']
            metrics['avg_environment_steps'] = metrics['total_environment_steps'] / metrics['total_tasks']
    
    subgoal = results['subgoal_success']
    if subgoal['total_subgoals'] > 0:
        subgoal['subgoal_success_rate'] = subgoal['successful_subgoals'] / subgoal['total_subgoals']
    
    return results


def print_embodiedbench_report(results: Dict):
    """Print evaluation report in EmbodiedBench paper format."""
    print("\n" + "="*80)
    print("EMBODIEDBENCH-STYLE EVALUATION REPORT")
    print("="*80)
    
    # Overall metrics
    overall = results['overall']
    print(f"\n📊 OVERALL METRICS")
    print(f"  Total Tasks: {overall['total_tasks']}")
    print(f"  Successful Tasks: {overall['successful_tasks']}")
    print(f"  Task Success Rate: {overall['task_success_rate']:.2%}")
    print(f"  Avg Planner Steps: {overall['avg_planner_steps']:.2f}")
    print(f"  Avg Environment Steps: {overall['avg_environment_steps']:.2f}")
    
    # Capability breakdown
    print(f"\n📈 TASK SUCCESS RATE BY CAPABILITY SUBSET")
    print(f"  {'Subset':<20} {'Tasks':<10} {'Success':<10} {'Success Rate':<15} {'Avg Planner':<15} {'Avg Env':<15}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    # Standard order from paper
    standard_subsets = ['base', 'common_sense', 'complex', 'visual', 'spatial', 'long_horizon']
    for subset in standard_subsets:
        if subset in results['by_capability']:
            metrics = results['by_capability'][subset]
            print(f"  {subset:<20} {metrics['total_tasks']:<10} {metrics['successful_tasks']:<10} "
                  f"{metrics['task_success_rate']:>13.2%} {metrics['avg_planner_steps']:>13.2f} "
                  f"{metrics['avg_environment_steps']:>13.2f}")
    
    # Other subsets
    for subset, metrics in sorted(results['by_capability'].items()):
        if subset not in standard_subsets:
            print(f"  {subset:<20} {metrics['total_tasks']:<10} {metrics['successful_tasks']:<10} "
                  f"{metrics['task_success_rate']:>13.2%} {metrics['avg_planner_steps']:>13.2f} "
                  f"{metrics['avg_environment_steps']:>13.2f}")
    
    # Subgoal success
    subgoal = results['subgoal_success']
    print(f"\n🎯 SUBGOAL SUCCESS RATE")
    print(f"  Total Subgoals: {subgoal['total_subgoals']}")
    print(f"  Successful Subgoals: {subgoal['successful_subgoals']}")
    print(f"  Subgoal Success Rate: {subgoal['subgoal_success_rate']:.2%}")
    
    # Error analysis
    print(f"\n🔍 ERROR ANALYSIS")
    if results['error_analysis']:
        total_errors = sum(results['error_analysis'].values())
        for error_type, count in sorted(results['error_analysis'].items(), key=lambda x: -x[1]):
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"  {error_type:<20} {count:<10} ({percentage:>5.1f}%)")
    else:
        print("  No errors recorded.")
    
    print("\n" + "="*80)


def save_results(results: Dict, output_dir: Path, checkpoint_name: str):
    """Save evaluation results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results JSON
    json_file = output_dir / f"embodiedbench_evaluation_{checkpoint_name}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Full results saved to: {json_file}")
    
    # Save summary CSV (matching paper table format)
    summary_data = []
    
    # Overall row
    overall = results['overall']
    summary_data.append({
        'Capability_Subset': 'Overall',
        'Total_Tasks': overall['total_tasks'],
        'Successful_Tasks': overall['successful_tasks'],
        'Task_Success_Rate': overall['task_success_rate'],
        'Avg_Planner_Steps': overall['avg_planner_steps'],
        'Avg_Environment_Steps': overall['avg_environment_steps'],
    })
    
    # Capability subsets
    standard_subsets = ['base', 'common_sense', 'complex', 'visual', 'spatial', 'long_horizon']
    for subset in standard_subsets:
        if subset in results['by_capability']:
            metrics = results['by_capability'][subset]
            summary_data.append({
                'Capability_Subset': subset.replace('_', ' ').title(),
                'Total_Tasks': metrics['total_tasks'],
                'Successful_Tasks': metrics['successful_tasks'],
                'Task_Success_Rate': metrics['task_success_rate'],
                'Avg_Planner_Steps': metrics['avg_planner_steps'],
                'Avg_Environment_Steps': metrics['avg_environment_steps'],
            })
    
    df = pd.DataFrame(summary_data)
    csv_file = output_dir / f"embodiedbench_summary_{checkpoint_name}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Summary CSV saved to: {csv_file}")
    
    return json_file, csv_file


def main():
    parser = argparse.ArgumentParser(description='EmbodiedBench-style evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./data/EB-Man_trajectory_dataset',
                        help='Path to data directory')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to evaluate (None = all)')
    parser.add_argument('--success-threshold', type=float, default=0.8,
                        help='Success threshold (0-1), default 0.8 = 80%% accuracy')
    parser.add_argument('--output-dir', type=str, default='./logs',
                        help='Directory to save results')
    parser.add_argument('--s3-bucket', type=str, default=None,
                        help='S3 bucket for uploading results (optional)')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    bins = checkpoint.get('bins', [101, 101, 101, 121, 121, 121, 2])
    model = AgentModel(bins=bins, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_root}")
    dataset = EmbodiedDataset(
        data_root=args.data_root,
        debug=False,
        dataset_type="single_step"
    )
    print(f"Dataset loaded: {len(dataset)} episodes")
    
    # Evaluate
    results = evaluate_embodiedbench_style(
        model=model,
        dataset=dataset,
        device=device,
        success_threshold=args.success_threshold,
        max_episodes=args.max_episodes
    )
    
    # Print report
    print_embodiedbench_report(results)
    
    # Save results
    checkpoint_name = Path(args.checkpoint).stem
    output_dir = Path(args.output_dir)
    json_file, csv_file = save_results(results, output_dir, checkpoint_name)
    
    # Upload to S3 if specified
    if args.s3_bucket:
        import boto3
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(str(json_file), args.s3_bucket, f"metrics/{json_file.name}")
            s3_client.upload_file(str(csv_file), args.s3_bucket, f"metrics/{csv_file.name}")
            print(f"\n✓ Results uploaded to s3://{args.s3_bucket}/metrics/")
        except Exception as e:
            print(f"\n⚠️  Failed to upload to S3: {e}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

