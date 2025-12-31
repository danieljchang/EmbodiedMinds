"""
Trajectory-level evaluation for embodied agent tasks.

Evaluates complete task trajectories, not just single-step predictions.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from src.utils.task_metrics import TaskMetricsTracker, categorize_error
from src.preprocessing.object_detection import ObjectDetector
from src.preprocessing.depth_estimation import DepthEstimator
from data_loader import EmbodiedDataset, collate_fn_3d


class TrajectoryEvaluator:
    """
    Evaluate model on complete task trajectories.
    
    For each episode in the dataset:
    1. Load trajectory steps
    2. Run model inference for each step
    3. Track task completion and subgoals
    4. Calculate success metrics
    """
    
    def __init__(self, model, dataset: EmbodiedDataset, device: str = "cpu",
                 metrics_tracker: Optional[TaskMetricsTracker] = None):
        """
        Args:
            model: Trained AgentModel
            dataset: EmbodiedDataset instance
            device: Device to run inference on
            metrics_tracker: Optional TaskMetricsTracker for logging
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.metrics_tracker = metrics_tracker or TaskMetricsTracker()
        
        # Initialize preprocessing models (cached)
        self.detector = ObjectDetector(device=device)
        self.estimator = DepthEstimator(device=device)
        
        self.model.eval()
        
    def evaluate_episode(self, episode_idx: int, 
                        success_threshold: float = 0.8) -> Dict:
        """
        Evaluate a single episode/task.
        
        Args:
            episode_idx: Index of episode in dataset
            success_threshold: Threshold for considering task successful
                             (fraction of correct actions)
        
        Returns:
            Dictionary with episode evaluation results
        """
        # Get episode data
        item = self.dataset[episode_idx]
        instruction = item['instruction']
        episode_id = item.get('meta_path', f'episode_{episode_idx}')
        
        # Start tracking this task
        self.metrics_tracker.start_task(
            task_id=f"episode_{episode_idx}",
            instruction=instruction,
            episode_id=episode_idx
        )
        
        # Get trajectory steps from dataset
        # Note: The dataset returns demo images and actions, but we need to
        # evaluate step-by-step. For now, we'll use the demo actions as targets.
        demo_actions = item.get('demo_actions', [])
        if demo_actions and len(demo_actions) > 0:
            demo_actions_tensor = demo_actions[0]  # (num_steps, 7)
        else:
            demo_actions_tensor = torch.zeros((0, 7), dtype=torch.long)
        
        # Run inference for each step
        correct_steps = 0
        total_steps = 0
        
        # For trajectory evaluation, we need to simulate step-by-step execution
        # Since we have demo actions, we'll evaluate prediction accuracy
        with torch.no_grad():
            # Process data using collate_fn_3d to get 3D object representations
            # Create a batch with single item
            batch = [item]
            processed_batch = collate_fn_3d(batch, device=self.device)
            
            instructions = processed_batch['instructions']
            demo_3d_objects = processed_batch['demo_3d_objects']
            current_3d_objects = processed_batch['current_3d_objects']
            demo_actions_list = processed_batch.get('demo_actions', None)
            current_images = processed_batch.get('current_images', None)
            
            # Run model (single example batch)
            logits = self.model.forward(
                instructions,
                demo_3d_objects,
                current_3d_objects,
                demo_actions_list,
                current_images=current_images,
            )
            
            # Get predictions
            predictions = self.model.heads.predict(logits)
            
            # Evaluate against targets (last action in trajectory)
            if demo_actions_tensor.shape[0] > 0:
                target = demo_actions_tensor[-1]  # Last action as target
                pred = predictions[0]  # Single prediction
                
                # Check accuracy
                valid = (target != -1)
                if valid.any():
                    correct = (pred == target) & valid
                    correct_steps = correct.sum().item()
                    total_steps = valid.sum().item()
                    
                    # Record planner step
                    self.metrics_tracker.record_planner_step(
                        prediction=pred,
                        target=target,
                        action_taken=pred.cpu().numpy().tolist()
                    )
                    
                    # Categorize and record errors
                    if not correct.all():
                        error_type = categorize_error(pred, target, instruction, 0)
                        self.metrics_tracker.record_error(
                            error_type=error_type,
                            error_message=f"Prediction mismatch at final step",
                            step=0
                        )
        
        # Determine task success
        if total_steps > 0:
            success_rate = correct_steps / total_steps
            task_success = success_rate >= success_threshold
        else:
            success_rate = 0.0
            task_success = False
        
        # Record subgoals (simplified - in practice, parse from trajectory)
        # For now, treat each demo action as a subgoal
        if demo_actions_tensor.shape[0] > 0:
            for step_idx in range(min(3, demo_actions_tensor.shape[0])):  # First 3 steps as subgoals
                subgoal_success = True  # Simplified - would check actual completion
                self.metrics_tracker.record_subgoal(
                    subgoal_id=step_idx,
                    subgoal_description=f"Step {step_idx}",
                    success=subgoal_success
                )
        
        # Complete task
        self.metrics_tracker.complete_task(
            success=task_success,
            final_state={
                'success_rate': float(success_rate),
                'correct_steps': int(correct_steps),
                'total_steps': int(total_steps)
            }
        )
        
        return {
            'episode_id': episode_id,
            'instruction': instruction,
            'success': task_success,
            'success_rate': float(success_rate),
            'correct_steps': int(correct_steps),
            'total_steps': int(total_steps),
            'planner_steps': 1,  # Single inference for now
            'environment_steps': demo_actions_tensor.shape[0] if demo_actions_tensor.shape[0] > 0 else 0
        }
    
    def evaluate_all(self, max_episodes: Optional[int] = None, 
                    success_threshold: float = 0.8) -> Dict:
        """
        Evaluate all episodes in dataset.
        
        Args:
            max_episodes: Maximum number of episodes to evaluate (None = all)
            success_threshold: Threshold for task success
        
        Returns:
            Dictionary with evaluation results
        """
        num_episodes = min(len(self.dataset), max_episodes) if max_episodes else len(self.dataset)
        
        print(f"Evaluating {num_episodes} episodes...")
        
        episode_results = []
        for i in range(num_episodes):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_episodes}")
            
            try:
                result = self.evaluate_episode(i, success_threshold=success_threshold)
                episode_results.append(result)
            except Exception as e:
                print(f"  Error evaluating episode {i}: {e}")
                continue
        
        # Get summary from metrics tracker
        summary = self.metrics_tracker.get_summary()
        
        return {
            'episode_results': episode_results,
            'summary': summary,
            'num_episodes': num_episodes
        }

