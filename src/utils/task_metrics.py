"""
Comprehensive metrics tracking for embodied agent tasks.

Tracks:
1. Task Success Rate
2. Subgoal Success Rate
3. Planner Steps
4. Environment Steps
5. Error Analysis
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import boto3
from botocore.exceptions import ClientError


class TaskMetricsTracker:
    """Track comprehensive task-level metrics"""
    
    def __init__(self, log_dir: str = "./logs", s3_bucket: Optional[str] = None, s3_prefix: str = "metrics/"):
        """
        Args:
            log_dir: Local directory to save metrics
            s3_bucket: S3 bucket name for uploading metrics (optional)
            s3_prefix: S3 prefix for metrics files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None
        if s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
            except Exception as e:
                print(f"Warning: Could not initialize S3 client: {e}")
        
        # Metrics storage
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.task_results = []  # List of task-level results
        self.current_task = None
        self.current_task_id = None
        
    def start_task(self, task_id: str, instruction: str, episode_id: Optional[int] = None):
        """Start tracking a new task"""
        self.current_task = {
            'task_id': task_id,
            'episode_id': episode_id,
            'instruction': instruction,
            'start_time': datetime.now().isoformat(),
            'planner_steps': 0,
            'environment_steps': 0,
            'subgoals': [],
            'actions': [],
            'predictions': [],
            'targets': [],
            'errors': [],
            'completed': False,
            'success': False,
        }
        self.current_task_id = task_id
        
    def record_planner_step(self, prediction: torch.Tensor, target: torch.Tensor, 
                           action_taken: Optional[List[int]] = None):
        """Record a planner inference step"""
        if self.current_task is None:
            return
            
        self.current_task['planner_steps'] += 1
        self.current_task['predictions'].append(prediction.cpu().numpy().tolist() if isinstance(prediction, torch.Tensor) else prediction)
        self.current_task['targets'].append(target.cpu().numpy().tolist() if isinstance(target, torch.Tensor) else target)
        
        if action_taken:
            self.current_task['actions'].append(action_taken)
            self.current_task['environment_steps'] += 1
    
    def record_subgoal(self, subgoal_id: int, subgoal_description: str, 
                      success: bool, error_type: Optional[str] = None):
        """Record a subgoal attempt"""
        if self.current_task is None:
            return
            
        subgoal_result = {
            'subgoal_id': subgoal_id,
            'description': subgoal_description,
            'success': success,
            'error_type': error_type,
            'step': self.current_task['planner_steps']
        }
        self.current_task['subgoals'].append(subgoal_result)
        
        if not success and error_type:
            self.current_task['errors'].append({
                'type': error_type,
                'step': self.current_task['planner_steps'],
                'subgoal_id': subgoal_id
            })
    
    def record_error(self, error_type: str, error_message: str, step: Optional[int] = None):
        """Record an error during task execution"""
        if self.current_task is None:
            return
            
        error = {
            'type': error_type,
            'message': error_message,
            'step': step if step is not None else self.current_task['planner_steps']
        }
        self.current_task['errors'].append(error)
    
    def complete_task(self, success: bool, final_state: Optional[Dict] = None):
        """Mark task as complete"""
        if self.current_task is None:
            return
            
        self.current_task['completed'] = True
        self.current_task['success'] = success
        self.current_task['end_time'] = datetime.now().isoformat()
        
        if final_state:
            self.current_task['final_state'] = final_state
        
        # Calculate task-level metrics
        self.current_task['metrics'] = self._calculate_task_metrics()
        
        # Add to results
        self.task_results.append(self.current_task.copy())
        
        # Reset current task
        self.current_task = None
        self.current_task_id = None
    
    def _calculate_task_metrics(self) -> Dict:
        """Calculate metrics for current task"""
        if self.current_task is None:
            return {}
        
        task = self.current_task
        
        # Subgoal success rate
        if task['subgoals']:
            successful_subgoals = sum(1 for sg in task['subgoals'] if sg['success'])
            subgoal_success_rate = successful_subgoals / len(task['subgoals'])
        else:
            subgoal_success_rate = 0.0
        
        # Action accuracy (per-dimension and overall)
        if task['predictions'] and task['targets']:
            preds = np.array(task['predictions'])
            targets = np.array(task['targets'])
            
            # Filter out invalid targets (-1)
            valid_mask = (targets != -1)
            if valid_mask.any():
                correct = (preds == targets) & valid_mask
                per_dim_acc = correct.sum(axis=0) / valid_mask.sum(axis=0)
                overall_acc = correct.sum() / valid_mask.sum()
            else:
                per_dim_acc = np.zeros(7)
                overall_acc = 0.0
        else:
            per_dim_acc = np.zeros(7)
            overall_acc = 0.0
        
        # Error categorization
        error_counts = defaultdict(int)
        for error in task['errors']:
            error_counts[error['type']] += 1
        
        return {
            'subgoal_success_rate': float(subgoal_success_rate),
            'action_accuracy': float(overall_acc),
            'per_dimension_accuracy': per_dim_acc.tolist(),
            'planner_steps': task['planner_steps'],
            'environment_steps': task['environment_steps'],
            'error_counts': dict(error_counts),
            'total_errors': len(task['errors'])
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all tasks"""
        if not self.task_results:
            return {
                'task_success_rate': 0.0,
                'avg_subgoal_success_rate': 0.0,
                'avg_planner_steps': 0.0,
                'avg_environment_steps': 0.0,
                'total_tasks': 0,
                'error_analysis': {}
            }
        
        total_tasks = len(self.task_results)
        successful_tasks = sum(1 for t in self.task_results if t['success'])
        task_success_rate = successful_tasks / total_tasks
        
        # Average subgoal success rate
        subgoal_rates = [t['metrics']['subgoal_success_rate'] for t in self.task_results if t['subgoals']]
        avg_subgoal_success_rate = np.mean(subgoal_rates) if subgoal_rates else 0.0
        
        # Average steps
        avg_planner_steps = np.mean([t['planner_steps'] for t in self.task_results])
        avg_environment_steps = np.mean([t['environment_steps'] for t in self.task_results])
        
        # Error analysis
        all_errors = []
        for task in self.task_results:
            all_errors.extend(task['errors'])
        
        error_analysis = defaultdict(int)
        for error in all_errors:
            error_analysis[error['type']] += 1
        
        return {
            'task_success_rate': float(task_success_rate),
            'avg_subgoal_success_rate': float(avg_subgoal_success_rate),
            'avg_planner_steps': float(avg_planner_steps),
            'avg_environment_steps': float(avg_environment_steps),
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'error_analysis': dict(error_analysis),
            'total_errors': len(all_errors)
        }
    
    def save_metrics(self, filename: Optional[str] = None, upload_to_s3: bool = True):
        """Save metrics to file and optionally upload to S3"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Prepare data
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'task_results': self.task_results
        }
        
        # Save locally
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
        
        # Upload to S3 if configured
        if upload_to_s3 and self.s3_client and self.s3_bucket:
            try:
                s3_key = f"{self.s3_prefix}{filename}"
                self.s3_client.upload_file(str(filepath), self.s3_bucket, s3_key)
                print(f"Metrics uploaded to: s3://{self.s3_bucket}/{s3_key}")
            except ClientError as e:
                print(f"Warning: Failed to upload metrics to S3: {e}")
        
        return filepath
    
    def save_summary_csv(self, filename: Optional[str] = None):
        """Save summary as CSV for easy analysis"""
        import csv
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_summary_{timestamp}.csv"
        
        filepath = self.log_dir / filename
        
        summary = self.get_summary()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Task Success Rate', f"{summary['task_success_rate']:.4f}"])
            writer.writerow(['Avg Subgoal Success Rate', f"{summary['avg_subgoal_success_rate']:.4f}"])
            writer.writerow(['Avg Planner Steps', f"{summary['avg_planner_steps']:.2f}"])
            writer.writerow(['Avg Environment Steps', f"{summary['avg_environment_steps']:.2f}"])
            writer.writerow(['Total Tasks', summary['total_tasks']])
            writer.writerow(['Successful Tasks', summary['successful_tasks']])
            writer.writerow(['Failed Tasks', summary['failed_tasks']])
            writer.writerow(['Total Errors', summary['total_errors']])
            
            writer.writerow([])
            writer.writerow(['Error Type', 'Count'])
            for error_type, count in summary['error_analysis'].items():
                writer.writerow([error_type, count])
        
        print(f"Summary CSV saved to: {filepath}")
        return filepath


def categorize_error(prediction: torch.Tensor, target: torch.Tensor, 
                    instruction: str, step: int) -> str:
    """
    Categorize error type based on prediction vs target.
    
    Error types:
    - 'perception': Errors in position (x, y, z)
    - 'orientation': Errors in rotation (roll, pitch, yaw)
    - 'gripper': Errors in gripper state
    - 'planning': Multiple dimension errors (planning failure)
    """
    pred = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else np.array(prediction)
    tgt = target.cpu().numpy() if isinstance(target, torch.Tensor) else np.array(target)
    
    # Filter valid targets
    valid = (tgt != -1)
    if not valid.any():
        return 'invalid_target'
    
    # Check which dimensions are wrong
    wrong = (pred != tgt) & valid
    
    if not wrong.any():
        return 'none'  # No error
    
    # Categorize
    position_wrong = wrong[:3].any()  # x, y, z
    orientation_wrong = wrong[3:6].any()  # roll, pitch, yaw
    gripper_wrong = wrong[6] if len(wrong) > 6 else False
    
    if position_wrong and orientation_wrong:
        return 'planning'  # Multiple errors = planning failure
    elif position_wrong:
        return 'perception'  # Position error
    elif orientation_wrong:
        return 'orientation'  # Orientation error
    elif gripper_wrong:
        return 'gripper'  # Gripper error
    else:
        return 'unknown'

