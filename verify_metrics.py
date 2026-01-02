#!/usr/bin/env python3
"""
Quick script to verify all requested metrics are being saved.
"""
import json
import glob
from pathlib import Path

def check_metrics():
    """Check if all 5 requested metrics are present"""
    print("="*70)
    print("METRICS VERIFICATION CHECKLIST")
    print("="*70)
    print()
    
    # Check for comprehensive metrics files
    log_dir = Path("logs")
    metrics_files = list(log_dir.glob("metrics_*.json"))
    csv_files = list(log_dir.glob("metrics_summary_*.csv"))
    
    required_metrics = [
        "Task Success Rate",
        "Subgoal Success Rate", 
        "Planner Steps",
        "Environment Steps",
        "Error Analysis"
    ]
    
    print("1. Checking for comprehensive metrics files...")
    if metrics_files:
        latest_metrics = sorted(metrics_files)[-1]
        print(f"   ✓ Found: {latest_metrics}")
        
        with open(latest_metrics) as f:
            data = json.load(f)
            summary = data.get('summary', {})
            
            print()
            print("2. Checking for all 5 requested metrics:")
            print()
            
            # Check each metric
            checks = {
                "Task Success Rate": 'task_success_rate' in summary,
                "Subgoal Success Rate": 'avg_subgoal_success_rate' in summary,
                "Planner Steps": 'avg_planner_steps' in summary,
                "Environment Steps": 'avg_environment_steps' in summary,
                "Error Analysis": 'error_analysis' in summary
            }
            
            all_present = True
            for metric_name, present in checks.items():
                status = "✓" if present else "✗"
                print(f"   {status} {metric_name}")
                if not present:
                    all_present = False
            
            print()
            if all_present:
                print("   ✅ ALL 5 METRICS ARE PRESENT!")
                print()
                print("3. Current Values:")
                print()
                print(f"   Task Success Rate: {summary.get('task_success_rate', 0):.2%}")
                print(f"   Subgoal Success Rate: {summary.get('avg_subgoal_success_rate', 0):.2%}")
                print(f"   Avg Planner Steps: {summary.get('avg_planner_steps', 0):.2f}")
                print(f"   Avg Environment Steps: {summary.get('avg_environment_steps', 0):.2f}")
                print(f"   Error Analysis: {summary.get('error_analysis', {})}")
            else:
                print("   ⚠️  Some metrics are missing!")
    else:
        print("   ⚠️  No comprehensive metrics files found yet.")
        print("   Metrics will be saved after training completes.")
        print()
        print("   Expected files:")
        print("   - logs/metrics_YYYYMMDD_HHMMSS.json")
        print("   - logs/metrics_summary_YYYYMMDD_HHMMSS.csv")
    
    print()
    print("4. Checking CSV summary...")
    if csv_files:
        latest_csv = sorted(csv_files)[-1]
        print(f"   ✓ Found: {latest_csv}")
        print()
        print("   CSV Contents:")
        with open(latest_csv) as f:
            lines = f.readlines()[:15]
            for line in lines:
                print(f"   {line.rstrip()}")
    else:
        print("   ⚠️  No CSV summary found yet.")
    
    print()
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    check_metrics()

