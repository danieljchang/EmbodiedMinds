#!/usr/bin/env python3
"""
Test script to find optimal batch size for GPU memory.
"""
import torch
import sys
sys.path.insert(0, '.')

from src.models.agent_model import AgentModel
from data_loader import build_dataloader

def test_batch_size(batch_size, device="cuda"):
    """Test if a batch size fits in GPU memory"""
    print(f"\nTesting batch_size={batch_size}...")
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Create model
        bins = [101, 101, 101, 121, 121, 121, 2]
        model = AgentModel(token_dim=256, out_dim=512, bins=bins, device=device)
        model.eval()
        
        # Get a small dataloader
        dataloader = build_dataloader(
            batch_size=batch_size,
            debug=True,  # Use small dataset
            data_root="./data/EB-Man_trajectory_dataset",
            num_workers=0,
            use_3d_preprocessing=True,
            device=device
        )
        
        # Try forward pass
        with torch.no_grad():
            batch = next(iter(dataloader))
            instructions = batch['instructions']
            demo_3d_objects = batch['demo_3d_objects']
            current_3d_objects = batch['current_3d_objects']
            demo_actions = batch.get('demo_actions', None)
            
            logits = model.forward(
                instructions,
                demo_3d_objects,
                current_3d_objects,
                demo_actions
            )
        
        # Check memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"  ✓ Success! Memory used: {memory_used:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ Out of memory: {e}")
            torch.cuda.empty_cache()
            return False
        else:
            raise e
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")
    
    # Test different batch sizes
    batch_sizes = [8, 12, 16, 20, 24, 32]
    
    max_safe_batch_size = 8
    for bs in batch_sizes:
        if test_batch_size(bs, device):
            max_safe_batch_size = bs
        else:
            break
    
    print(f"\n{'='*50}")
    print(f"Recommended batch size: {max_safe_batch_size}")
    print(f"{'='*50}")

