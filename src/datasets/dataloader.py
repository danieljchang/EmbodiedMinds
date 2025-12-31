"""
Data loader module - imports from root-level data_loader.py for compatibility.
This module provides the EmbodiedDataset and build_dataloader functions.
"""
import sys
from pathlib import Path

# Add parent directory to path to import from root
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import from root-level data_loader
from data_loader import (
    EmbodiedDataset,
    build_dataloader,
    collate_fn_3d,
    resolve_data_root,
)

# Re-export for compatibility
__all__ = [
    'EmbodiedDataset',
    'build_dataloader',
    'collate_fn_3d',
    'resolve_data_root',
]

