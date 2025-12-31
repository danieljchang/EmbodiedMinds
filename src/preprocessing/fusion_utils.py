import torch
import numpy as np
from typing import List, Dict


def create_3d_object_representations(
    objects: List[Dict],
    depth_map: np.ndarray,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    """
    Fuse 2D bounding boxes with depth map to create 3D representations.
    
    Args:
        objects: List of detections from ObjectDetector
                 Each has 'box', 'center', 'confidence', 'class_id'
        depth_map: (H, W) depth map from DepthEstimator
        image_h, image_w: Image dimensions
        
    Returns:
        (num_objects, 7) tensor of 3D representations:
        [center_x, center_y, depth, width, height, confidence, class_id]
        All normalized to [0, 1]
    """
    
    representations = []
    
    for obj in objects:
        x1_norm, y1_norm, x2_norm, y2_norm = obj['box']
        cx_norm, cy_norm = obj['center']
        
        # Convert normalized coords to pixel coords
        x1_px = int(x1_norm * image_w)
        y1_px = int(y1_norm * image_h)
        x2_px = int(x2_norm * image_w)
        y2_px = int(y2_norm * image_h)
        
        # Clamp to image bounds
        x1_px = max(0, min(x1_px, image_w - 1))
        y1_px = max(0, min(y1_px, image_h - 1))
        x2_px = max(x1_px + 1, min(x2_px, image_w))
        y2_px = max(y1_px + 1, min(y2_px, image_h))
        
        # Sample depth at bounding box
        bbox_region = depth_map[y1_px:y2_px, x1_px:x2_px]
        if bbox_region.size > 0:
            z_norm = float(np.mean(bbox_region))
        else:
            z_norm = 0.5
        
        # Width and height (already normalized)
        w_norm = x2_norm - x1_norm
        h_norm = y2_norm - y1_norm
        
        # Create 3D representation
        obj_3d = [
            float(cx_norm),              # center_x (normalized)
            float(cy_norm),              # center_y (normalized)
            float(z_norm),               # depth (normalized, 0=close, 1=far)
            float(w_norm),               # width (normalized)
            float(h_norm),               # height (normalized)
            float(obj['confidence']),    # confidence
            float(obj['class_id']),      # class_id
        ]
        
        representations.append(obj_3d)
    
    if representations:
        return torch.tensor(representations, dtype=torch.float32)
    else:
        # Return empty tensor with correct shape
        return torch.zeros((0, 7), dtype=torch.float32)

