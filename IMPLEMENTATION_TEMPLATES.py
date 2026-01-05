"""
Template implementation for Phase 1 of the architecture.
This file provides starter code for building the 3D perception pipeline.

Copy and adapt these implementations into your actual files.
"""

# ============================================================================
# FILE: src/preprocessing/object_detection.py (REPLACEMENT)
# ============================================================================

from ultralytics import YOLO
import numpy as np
from typing import List, Dict


class ObjectDetector:
    """
    Detects objects using YOLOv8. Modern replacement for OpenCV DNN.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
        """
        Args:
            model_name: YOLOv8 model size
                - "yolov8n.pt" (nano, fast, least accurate)
                - "yolov8s.pt" (small)
                - "yolov8m.pt" (medium, recommended)
                - "yolov8l.pt" (large)
                - "yolov8x.pt" (extra-large, most accurate)
            device: "cpu" or "cuda"
        """
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(device)
        
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), values in [0, 255]
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of dicts with keys:
                - 'box': [x1_norm, y1_norm, x2_norm, y2_norm] (normalized 0-1)
                - 'center': [cx_norm, cy_norm] (normalized 0-1)
                - 'confidence': float in [0, 1]
                - 'class_id': int
                - 'class_name': str
        """
        h, w = image.shape[:2]
        
        # Run detection
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Normalized coordinates
                x1, y1, x2, y2 = (box.xyxy[0] / np.array([w, h, w, h])).cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                detections.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(cx), float(cy)],
                    'confidence': float(box.conf[0].cpu()),
                    'class_id': int(box.cls[0].cpu()),
                    'class_name': self.model.names[int(box.cls[0])],
                })
        
        return detections


# ============================================================================
# FILE: src/preprocessing/depth_estimation.py (REPLACEMENT)
# ============================================================================

import torch
import numpy as np
import cv2


class DepthEstimator:
    """
    Estimates depth maps using MiDaS pre-trained model.
    """
    
    def __init__(self, model_type: str = "DPT_Large", device: str = "cpu"):
        """
        Args:
            model_type: "DPT_Large" (best), "DPT_Hybrid" (balanced), "MiDaS_small" (fast)
            device: "cpu" or "cuda"
        """
        self.device = device
        
        # Load model
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(device)
        self.midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), values in [0, 255]
            
        Returns:
            Depth map as numpy array (H, W), normalized to [0, 1]
        """
        # Prepare input
        input_batch = self.transform(image).to(self.device)
        
        # Estimate depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize to input resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.ones_like(depth) * 0.5
        
        return depth_normalized.astype(np.float32)


# ============================================================================
# FILE: src/preprocessing/fusion_utils.py (NEW FILE)
# ============================================================================

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


# ============================================================================
# FILE: src/encoders/object_encoder.py (NEW FILE)
# ============================================================================

import torch
import torch.nn as nn


class ObjectEncoder(nn.Module):
    """
    Encodes 3D object representations into learned embeddings.
    
    Input: 3D object features (num_objects, 7)
    Output: Object embeddings (num_objects, embedding_dim)
    """
    
    def __init__(self, object_feature_dim: int = 7, embedding_dim: int = 256):
        """
        Args:
            object_feature_dim: Dimensionality of 3D features (should be 7)
            embedding_dim: Dimensionality of output embeddings
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """
        Encode objects to embeddings.
        
        Args:
            objects: (num_objects, 7) tensor or (B, num_objects, 7)
            
        Returns:
            (num_objects, embedding_dim) or (B, num_objects, embedding_dim)
        """
        original_shape = objects.shape
        
        # Handle batched input
        if len(original_shape) == 3:
            B, N, D = original_shape
            objects_flat = objects.reshape(B * N, D)
            embeddings_flat = self.encoder(objects_flat)
            return embeddings_flat.reshape(B, N, self.embedding_dim)
        else:
            return self.encoder(objects)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import cv2
    
    # Load a sample image
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    detector = ObjectDetector(model_name="yolov8n.pt", device=device)
    depth_estimator = DepthEstimator(model_type="DPT_Large", device=device)
    object_encoder = ObjectEncoder(object_feature_dim=7, embedding_dim=256)
    
    # Process image
    print("Detecting objects...")
    objects = detector.detect_objects(image, conf_threshold=0.5)
    print(f"Found {len(objects)} objects")
    
    print("Estimating depth...")
    depth_map = depth_estimator.estimate_depth(image)
    print(f"Depth map shape: {depth_map.shape}")
    
    print("Creating 3D representations...")
    h, w = image.shape[:2]
    obj_3d = create_3d_object_representations(objects, depth_map, h, w)
    print(f"3D representations shape: {obj_3d.shape}")
    
    if obj_3d.shape[0] > 0:
        print("Encoding objects...")
        obj_embeddings = object_encoder(obj_3d)
        print(f"Object embeddings shape: {obj_embeddings.shape}")
    else:
        print("No objects detected in image")
