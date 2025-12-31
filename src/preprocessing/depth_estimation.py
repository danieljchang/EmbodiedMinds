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