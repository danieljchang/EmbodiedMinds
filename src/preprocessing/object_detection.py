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
                # Get box coordinates
                box_coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box_coords
                
                # Normalize coordinates
                x1_norm = float(x1 / w)
                y1_norm = float(y1 / h)
                x2_norm = float(x2 / w)
                y2_norm = float(y2 / h)
                cx_norm = (x1_norm + x2_norm) / 2
                cy_norm = (y1_norm + y2_norm) / 2
                
                detections.append({
                    'box': [x1_norm, y1_norm, x2_norm, y2_norm],
                    'center': [cx_norm, cy_norm],
                    'confidence': float(box.conf[0].cpu()),
                    'class_id': int(box.cls[0].cpu()),
                    'class_name': self.model.names[int(box.cls[0].cpu())],
                })
        
        return detections