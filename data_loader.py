from huggingface_hub import snapshot_download
from pathlib import Path
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.object_detection import ObjectDetector
from src.preprocessing.depth_estimation import DepthEstimator
from src.preprocessing.fusion_utils import create_3d_object_representations

local_folder = snapshot_download(
    repo_id="EmbodiedBench/EB-Man_trajectory_dataset",
    repo_type="dataset",
    local_dir="./data/EB-Man_trajectory_dataset",     # or any folder you choose
    local_dir_use_symlinks=False
)

def resolve_data_root():
    """
    Resolve the data root directory.

    Priority:
      1. EMBODIEDBENCH_DATA env var
      2. ./data relative to cwd
      3. ../data relative to cwd
      4. ~/data
    """
    env_path = os.environ.get("EMBODIEDBENCH_DATA", None)
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p.resolve()
    candidates = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path.home() / "data",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not find data folder. Set EMBODIEDBENCH_DATA in your .env or environment\n"
        "Examples:\n"
        "  export EMBODIEDBENCH_DATA=/absolute/path/to/your/data\n"
        "  or create ./data or ../data relative to the repo root\n"
    )

class EmbodiedDataset(Dataset):
    """
    Dataset loader for EB-Man trajectory dataset.
    Loads from JSON files and image directories with proper train/val/test splits.
    """
    def __init__(self, data_root=None, debug=False, dataset_type="single_step", split="train", seed=42):
        """
        Args:
            data_root: Root directory containing EB-Man_trajectory_dataset
            debug: If True, use smaller subset
            dataset_type: "single_step" or "multi_step"
            split: "train", "val", or "test" (80/10/10 split)
            seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root) if data_root else resolve_data_root()
        self.debug = debug
        self.dataset_type = dataset_type
        self.split = split
        
        # Find dataset directory
        dataset_dir = self.data_root / "EB-Man_trajectory_dataset"
        if not dataset_dir.exists():
            dataset_dir = self.data_root
        
        # Load JSON file
        json_file = dataset_dir / f"eb-man_dataset_{dataset_type}.json"
        if not json_file.exists():
            # Try alternative locations
            json_file = dataset_dir / "eb-man_dataset_single_step.json"
            if not json_file.exists():
                raise FileNotFoundError(f"Could not find dataset JSON in {dataset_dir}")
        
        print(f"Loading dataset from: {json_file}")
        with open(json_file, 'r') as f:
            all_data = json.load(f)
        
        # Create reproducible train/val/test split (80/10/10)
        np.random.seed(seed)
        indices = np.random.permutation(len(all_data))
        
        n_train = int(0.8 * len(all_data))
        n_val = int(0.1 * len(all_data))
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        if split == "train":
            selected_indices = train_indices
        elif split == "val":
            selected_indices = val_indices
        elif split == "test":
            selected_indices = test_indices
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        self.data = [all_data[i] for i in selected_indices]
        
        # Filter and limit for debug
        if debug:
            self.data = self.data[:20]
        
        self.dataset_dir = dataset_dir
        print(f"Loaded {len(self.data)} examples for {split} split")

    def __len__(self):
        return len(self.data)

    def _parse_action_string(self, action_str):
        """Parse action string like '[33, 43, 27, 0, 60, 90, 1]' to list"""
        if isinstance(action_str, str):
            # Remove brackets and split
            action_str = action_str.strip('[]')
            return [int(x.strip()) for x in action_str.split(',')]
        return action_str

    def _load_image(self, image_path):
        """Load image from path relative to dataset directory"""
        import cv2
        from PIL import Image
        
        # Handle relative paths
        if not Path(image_path).is_absolute():
            full_path = self.dataset_dir / image_path
        else:
            full_path = Path(image_path)
        
        # Try multiple path resolutions
        if not full_path.exists():
            path_parts = Path(image_path).parts
            
            # Strategy 1: Try images/visual/ directory (based on user's folder structure)
            if len(path_parts) >= 2:
                episode_name = path_parts[-2]  # e.g., "episode_1"
                filename = path_parts[-1]      # e.g., "step_1.png"
                
                # Look in images/visual/episode_X/
                visual_path = self.dataset_dir / "images" / "visual" / episode_name / filename
                if visual_path.exists():
                    full_path = visual_path
            
            # Strategy 2: Search for the file in any subdirectory
            if not full_path.exists() and len(path_parts) >= 2:
                episode_name = path_parts[-2]
                filename = path_parts[-1]
                
                # Search in all model directories
                images_dir = self.dataset_dir / "images"
                if images_dir.exists():
                    for model_dir in images_dir.iterdir():
                        if model_dir.is_dir():
                            for category_dir in model_dir.iterdir():
                                if category_dir.is_dir():
                                    episode_dir = category_dir / episode_name
                                    if episode_dir.exists():
                                        candidate = episode_dir / filename
                                        if candidate.exists():
                                            full_path = candidate
                                            break
                            if full_path.exists():
                                break
            
            # Strategy 3: Direct search by filename
            if not full_path.exists() and len(path_parts) > 0:
                filename = path_parts[-1]
                images_dir = self.dataset_dir / "images"
                if images_dir.exists():
                    matches = list(images_dir.rglob(filename))
                    if matches:
                        full_path = matches[0]
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path} (tried: {full_path})")
        
        # Load image
        img = cv2.imread(str(full_path))
        if img is None:
            # Try with PIL
            img = np.array(Image.open(full_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - instruction: str
                - demo_images: List of (3, H, W) tensors
                - current_image: (3, H, W) tensor
                - demo_actions: List of action tensors (7,) or (max_steps, 7)
        """
        item = self.data[idx]
        trajectory = item.get('trajectory', [])
        
        # Extract instruction
        instruction = item.get('instruction', '')
        
        # Extract demo images and actions from trajectory
        demo_images = []
        demo_actions = []
        
        for step in trajectory:
            # Get executable plan
            exec_plan = step.get('executable_plan', {})
            
            # Handle both dict and list formats
            if isinstance(exec_plan, list):
                # Multi-step format: executable_plan is a list
                for plan_item in exec_plan:
                    img_path = plan_item.get('img_path')
                    action_str = plan_item.get('action')
                    if img_path and action_str:
                        try:
                            img = self._load_image(img_path)
                            demo_images.append(img)
                            action = self._parse_action_string(action_str)
                            demo_actions.append(torch.tensor(action, dtype=torch.long))
                        except Exception as e:
                            print(f"Warning: Failed to load {img_path}: {e}")
                            continue
            elif isinstance(exec_plan, dict):
                # Single-step format: executable_plan is a dict
                img_path = exec_plan.get('img_path')
                action_str = exec_plan.get('action')
                if img_path and action_str:
                    try:
                        img = self._load_image(img_path)
                        demo_images.append(img)
                        action = self._parse_action_string(action_str)
                        demo_actions.append(torch.tensor(action, dtype=torch.long))
                    except Exception as e:
                        print(f"Warning: Failed to load {img_path}: {e}")
                        continue
            
            # Also check input_image_path for additional context
            input_img_path = step.get('input_image_path')
            if input_img_path and input_img_path not in [exec_plan.get('img_path') if isinstance(exec_plan, dict) else None]:
                try:
                    img = self._load_image(input_img_path)
                    # Use as demo if we don't have many demos yet
                    if len(demo_images) < 3:
                        demo_images.append(img)
                except:
                    pass
        
        # Get current image (last step's input image or last demo image)
        current_image = None
        if trajectory:
            last_step = trajectory[-1]
            input_img_path = last_step.get('input_image_path')
            if input_img_path:
                try:
                    current_image = self._load_image(input_img_path)
                except:
                    pass
        
        # Fallback: use last demo image as current
        if current_image is None and demo_images:
            current_image = demo_images[-1]
        
        # Fallback: create dummy image
        if current_image is None:
            current_image = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        # Stack demo images if we have any
        if demo_images:
            # Pad to same size if needed
            max_h = max(img.shape[1] for img in demo_images)
            max_w = max(img.shape[2] for img in demo_images)
            padded_demos = []
            for img in demo_images:
                if img.shape[1] != max_h or img.shape[2] != max_w:
                    # Resize
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0), 
                        size=(max_h, max_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                padded_demos.append(img)
            demo_images_tensor = torch.stack(padded_demos, dim=0)  # (num_demos, 3, H, W)
        else:
            demo_images_tensor = torch.zeros(0, 3, 224, 224, dtype=torch.float32)
        
        # Format demo actions
        if demo_actions:
            # Stack actions: (num_demos, 7)
            demo_actions_tensor = torch.stack(demo_actions, dim=0)
        else:
            demo_actions_tensor = torch.zeros(0, 7, dtype=torch.long)
        
        return {
            "instruction": instruction,
            "demo_images": demo_images_tensor,  # (num_demos, 3, H, W)
            "current_image": current_image,  # (3, H, W)
            "demo_actions": [demo_actions_tensor],  # List with one tensor (num_demos, 7)
            "meta_path": f"episode_{item.get('episode_id', idx)}"
        }

# Global cache for preprocessing models (to avoid reloading for each batch)
_detector_cache = {}
_estimator_cache = {}

def collate_fn_3d(batch, device="cpu"):
    """
    Collate function that performs 3D preprocessing on images.
    
    Args:
        batch: List of dicts from EmbodiedDataset with keys:
            - instruction: str
            - demo_images: (num_demos, 3, H, W) tensor
            - current_image: (3, H, W) tensor
            - demo_actions: list of (max_steps, 7) tensors
        device: Device to run preprocessing on
        
    Returns:
        dict with keys:
            - instructions: List[str]
            - demo_3d_objects: List[List[torch.Tensor]] - outer list is batch, inner is demos
            - current_3d_objects: List[torch.Tensor] - one per batch element
            - demo_actions: List[torch.Tensor] - one per batch element (last action from each demo)
            - targets: (B, 7) tensor
    """
    # Use cached models to avoid reloading for each batch
    global _detector_cache, _estimator_cache
    
    if device not in _detector_cache:
        _detector_cache[device] = ObjectDetector(device=device)
    detector = _detector_cache[device]
    
    if device not in _estimator_cache:
        _estimator_cache[device] = DepthEstimator(device=device)
    estimator = _estimator_cache[device]
    
    instructions = []
    demo_3d_objects_list = []
    current_3d_objects_list = []
    demo_actions_list = []
    targets = []
    # Raw RGB images for optional VLM vision encoding
    demo_images_batch: List[torch.Tensor] = []
    current_images_batch: List[torch.Tensor] = []
    
    for b in batch:
        instructions.append(b["instruction"])
        
        # Process demonstration images
        demo_3d_objs_per_example = []
        if "demo_images" in b and b["demo_images"] is not None:
            demo_images = b["demo_images"]  # (num_demos, 3, H, W)
            num_demos = demo_images.shape[0]
            # Keep raw demo images for potential VLM vision usage
            demo_images_batch.append(demo_images)
            
            for demo_idx in range(num_demos):
                # Average frames if needed, or take first frame
                demo_img = demo_images[demo_idx]  # (3, H, W)
                
                # Convert to numpy RGB format (H, W, 3) in [0, 255]
                if demo_img.max() <= 1.0:
                    demo_img = (demo_img * 255).clamp(0, 255)
                demo_img_np = demo_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                h, w = demo_img_np.shape[:2]
                
                # Detect objects
                objects = detector.detect_objects(demo_img_np, conf_threshold=0.5)
                
                # Estimate depth
                depth_map = estimator.estimate_depth(demo_img_np)
                
                # Create 3D representations
                obj_3d = create_3d_object_representations(objects, depth_map, h, w)
                demo_3d_objs_per_example.append(obj_3d)
        else:
            # No demos - add empty list and empty image tensor
            demo_3d_objs_per_example = []
            demo_images_batch.append(torch.zeros(0, 3, 224, 224, dtype=torch.float32))
        
        demo_3d_objects_list.append(demo_3d_objs_per_example)
        
        # Process current image
        if "current_image" in b and b["current_image"] is not None:
            current_img = b["current_image"]  # (3, H, W)
            current_images_batch.append(current_img)
            
            # Convert to numpy RGB format
            if current_img.max() <= 1.0:
                current_img = (current_img * 255).clamp(0, 255)
            current_img_np = current_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            h, w = current_img_np.shape[:2]
            
            # Detect objects
            objects = detector.detect_objects(current_img_np, conf_threshold=0.5)
            
            # Estimate depth
            depth_map = estimator.estimate_depth(current_img_np)
            
            # Create 3D representations
            obj_3d = create_3d_object_representations(objects, depth_map, h, w)
            current_3d_objects_list.append(obj_3d)
        else:
            # No current image - add empty tensors
            current_3d_objects_list.append(torch.zeros((0, 7), dtype=torch.float32))
            current_images_batch.append(torch.zeros(3, 224, 224, dtype=torch.float32))
        
        # Extract demo actions (last valid action from each demo)
        demo_actions_per_example = []
        if "demo_actions" in b and b["demo_actions"] is not None:
            for demo_action_seq in b["demo_actions"]:  # (max_steps, 7)
                # Find last valid action
                valid = (demo_action_seq != -1).all(dim=1)
                idxs = valid.nonzero(as_tuple=False)
                if len(idxs) > 0:
                    last_action = demo_action_seq[idxs[-1].item()]  # (7,)
                    demo_actions_per_example.append(last_action)
                else:
                    demo_actions_per_example.append(torch.full((7,), -1, dtype=torch.long))
        else:
            demo_actions_per_example = []
        
        demo_actions_list.append(demo_actions_per_example)
        
        # Extract target (last valid action from first demo)
        if "demo_actions" in b and b["demo_actions"] is not None and len(b["demo_actions"]) > 0:
            seq = b["demo_actions"][0]  # (max_steps, 7)
            valid = (seq != -1).all(dim=1)
            idxs = valid.nonzero(as_tuple=False)
            if len(idxs) == 0:
                targets.append([-1] * 7)
            else:
                last = idxs[-1].item()
                targets.append(seq[last].tolist())
        else:
            targets.append([-1] * 7)
    
    # Convert demo_actions to list of tensors (one per demo, not per batch element)
    # This is a bit complex - we need to reorganize
    num_demos = max(len(demo_actions_list[b]) for b in range(len(demo_actions_list))) if demo_actions_list else 0
    demo_actions_by_demo = []
    for demo_idx in range(num_demos):
        actions_for_this_demo = []
        for b in range(len(demo_actions_list)):
            if demo_idx < len(demo_actions_list[b]):
                actions_for_this_demo.append(demo_actions_list[b][demo_idx])
            else:
                actions_for_this_demo.append(torch.full((7,), -1, dtype=torch.long))
        demo_actions_by_demo.append(torch.stack(actions_for_this_demo, dim=0))  # (B, 7)
    
    return {
        'instructions': instructions,
        'demo_3d_objects': demo_3d_objects_list,
        'current_3d_objects': current_3d_objects_list,
        'demo_actions': demo_actions_by_demo if demo_actions_by_demo else None,
        'targets': torch.tensor(targets, dtype=torch.long),
        # Raw images, as lists of tensors, for optional VLM vision encoding
        'demo_images': demo_images_batch,
        'current_images': current_images_batch,
    }


def build_dataloader(batch_size=4, debug=False, data_root=None, num_workers=0, use_3d_preprocessing=True, device="cpu", split="train", seed=42):
    """
    Build dataloader with optional 3D preprocessing and proper data splits.
    
    Args:
        batch_size: Batch size
        debug: Use debug mode (smaller dataset)
        data_root: Root directory for data
        num_workers: Number of worker processes
        use_3d_preprocessing: If True, use collate_fn_3d for 3D preprocessing
        device: Device for preprocessing (if use_3d_preprocessing=True)
        split: "train", "val", or "test" (80/10/10 split)
        seed: Random seed for reproducible splits
    """
    ds = EmbodiedDataset(data_root=data_root, debug=debug, split=split, seed=seed)
    # Shuffle only for training
    shuffle = (split == "train")
    collate_fn = collate_fn_3d if use_3d_preprocessing else None
    if collate_fn is not None:
        # Create a lambda that passes device
        collate_fn_with_device = lambda batch: collate_fn_3d(batch, device=device)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_with_device)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)