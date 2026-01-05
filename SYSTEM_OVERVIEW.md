# System Overview: Data, Pipeline, and Training

Complete rundown of how data flows through the system, from raw dataset to model predictions.

---

## 1. Data Structure

### Dataset Format: EB-Man Trajectory Dataset

**Source**: HuggingFace `EmbodiedBench/EB-Man_trajectory_dataset`

**JSON Structure** (`eb-man_dataset_single_step.json`):
```json
{
  "episode_id": 1,
  "instruction": "Pick up the star and place it into the silver container",
  "trajectory": [
    {
      "executable_plan": {
        "img_path": "claude-3-5-sonnet-20241022/base/episode_1/step_0.png",
        "action": "[33, 43, 27, 0, 60, 90, 1]"
      },
      "input_image_path": "claude-3-5-sonnet-20241022/base/episode_1/step_0_input.png"
    },
    {
      "executable_plan": {
        "img_path": "claude-3-5-sonnet-20241022/base/episode_1/step_1.png",
        "action": "[35, 45, 28, 0, 60, 90, 1]"
      },
      "input_image_path": "claude-3-5-sonnet-20241022/base/episode_1/step_1_input.png"
    }
  ]
}
```

**Key Components**:
- **instruction**: Natural language task description (string)
- **trajectory**: List of steps showing how to complete the task
- **executable_plan**: Contains `img_path` (demonstration image) and `action` (7D action vector)
- **input_image_path**: Current state image before taking action
- **action**: 7-dimensional vector `[x, y, z, roll, pitch, yaw, gripper]`
  - x, y, z: Position (0-100, discretized)
  - roll, pitch, yaw: Rotation (0-120, discretized, each bin = 3°)
  - gripper: Open/closed (0 or 1)

---

## 2. Data Loading Pipeline

### Step 1: Dataset Loading (`EmbodiedDataset.__getitem__`)

**Input**: Index into JSON dataset  
**Output**: Dictionary with raw data

```python
{
    "instruction": str,                    # "Pick up the star..."
    "demo_images": Tensor,                 # (num_demos, 3, H, W) - demonstration images
    "current_image": Tensor,               # (3, H, W) - current scene image
    "demo_actions": [Tensor],              # List with (num_demos, 7) - demo actions
    "meta_path": str                       # Episode identifier
}
```

**Process**:
1. Load JSON entry
2. Extract instruction string
3. For each step in trajectory:
   - Load demonstration image from `executable_plan.img_path`
   - Parse action string `"[33, 43, 27, ...]"` → `[33, 43, 27, ...]`
4. Load current image from last step's `input_image_path`
5. Stack demo images and actions into tensors

### Step 2: 3D Preprocessing (`collate_fn_3d`)

**Input**: Batch of raw data dictionaries  
**Output**: Batch with 3D object representations

**For each image in batch**:
1. **Object Detection** (YOLOv8):
   - Input: RGB image (H, W, 3)
   - Output: List of detections
     ```python
     {
         'box': [x1_norm, y1_norm, x2_norm, y2_norm],  # Normalized [0, 1]
         'center': [cx_norm, cy_norm],                  # Normalized [0, 1]
         'confidence': float,                           # [0, 1]
         'class_id': int,                               # Object class
         'class_name': str                              # e.g., "cup", "bottle"
     }
     ```

2. **Depth Estimation** (MiDaS):
   - Input: RGB image (H, W, 3)
   - Output: Depth map (H, W) normalized to [0, 1]
     - 0 = close to camera
     - 1 = far from camera

3. **3D Fusion** (`create_3d_object_representations`):
   - Input: Detections + Depth map
   - Output: 3D object representations (N, 7)
     ```python
     [
         center_x_norm,    # [0, 1] - X position in image
         center_y_norm,    # [0, 1] - Y position in image
         depth_norm,       # [0, 1] - Z depth (from depth map)
         width_norm,       # [0, 1] - Bounding box width
         height_norm,      # [0, 1] - Bounding box height
         confidence,       # [0, 1] - Detection confidence
         class_id          # int - Object class ID
     ]
     ```

**Batch Output**:
```python
{
    'instructions': List[str],                          # B instruction strings
    'demo_3d_objects': List[List[Tensor]],             # B × num_demos × (N, 7)
    'current_3d_objects': List[Tensor],                # B × (N, 7)
    'demo_actions': List[Tensor],                      # num_demos × (B, 7)
    'targets': Tensor                                  # (B, 7) - ground truth actions
}
```

---

## 3. Data Flow Through Network

### Architecture Overview

```
Raw Inputs
    ↓
[Text Encoder] → Instruction Embedding (B, 768)
[Object Encoder] → Object Embeddings (B, N, 256)
[Sequence Builder] → Multimodal Sequence (B, seq_len, 256)
    ↓
[Policy Transformer] → Decision Vector (B, 512)
    ↓
[Output Heads] → 7 Logit Vectors (B, bins_i)
```

### Step-by-Step Forward Pass

#### Step 1: Text Encoding
**Input**: `instructions: List[str]` (B strings)  
**Process**: BERT encoder (frozen)  
**Output**: `instr_embed: (B, 768)`

```python
TextEncoder.encode(["Pick up the star..."]) 
→ torch.tensor([[0.1, 0.3, ..., 0.7]])  # (1, 768)
```

#### Step 2: Object Encoding
**Input**: `demo_3d_objects: List[List[Tensor]]`, `current_3d_objects: List[Tensor]`  
**Process**: ObjectEncoder MLP (trainable)  
**Output**: Object embeddings

**For each batch element**:
- Demo objects: `(num_demos, N, 7)` → `(num_demos, N, 256)`
- Current objects: `(N, 7)` → `(N, 256)`

**Padding**: Variable number of objects per image → padded to `(B, max_objs, 256)`

```python
ObjectEncoder(
    torch.tensor([[0.5, 0.3, 0.2, 0.1, 0.05, 0.9, 42]])  # (1, 7)
)
→ torch.tensor([[0.1, 0.2, ..., 0.8]])  # (1, 256)
```

#### Step 3: Sequence Building
**Input**: 
- Instruction embedding: `(B, 768)`
- Demo object embeddings: List of `(B, max_objs, 256)`
- Demo actions: List of `(B, 7)`
- Current object embeddings: `(B, max_objs, 256)`

**Process**: MultimodalSequenceBuilder (trainable)
1. Project instruction: `(B, 768)` → `(B, 256)`
2. Project demo actions: `(B, 7)` → `(B, 256)`
3. Project object embeddings: `(B, N, 256)` → `(B, N, 256)` (identity if already 256)
4. Concatenate in sequence:
   ```
   [instruction_token,
    demo1_objects (N tokens),
    demo1_action (1 token),
    demo2_objects (N tokens),
    demo2_action (1 token),
    ...,
    current_objects (N tokens)]
   ```
5. Pad to fixed length: `(B, max_seq_len, 256)`

**Output**: `tokens: (B, max_seq_len, 256)`

**Example Sequence** (for one batch element):
```
Token 0:  [instruction embedding]                    # (256,)
Token 1:  [object 1 from demo 1]                    # (256,)
Token 2:  [object 2 from demo 1]                    # (256,)
Token 3:  [action from demo 1]                      # (256,)
Token 4:  [object 1 from demo 2]                    # (256,)
Token 5:  [action from demo 2]                      # (256,)
Token 6:  [object 1 from current scene]             # (256,)
Token 7:  [object 2 from current scene]             # (256,)
Token 8+: [padding zeros]                           # (256,)
```

#### Step 4: Policy Reasoning
**Input**: `tokens: (B, max_seq_len, 256)`  
**Process**: PolicyTransformer (trainable)
- Multi-head self-attention
- Feed-forward layers
- Layer normalization

**Output**: `decision: (B, 512)` - Single decision vector per batch element

#### Step 5: Output Heads
**Input**: `decision: (B, 512)`  
**Process**: 7 independent linear heads (trainable)

**Output**: List of 7 logit vectors
```python
[
    logits_x: (B, 101),      # 101 bins for x position
    logits_y: (B, 101),      # 101 bins for y position
    logits_z: (B, 101),      # 101 bins for z position
    logits_roll: (B, 121),   # 121 bins for roll
    logits_pitch: (B, 121),  # 121 bins for pitch
    logits_yaw: (B, 121),    # 121 bins for yaw
    logits_gripper: (B, 2)   # 2 bins for gripper (open/closed)
]
```

---

## 4. Training Pipeline

### Training Loop (`train()` function)

#### Initialization
```python
# Model
model = AgentModel(
    token_dim=256,
    out_dim=512,
    bins=[101, 101, 101, 121, 121, 121, 2],
    device="cuda"
)

# Optimizer (only trainable params)
trainable_params = [
    model.object_enc.parameters(),      # Object encoder
    model.seq_builder.parameters(),     # Sequence builder
    model.policy.parameters(),          # Policy transformer
    model.heads.parameters()            # Output heads
]
optimizer = Adam(trainable_params, lr=1e-4)

# Frozen: TextEncoder (BERT), VisionEncoder (CLIP) - not in optimizer
```

#### Per Epoch

**Training Phase**:
```python
for batch in train_dataloader:
    # Forward pass
    logits = model.forward(
        instructions=batch['instructions'],
        demo_3d_objects=batch['demo_3d_objects'],
        current_3d_objects=batch['current_3d_objects'],
        demo_actions=batch['demo_actions']
    )
    # logits: List of 7 tensors, each (B, bins_i)
    
    # Compute loss
    targets = batch['targets']  # (B, 7)
    loss = model.heads.loss(logits, targets)
    # Loss: Average CrossEntropyLoss across 7 dimensions
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Validation Phase**:
```python
for batch in val_dataloader:
    with torch.no_grad():
        logits = model.forward(...)
        preds = model.heads.predict(logits)  # (B, 7) - argmax per dimension
        targets = batch['targets']  # (B, 7)
        
        # Compute accuracy per dimension
        for dim in range(7):
            correct[dim] += (preds[:, dim] == targets[:, dim]).sum()
        
        # Save predictions for analysis
        all_predictions.append(preds)
        all_targets.append(targets)
```

**Early Stopping**:
- Monitor validation loss
- If no improvement for `patience=5` epochs → stop training
- Save best checkpoint when validation improves

**Checkpoint Saving**:
- After each epoch: `checkpoints/agent_epoch{N}.pt`
- Best model: `checkpoints/agent_best.pt`
- Metrics: `logs/training_metrics_epoch_{N}.json`
- Predictions: `logs/predictions_epoch_{N}.npz`

---

## 5. Loss Function

### CrossEntropyLoss (Multi-class Classification)

**Per Dimension**:
```python
for i in range(7):  # For each action dimension
    logits_i = logits[i]  # (B, bins[i])
    targets_i = targets[:, i]  # (B,)
    
    # Clamp invalid targets
    valid_mask = (targets_i >= 0) & (targets_i < bins[i])
    targets_i[~valid_mask] = -1  # Mark as invalid
    
    # Compute loss (ignores -1 targets)
    loss_i = CrossEntropyLoss(ignore_index=-1)(logits_i, targets_i)
    
    total_loss += loss_i

final_loss = total_loss / 7  # Average across dimensions
```

**Key Features**:
- **7 independent classification heads** (one per action dimension)
- **Ignore invalid targets** (`-1` values)
- **Target validation** (clamp out-of-range values)
- **Averaged loss** across all 7 dimensions

**Bins**:
- x, y, z: 101 bins each (0-100)
- roll, pitch, yaw: 121 bins each (0-120, each bin = 3°)
- gripper: 2 bins (0=closed, 1=open)

---

## 6. Metrics

### During Training (Per Epoch)

**Saved to**: `logs/training_metrics_epoch_{N}.json`

**Metrics**:
- **Train Loss**: Average loss across training batches
- **Validation Loss**: Average loss across validation batches
- **Per-Dimension Accuracy**: Accuracy for each of 7 action dimensions
  - x, y, z accuracy
  - roll, pitch, yaw accuracy
  - gripper accuracy
- **Average Accuracy**: Mean of all 7 dimension accuracies

**Predictions Saved**: `logs/predictions_epoch_{N}.npz`
- `predictions`: (N, 7) numpy array
- `targets`: (N, 7) numpy array
- `instructions`: List of instruction strings

### After Training (Comprehensive Evaluation)

**Saved to**: `logs/metrics_YYYYMMDD_HHMMSS.json` and `.csv`

**5 Comprehensive Metrics**:

1. **Task Success Rate** (0.0-1.0)
   - Percentage of tasks completed successfully
   - Based on full trajectory evaluation

2. **Subgoal Success Rate** (0.0-1.0)
   - Fraction of intermediate goals completed correctly
   - For multi-step tasks

3. **Planner Steps** (integer)
   - Average number of model inferences per task
   - How many planning calls needed

4. **Environment Steps** (integer)
   - Average number of environment interactions per task
   - How many actions executed

5. **Error Analysis** (dict)
   - Categorization of failures:
     - `perception`: Position errors (x, y, z)
     - `orientation`: Rotation errors (roll, pitch, yaw)
     - `gripper`: Gripper state errors
     - `planning`: Multiple dimension errors
     - `invalid_target`: Invalid target values

**Computed by**: `evaluate_trajectory.py` (runs automatically after training)

---

## 7. Data Flow Summary

```
JSON Dataset
    ↓
[EmbodiedDataset] → Raw images + actions
    ↓
[collate_fn_3d] → 3D preprocessing
    ├─ [YOLOv8] → Object detections
    ├─ [MiDaS] → Depth maps
    └─ [Fusion] → 3D object representations (N, 7)
    ↓
[AgentModel.forward]
    ├─ [TextEncoder] → Instruction embedding (B, 768)
    ├─ [ObjectEncoder] → Object embeddings (B, N, 256)
    ├─ [SequenceBuilder] → Multimodal sequence (B, seq_len, 256)
    ├─ [PolicyTransformer] → Decision (B, 512)
    └─ [OutputHeads] → 7 logit vectors
    ↓
[Loss Computation] → CrossEntropyLoss per dimension
    ↓
[Backpropagation] → Update trainable parameters
    ↓
[Validation] → Compute accuracy metrics
    ↓
[Save] → Checkpoints + metrics + predictions
```

---

## 8. Key Design Decisions

### Why 3D Preprocessing?
- **Explicit spatial reasoning**: Model sees object positions in 3D space
- **Better manipulation**: Can reason about depth, not just 2D appearance
- **Robust to viewpoint**: 3D coordinates are viewpoint-invariant

### Why Multimodal Sequence?
- **In-context learning**: Model sees demonstrations + current state
- **Transformer architecture**: Self-attention can relate instruction → demos → current scene
- **Flexible length**: Handles variable numbers of demos and objects

### Why 7 Independent Heads?
- **Discrete action space**: Each dimension is a classification problem
- **Different scales**: Position (0-100) vs rotation (0-120) vs gripper (0-1)
- **Independent learning**: Each dimension can learn at different rates

### Why Frozen Encoders?
- **Pre-trained features**: BERT and CLIP provide strong representations
- **Faster training**: Only train policy and fusion components
- **Transfer learning**: Leverage large-scale pre-training

---

## 9. Example: Complete Forward Pass

**Input**:
- Instruction: `"Pick up the star"`
- Demo image: Shows picking up star
- Current image: Scene with star visible

**Processing**:
1. Detect 3 objects in demo image → 3 × (7,) → 3 × (256,) embeddings
2. Detect 2 objects in current image → 2 × (7,) → 2 × (256,) embeddings
3. Build sequence: `[instruction, demo_objs, demo_action, current_objs]`
4. Transformer processes sequence → `(512,)` decision vector
5. 7 heads output: `[logits_x, logits_y, ..., logits_gripper]`
6. Argmax per head → `[45, 32, 78, 0, 60, 90, 1]` (predicted action)

**Target**: `[45, 32, 78, 0, 60, 90, 1]`  
**Match**: ✓ All dimensions correct

---

This is the complete system from data to predictions!

