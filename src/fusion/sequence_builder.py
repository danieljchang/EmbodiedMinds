import torch
import torch.nn as nn
from typing import List, Optional


class MultimodalSequenceBuilder(nn.Module):
    """
    Constructs the multimodal token sequence for the policy transformer.
    
    Sequence structure (per example in batch):
    [instruction_embedding, 
     demo1_objects, demo1_action,
     demo2_objects, demo2_action,
     ...,
     current_objects]
    """
    
    def __init__(
        self,
        token_dim: int = 256,
        instr_dim: int = 768,
        obj_dim: int = 256,
        action_dim: int = 7,
        vlm_vision_dim: Optional[int] = None,
    ):
        super().__init__()
        self.token_dim = token_dim
        
        # Projection layers
        # Text encoder / VLM output (instr_dim) -> token_dim
        self.instr_proj = nn.Linear(instr_dim, token_dim)
        # Optional VLM vision embedding -> token_dim
        self.vlm_vision_proj: Optional[nn.Linear] = (
            nn.Linear(vlm_vision_dim, token_dim) if vlm_vision_dim is not None else None
        )
        # 7D action -> token_dim (or configurable action_dim)
        self.action_proj = nn.Linear(action_dim, token_dim)
        # Object embeddings (obj_dim) -> token_dim
        self.obj_proj = nn.Linear(obj_dim, token_dim)
    
    def forward(
        self,
        instr_embedding: torch.Tensor,  # (B, instr_dim)
        demo_object_embeddings: List[torch.Tensor],  # list of (B, num_obj, obj_dim) per demo
        current_object_embeddings: torch.Tensor = None,  # (B, num_obj, obj_dim)
        demo_actions: Optional[List[torch.Tensor]] = None,  # list of (B, action_dim) per demo
        vlm_vision_embedding: Optional[torch.Tensor] = None,  # (B, vlm_vision_dim)
    ) -> torch.Tensor:
        """
        Build multimodal sequence for transformer.
        
        Args:
            instr_embedding: (B, instr_dim) instruction embeddings
            demo_object_embeddings: List of (B, num_obj, obj_dim) tensors, one per demo
            current_object_embeddings: (B, num_obj, obj_dim) current scene objects
            demo_actions: Optional list of (B, action_dim) action tensors, one per demo
            vlm_vision_embedding: Optional (B, vlm_vision_dim) global image embeddings
            
        Returns:
            (B, max_seq_len, token_dim) tensor ready for transformer
        """
        B = instr_embedding.size(0)
        device = instr_embedding.device
        
        sequences = []
        max_seq_len = 0
        
        for b in range(B):
            seq = []
            
            # 1. Instruction token
            instr_token = self.instr_proj(instr_embedding[b:b+1])  # (1, token_dim)
            seq.append(instr_token)

            # 1b. Optional global VLM vision token (e.g., from current RGB frame)
            if (
                vlm_vision_embedding is not None
                and self.vlm_vision_proj is not None
                and vlm_vision_embedding.size(0) == B
            ):
                vision_token = self.vlm_vision_proj(vlm_vision_embedding[b:b+1])
                seq.append(vision_token)
            
            # 2. Demo sequences
            num_demos = len(demo_object_embeddings) if demo_object_embeddings else 0
            for demo_idx in range(num_demos):
                # Demo objects
                demo_objs = demo_object_embeddings[demo_idx][b]  # (num_obj, 256)
                if demo_objs.size(0) > 0:
                    demo_obj_tokens = self.obj_proj(demo_objs)  # (num_obj, token_dim)
                    seq.append(demo_obj_tokens)
                
                # Demo action (if provided)
                if demo_actions is not None and demo_idx < len(demo_actions):
                    demo_action = demo_actions[demo_idx][b:b+1]  # (1, 7)
                    action_token = self.action_proj(demo_action.float())  # (1, token_dim)
                    seq.append(action_token)
            
            # 3. Current objects
            if current_object_embeddings is not None:
                cur_objs = current_object_embeddings[b]  # (num_obj, 256)
                if cur_objs.size(0) > 0:
                    cur_obj_tokens = self.obj_proj(cur_objs)  # (num_obj, token_dim)
                    seq.append(cur_obj_tokens)
            
            # Concatenate all tokens for this example
            if len(seq) > 0:
                seq_tensor = torch.cat(seq, dim=0)  # (total_tokens, token_dim)
            else:
                # Fallback: just instruction token
                seq_tensor = instr_token
            
            sequences.append(seq_tensor)
            max_seq_len = max(max_seq_len, seq_tensor.size(0))
        
        # Pad sequences to same length
        if max_seq_len == 0:
            max_seq_len = 1
        
        padded_sequences = []
        for seq in sequences:
            if seq.size(0) < max_seq_len:
                padding = torch.zeros(
                    max_seq_len - seq.size(0), 
                    self.token_dim, 
                    device=device
                )
                padded = torch.cat([seq, padding], dim=0)
            else:
                padded = seq
            padded_sequences.append(padded)
        
        return torch.stack(padded_sequences, dim=0)  # (B, max_seq_len, token_dim)

