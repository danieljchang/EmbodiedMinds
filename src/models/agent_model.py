import torch
import torch.nn as nn
from typing import List, Optional

from src.encoders.text_encoder import TextEncoder
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.object_encoder import ObjectEncoder
from src.fusion.sequence_builder import MultimodalSequenceBuilder
from src.policy.policy_transformer import PolicyTransformer
from src.heads.output_heads import OutputHeads
from src.vlm_adapters import build_vlm_adapter


class AgentModel(nn.Module):
    """
    Main agent model that combines all components for 3D spatial reasoning.
    """
    
    def __init__(
        self,
        token_dim: int = 256,
        out_dim: int = 512,
        bins: Optional[List[int]] = None,
        text_model_name: str = "bert-base-uncased",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        vlm_adapter_name: Optional[str] = None,
    ):
        super().__init__()
        if bins is None:
            bins = [101, 101, 101, 121, 121, 121, 2]
        
        self.device = device

        # VLM adapter (optional). If provided, this replaces the legacy
        # TextEncoder for instruction embeddings.
        self.vlm = None
        if vlm_adapter_name is not None:
            self.vlm = build_vlm_adapter(vlm_adapter_name, device=device)
            instr_dim = int(self.vlm.text_hidden_size)
            # We choose image embeddings to live in the same hidden size space when used.
            vlm_vision_dim: Optional[int] = instr_dim
        else:
            instr_dim = 768  # BERT default
            vlm_vision_dim = None
        
        # Frozen encoders (legacy baseline)
        # These are still constructed so existing configs keep working. When a
        # VLM adapter is used we rely on self.vlm.encode_text() instead.
        self.text_enc = TextEncoder(model_name=text_model_name, device=device)
        self.vision_enc = VisionEncoder(model_name=vision_model_name, device=device)
        
        # Object encoder (trainable)
        self.object_enc = ObjectEncoder(embedding_dim=256)
        
        # Sequence builder (trainable)
        self.seq_builder = MultimodalSequenceBuilder(
            token_dim=token_dim,
            instr_dim=instr_dim,
            obj_dim=256,
            action_dim=7,
            vlm_vision_dim=vlm_vision_dim,
        )
        
        # Policy & heads
        self.policy = PolicyTransformer(token_dim=token_dim, out_dim=out_dim)
        self.heads = OutputHeads(in_dim=out_dim, bins=bins)
        
        # Move all trainable modules to device
        self.object_enc = self.object_enc.to(device)
        self.seq_builder = self.seq_builder.to(device)
        self.policy = self.policy.to(device)
        self.heads = self.heads.to(device)
        
        # Also call to(device) on the whole model to ensure everything is moved
        self.to(device)
    
    def forward(
        self,
        instr_texts: List[str],
        demo_3d_objects: List[List[torch.Tensor]],
        current_3d_objects: List[torch.Tensor],
        demo_actions: Optional[List[torch.Tensor]] = None,
        current_images: Optional[List[torch.Tensor]] = None,
    ):
        """
        Forward pass through the model.
        
        Args:
            instr_texts: List of instruction strings (B,)
            demo_3d_objects: List of lists of 3D object tensors
                           Outer list: batch, inner list: demos
                           Each tensor: (num_objects, 7)
            current_3d_objects: List of 3D object tensors (B,)
                               Each tensor: (num_objects, 7)
            demo_actions: Optional list of action tensors (B, num_demos, 7)
            
        Returns:
            list of logits from output heads, each of shape (B, bins_i)
        """
        # Encode instruction (and possibly image) via VLM or legacy encoder.
        vlm_vision_embed = None
        if self.vlm is not None:
            # Some adapters (e.g., InternVLChatAdapter) require both text and images.
            if hasattr(self.vlm, "supports_text_only") and not self.vlm.supports_text_only:
                if current_images is None:
                    raise ValueError(
                        "Current RGB images are required for this VLM adapter but were not provided."
                    )
                instr_embed = self.vlm.encode_multimodal(instr_texts, current_images)
            else:
                instr_embed = self.vlm.encode_text(instr_texts)
                # Optional: encode current RGB image via VLM vision tower if available
                if current_images is not None:
                    try:
                        vlm_vision_embed = self.vlm.encode_image(current_images)
                    except NotImplementedError:
                        vlm_vision_embed = None
        else:
            instr_embed = self.text_enc.encode(instr_texts)  # (B, 768)

        # Ensure instruction embedding is on correct device
        instr_embed = instr_embed.to(self.device)
        if vlm_vision_embed is not None:
            vlm_vision_embed = vlm_vision_embed.to(self.device)
        
        # Encode 3D objects
        demo_obj_embeds = []
        # Find maximum number of demos across batch
        num_demos = max(len(demo_3d_objects[b]) for b in range(len(demo_3d_objects))) if demo_3d_objects and len(demo_3d_objects) > 0 else 0
        
        for demo_idx in range(num_demos):
            batch_objs = []
            for b in range(len(demo_3d_objects)):
                # Handle variable number of demos per example
                if demo_idx < len(demo_3d_objects[b]):
                    batch_objs.append(demo_3d_objects[b][demo_idx])
                else:
                    # Use empty tensor if this example has fewer demos
                    batch_objs.append(torch.zeros((0, 7), dtype=torch.float32))
            demo_obj_embeds.append(self._encode_object_batch(batch_objs))
        
        current_obj_embeds = self._encode_object_batch(current_3d_objects)
        
        # Move demo_actions to device if provided
        demo_actions_device = None
        if demo_actions is not None:
            demo_actions_device = [action.to(self.device) for action in demo_actions]
        
        # Build sequence
        tokens = self.seq_builder(
            instr_embedding=instr_embed,
            demo_object_embeddings=demo_obj_embeds,
            current_object_embeddings=current_obj_embeds,
            demo_actions=demo_actions_device if demo_actions_device else None,
            vlm_vision_embedding=vlm_vision_embed,
        )  # (B, seq_len, token_dim)
        
        # Policy reasoning
        decision = self.policy(tokens)  # (B, out_dim)
        
        # Output heads
        logits = self.heads(decision)  # list of (B, bins_i)
        
        return logits
    
    def _encode_object_batch(self, object_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Handle variable-sized object sets per batch element.
        
        Args:
            object_list: List of (num_objects, 7) tensors, one per batch element
            
        Returns:
            (B, max_objs, 256) tensor with padded object embeddings
        """
        B = len(object_list)
        max_objs = max(o.size(0) if o.size(0) > 0 else 0 for o in object_list)
        
        if max_objs == 0:
            return torch.zeros(B, 1, 256, device=self.device)
        
        padded_objs = []
        for objs in object_list:
            # Move to device if not already there
            objs = objs.to(self.device)
            
            if objs.size(0) > 0:
                encoded = self.object_enc(objs)  # (num_obj, 256)
            else:
                encoded = torch.zeros(1, 256, device=self.device)
            
            # Pad to fixed size
            if encoded.size(0) < max_objs:
                pad = torch.zeros(max_objs - encoded.size(0), 256, device=self.device)
                encoded = torch.cat([encoded, pad], dim=0)
            
            padded_objs.append(encoded)
        
        return torch.stack(padded_objs, dim=0)  # (B, max_objs, 256)

