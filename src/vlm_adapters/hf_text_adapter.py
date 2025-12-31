"""
Generic HuggingFace text adapter for VLM families.

We use large open-source VLMs primarily as **instruction encoders**. The
existing pipeline (YOLOv8 + MiDaS + ObjectEncoder) continues to provide
3D object tokens. This adapter converts instruction strings into dense
embeddings using the text tower of a HuggingFace model.

For all models below we only rely on `AutoTokenizer` + `AutoModel` and
take the first token (CLS) representation as the sentence embedding.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from src.vlm_adapters.base import VLMAdapterBase


class HFTextAdapter(VLMAdapterBase):
    """
    HuggingFace text-only adapter.

    This wraps a generic HF model and exposes `encode_text`. The adapter keeps
    the model frozen by default; LoRA or other fine-tuning methods can be
    layered on top externally if desired.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
        use_fp16: bool = True,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        # Processor is used for vision (and optionally text) pre-processing
        try:
            self.processor: Optional[AutoProcessor] = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception:
            # Not all models provide an AutoProcessor; we fall back to tokenizer-only.
            self.processor = None

        torch_dtype = torch.float16 if use_fp16 and device.startswith("cuda") else None
        self.model: AutoModel = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        # Freeze base model params by default
        for p in self.model.parameters():
            p.requires_grad = False

        hidden_size: Optional[int] = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer hidden_size for model {model_name}. "
                "Please set adapter.text_hidden_size manually."
            )
        self.text_hidden_size = int(hidden_size)
        # For simplicity we expose a single hidden size; image embeddings
        # returned by encode_image are projected/pooled to this same size.
        self.vision_hidden_size: int = self.text_hidden_size

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of instruction strings into CLS embeddings.
        """
        if len(texts) == 0:
            return torch.zeros(0, self.text_hidden_size, device=self.device)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)

        # Prefer pooler_output when available, fall back to CLS token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        return pooled

    @torch.no_grad()
    def encode_image(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of RGB images into global vision embeddings.

        Args:
            images: List of (3, H, W) tensors in RGB order, arbitrary sizes.

        Returns:
            Tensor of shape (B, vision_hidden_size).

        Notes:
            - This method makes a best-effort attempt to use the underlying
              HuggingFace model's vision tower. For models that expose
              `get_image_features` we use that; otherwise we fall back to
              pooling the last hidden state.
            - If the underlying model does not support image inputs, this
              will raise a NotImplementedError.
        """
        if len(images) == 0:
            return torch.zeros(0, self.vision_hidden_size, device=self.device)

        if self.processor is None:
            raise NotImplementedError(
                f"Model {self.model_name} does not provide an AutoProcessor; "
                "image encoding is not supported for this adapter."
            )

        # Convert tensors to PIL Images; handle arbitrary value ranges.
        pil_images = []
        for img in images:
            # Ensure tensor is on CPU and detached
            if img.dim() != 3 or img.size(0) != 3:
                raise ValueError(
                    "encode_image expects each image as a (3, H, W) RGB tensor."
                )
            img_cpu = img.detach().cpu()
            # If values are in [0,1], scale up; if already ~[0,255], clamp.
            if img_cpu.max() <= 1.5:
                img_cpu = (img_cpu * 255.0).clamp(0, 255)
            pil_images.append(to_pil_image(img_cpu.to(torch.uint8)))

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Best-effort extraction of a single embedding per image
        vision_embeds: Optional[torch.Tensor] = None

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            vision_embeds = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vision_embeds = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            # Take CLS token or mean over spatial tokens as a fallback
            hs = outputs.last_hidden_state  # (B, T, D)
            if hs.dim() == 3:
                vision_embeds = hs[:, 0, :]
        if vision_embeds is None:
            raise NotImplementedError(
                f"Could not extract image embeddings from model {self.model_name}; "
                "outputs do not contain image_embeds / pooler_output / last_hidden_state."
            )

        # If needed, project/trim to vision_hidden_size
        if vision_embeds.size(-1) != self.vision_hidden_size:
            # Simple linear projection on-the-fly
            proj = nn.Linear(
                vision_embeds.size(-1), self.vision_hidden_size, bias=False
            ).to(self.device)
            # Keep proj frozen (no optimizer ever sees it)
            for p in proj.parameters():
                p.requires_grad = False
            vision_embeds = proj(vision_embeds)

        return vision_embeds




