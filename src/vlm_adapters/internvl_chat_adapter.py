from typing import List, Optional

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel, AutoProcessor

from src.vlm_adapters.base import VLMAdapterBase


class InternVLChatAdapter(VLMAdapterBase):
    """
    Adapter for chat-style InternVL models (e.g., OpenGVLab/InternVL3-8B).

    These models are implemented as multimodal chat models whose main
    `forward` requires both text and images. We therefore expose a
    multimodal encoding interface and treat the resulting fused embedding
    as our instruction representation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.max_length = max_length

        # Processor handles both text and images.
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Load the full chat model with remote code.
        self.model: AutoModel = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

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

    # ------------------------------------------------------------------
    # Text-only API is not supported for this adapter
    # ------------------------------------------------------------------
    @property
    def supports_text_only(self) -> bool:
        return False

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        This adapter requires images; text-only encoding is not supported.
        """
        raise NotImplementedError(
            "InternVLChatAdapter requires images; use encode_multimodal instead."
        )

    # ------------------------------------------------------------------
    # Multimodal encoding (text + images)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_multimodal(
        self,
        texts: List[str],
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode (text, image) pairs into fused embeddings.

        Args:
            texts: List of instruction strings (length B).
            images: List of (3, H, W) RGB tensors (length B).

        Returns:
            Tensor of shape (B, D) where D == text_hidden_size.
        """
        if len(texts) == 0:
            return torch.zeros(0, self.text_hidden_size, device=self.device)
        if len(texts) != len(images):
            raise ValueError(
                f"encode_multimodal expected same number of texts and images, "
                f"got {len(texts)} texts and {len(images)} images."
            )

        # Convert images to PIL for the processor.
        pil_images = []
        for img in images:
            if img.dim() != 3 or img.size(0) != 3:
                raise ValueError(
                    "encode_multimodal expects each image as a (3, H, W) RGB tensor."
                )
            img_cpu = img.detach().cpu()
            # Normalize to uint8 [0, 255] if necessary.
            if img_cpu.max() <= 1.5:
                img_cpu = (img_cpu * 255.0).clamp(0, 255)
            pil_images.append(to_pil_image(img_cpu.to(torch.uint8)))

        inputs = self.processor(
            text=texts,
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Heuristic: prefer pooler_output, else CLS token from last_hidden_state.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            fused = outputs.pooler_output
        else:
            fused = outputs.last_hidden_state[:, 0, :]

        return fused


