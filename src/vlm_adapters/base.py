import abc
from typing import List, Optional

import torch
import torch.nn as nn


class VLMAdapterBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Base interface for Vision-Language Model (VLM) adapters.

    Each adapter is responsible for turning raw inputs (e.g., instruction strings
    and optionally RGB images) into continuous embeddings that the policy can use.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        # Concrete adapters must set this to the dimensionality of text (or text+vision) embeddings.
        self.text_hidden_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Text (or text-like) embeddings
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of instruction strings into embeddings.

        Args:
            texts: List of instruction strings, length B.

        Returns:
            Tensor of shape (B, D_text) on the adapter's device.
        """

    @property
    def supports_text_only(self) -> bool:
        """
        Whether this adapter can encode text without images.

        Some VLMs (e.g., certain chat-style models) require images for their
        main forward pass; they should override this to return False and
        implement `encode_multimodal` instead.
        """
        return True

    def encode_multimodal(
        self,
        texts: List[str],
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Optional multimodal encoding hook (text + images together).

        Default implementation is not provided; adapters that require both
        text and images to run (e.g., some chat-style VLMs) should override
        this method and set supports_text_only = False.
        """
        raise NotImplementedError("encode_multimodal not implemented for this adapter.")

    # ------------------------------------------------------------------
    # Image-only embeddings (optional)
    # ------------------------------------------------------------------
    def encode_image(self, images: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Optional image encoding hook.

        Args:
            images: List of image tensors, each (3, H, W) in RGB order.

        Returns:
            Tensor of shape (B, D_vision) on the adapter's device, or None.

        By default we raise NotImplementedError; concrete adapters that wish
        to add global image tokens should override this method.
        """
        raise NotImplementedError("Image encoding not implemented for this adapter.")



