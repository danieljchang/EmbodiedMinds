import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPFeatureExtractor


class VisionEncoder(nn.Module):
    """
    Vision encoder using CLIP's ViT backbone (frozen).

    This class is a drop-in replacement for the previous ResNet-based
    `VisionEncoder`. It accepts image tensors of shape (B, 3, H, W)
    (either in [0,255] uint8 or float in [0,1]) and returns a pooled
    CLS embedding (B, out_dim).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        super().__init__()
        # Load full CLIP model and use its vision submodule
        # Use safetensors to avoid torch.load version requirement
        import os
        os.environ['TRANSFORMERS_SAFE_LOADING'] = '1'
        try:
            clip = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        except:
            # Fallback if safetensors not available
            clip = CLIPModel.from_pretrained(model_name)
        self.vision = clip.vision_model
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name)

        # Freeze vision weights
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

        # output dimension (hidden size of ViT)
        self.out_dim = self.vision.config.hidden_size

        self.device = device
        self.to(device)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images to CLIP ViT CLS embeddings.

        Args:
            images: tensor (B, 3, H, W). Values may be uint8 0-255 or float 0-1.

        Returns:
            Tensor of shape (B, out_dim) containing the CLS embedding.
        """
        x = images.to(self.device)

        # normalize input range to [0, 1]
        if x.dtype == torch.uint8 or float(x.max()) > 1.0:
            x = x.float() / 255.0

        # apply CLIP feature-extractor normalization (mean/std)
        mean = torch.tensor(self.feature_extractor.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.feature_extractor.image_std, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            # forward through vision model; expects keyword 'pixel_values'
            outputs = self.vision(pixel_values=x)
            # last_hidden_state: (B, num_patches+1, hidden_dim)
            last_hidden = outputs.last_hidden_state
            # take the CLS token (first token)
            cls = last_hidden[:, 0, :]

        return cls  # (B, out_dim)