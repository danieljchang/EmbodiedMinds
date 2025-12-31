"""
Factory for building VLM adapters by name.

We currently use text-focused adapters for large VLM families listed in the
EmbodiedBench paper. Vision is still handled by YOLOv8 + MiDaS.

Supported logical adapter names:

  - "clip_bert"                (legacy baseline: TextEncoder/BERT)
  - "llama-3.2-11b-vision-ins"
  - "internvl-2.5-8b"
  - "internvl-3-8b"
  - "qwen2-vl-7b-ins"
  - "qwen2.5-vl-7b-ins"
  - "ovis2-16b"
  - "gemma-3-12b-it"

Note: The exact HuggingFace repo ids may need to be adjusted depending on
what you have access to. They are defined in the MODEL_REGISTRY dict below.
"""

from typing import Dict

from src.vlm_adapters.base import VLMAdapterBase
from src.vlm_adapters.hf_text_adapter import HFTextAdapter
from src.vlm_adapters.internvl_chat_adapter import InternVLChatAdapter


# Map logical adapter name -> HuggingFace model id
MODEL_REGISTRY: Dict[str, str] = {
    # Legacy baseline: we still use BERT through the existing TextEncoder,
    # so this entry is only for documentation and is not instantiated here.
    # "clip_bert": "bert-base-uncased",

    # Llama family
    "llama-3.2-11b-vision-ins": "meta-llama/Llama-3.2-11B-Vision-Instruct",

    # InternVL family
    "internvl-2.5-8b": "OpenGVLab/InternVL2_5-8B",
    "internvl-3-8b": "OpenGVLab/InternVL3-8B",

    # Qwen2 VL family
    "qwen2-vl-7b-ins": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b-ins": "Qwen/Qwen2.5-VL-7B-Instruct",

    # Ovis family
    "ovis2-16b": "BAAI/OVIS2-16B",

    # Gemma family (text-only, used as strong instruction encoder)
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}


def build_vlm_adapter(name: str, device: str = "cpu") -> VLMAdapterBase:
    """
    Build a VLM adapter by logical name.

    Args:
        name: Logical adapter name (see MODEL_REGISTRY keys).
        device: Torch device string.

    Returns:
        An instance of VLMAdapterBase.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown VLM adapter '{name}'. "
            f"Supported: {sorted(MODEL_REGISTRY.keys())}"
        )

    model_id = MODEL_REGISTRY[key]

    # Family-specific dispatch.
    if key == "internvl-3-8b":
        # Chat-style InternVL model that requires text + images together.
        return InternVLChatAdapter(model_name=model_id, device=device)

    # Default: generic HF text adapter.
    return HFTextAdapter(model_name=model_id, device=device)




