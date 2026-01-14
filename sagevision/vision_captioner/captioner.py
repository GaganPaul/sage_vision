"""Image-to-text captioner using BLIP (Hugging Face).

This wrapper uses `Salesforce/blip-image-captioning-base` by default and
supports CPU, MPS, and CUDA devices. Models are lazy-loaded on first use
so imports and downloads don't happen at module import time.
"""
from typing import Any, List, Optional

import torch


class VisionCaptioner:
    """Generate captions for images using a BLIP model.

    Args:
        model_name: Hugging Face model name (default: Salesforce/blip-image-captioning-base)
        device: Optional torch.device; if None auto-detects CUDA > MPS > CPU
        max_length: Maximum tokens to generate
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[torch.device] = None, max_length: int = 64):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or self._auto_device()
        self._model = None
        self._processor = None

    def _auto_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self):
        if self._model is not None:
            return
        # Lazy import to avoid heavy deps at module import time
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        # Move to chosen device
        try:
            self._model.to(self.device)
        except Exception:
            # Some devices may not be supported for certain backends; fall back to CPU
            self.device = torch.device("cpu")
            self._model.to(self.device)

    def caption(self, image: Any) -> str:
        """Return a short caption for `image`.

        The `image` can be a PIL Image, numpy array or similar types accepted
        by the BLIP processor.
        """
        self._load()
        inputs = self._processor(images=image, return_tensors="pt")
        # Move tensors to the model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self._model.generate(**inputs, max_new_tokens=self.max_length)
        # Decode using the tokenizer associated with the processor
        caption = self._processor.tokenizer.decode(out[0], skip_special_tokens=True)
        return caption

    def caption_batch(self, images: List[Any]) -> List[str]:
        return [self.caption(img) for img in images]
