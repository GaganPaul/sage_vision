"""Lightweight summarizer using a seq2seq model (BART by default).

This wrapper uses `facebook/bart-large-cnn` by default. The model is
lazy-loaded and executed on CPU/MPS/CUDA depending on availability.
"""
from typing import Iterable, List, Optional

import torch


class Summarizer:
    """Aggregate or compress a list of textual items into a short summary.

    Args:
        model_name: Hugging Face model (default: facebook/bart-large-cnn)
        device: Optional torch.device; if None auto-detects CUDA > MPS > CPU
        max_length: maximum tokens in generated summary
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[torch.device] = None, max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or self._auto_device()
        self._pipeline = None

    def _auto_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline

        # Use transformers pipeline for convenience
        device_arg = -1
        if self.device.type == "cuda":
            device_arg = 0
        # transformers supports torch.device('mps') as device arg in recent versions
        try:
            self._pipeline = pipeline("summarization", model=self.model_name, device=device_arg)
        except Exception:
            # Fallback to CPU pipeline
            self.device = torch.device("cpu")
            self._pipeline = pipeline("summarization", model=self.model_name, device=-1)

    def summarize(self, texts: Iterable[str]) -> str:
        texts = list(texts)
        if not texts:
            return ""
        self._load()
        # Simple heuristic: concatenate short captions and ask the model to compress
        joined = "\n".join(texts)
        out = self._pipeline(joined, max_length=self.max_length, truncation=True)
        if isinstance(out, list) and out:
            return out[0].get("summary_text", "")
        return ""
