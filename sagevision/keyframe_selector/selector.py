"""
Modular keyframe selection with pluggable strategies.

Default strategy: CLIP embeddings + farthest-point sampling.
"""

from typing import Iterable, Any, List, Optional
from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from PIL import Image


# ============================================================
# Strategy Interface
# ============================================================

class KeyframeStrategy(ABC):
    """Base interface for keyframe selection strategies."""

    @abstractmethod
    def select(self, frames: List[Any], n_keyframes: int) -> List[int]:
        """Return indices of selected keyframes."""
        pass


# ============================================================
# Default Strategy: CLIP + Farthest Point Sampling
# ============================================================

class CLIPFarthestPointStrategy(KeyframeStrategy):
    """CLIP-based embedding diversity sampling."""

    def __init__(
        self,
        model_name: str = "laion/CLIP-ViT-B-32",
        device: Optional[torch.device] = None,
        token: Optional[str] = None,
        use_open_clip: bool = False,
    ):
        self.model_name = model_name
        self.device = device or self._auto_device()
        self.token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.use_open_clip = use_open_clip or self.model_name.startswith("hf-hub:")
        self._model = None
        self._processor = None
        self._oc_model = None
        self._oc_preprocess = None
        self._using_open_clip = False

    def _auto_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self):
        if self._model is not None or self._oc_model is not None:
            return

        if self.use_open_clip:
            try:
                import open_clip
                repo = self.model_name
                self._oc_model, _, self._oc_preprocess = open_clip.create_model_and_transforms(repo)
                self._oc_model.eval()
                self._oc_model.to(self.device)
                self._using_open_clip = True
                return
            except Exception as e:
                print(f"Warning: open_clip failed ({e}), falling back to transformers.")

        from transformers import CLIPModel, CLIPProcessor

        load_kwargs = {}
        if self.token:
            load_kwargs["use_auth_token"] = self.token

        try:
            self._processor = CLIPProcessor.from_pretrained(self.model_name, **load_kwargs)
            self._model = CLIPModel.from_pretrained(self.model_name, **load_kwargs)
        except Exception:
            fallback = "openai/clip-vit-large-patch14"
            print(f"Falling back to {fallback}")
            self._processor = CLIPProcessor.from_pretrained(fallback)
            self._model = CLIPModel.from_pretrained(fallback)

        self._model.to(self.device)

    def _compute_embeddings(self, frames: List[Any]) -> np.ndarray:
        self._load()

        if self._using_open_clip:
            tensors = []
            for f in frames:
                img = f if not isinstance(f, np.ndarray) else Image.fromarray(f)
                t = self._oc_preprocess(img).unsqueeze(0)
                tensors.append(t)
            batch = torch.cat(tensors, dim=0).to(self.device)
            with torch.no_grad():
                feats = self._oc_model.encode_image(batch)
            embs = feats.cpu().numpy()
        else:
            batch_embeddings = []
            batch_size = 16
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self._model.get_image_features(**inputs)
                batch_embeddings.append(feats.cpu().numpy())
            embs = np.vstack(batch_embeddings)

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def _farthest_point_sampling(self, embs: np.ndarray, k: int) -> List[int]:
        n = embs.shape[0]
        if k >= n:
            return list(range(n))

        selected = [0]
        min_dists = 1.0 - np.dot(embs, embs[0])

        for _ in range(1, k):
            idx = int(np.argmax(min_dists))
            selected.append(idx)
            dists = 1.0 - np.dot(embs, embs[idx])
            min_dists = np.minimum(min_dists, dists)

        return selected

    def select(self, frames: List[Any], n_keyframes: int) -> List[int]:
        if not frames:
            return []

        n_keyframes = min(n_keyframes, len(frames))
        embs = self._compute_embeddings(frames)
        indices = self._farthest_point_sampling(embs, n_keyframes)
        indices.sort()
        return indices


# ============================================================
# Main Selector (Strategy Wrapper)
# ============================================================

class KeyframeSelector:
    """
    High-level keyframe selector with pluggable strategies.

    Args:
        n_keyframes: number of keyframes to select
        strategy: custom KeyframeStrategy (optional)
    """

    def __init__(
        self,
        n_keyframes: int = 5,
        strategy: Optional[KeyframeStrategy] = None,
    ):
        self.n_keyframes = max(1, n_keyframes)
        self.strategy = strategy or CLIPFarthestPointStrategy()

    def select(self, frames: Iterable[Any]) -> List[int]:
        frames = list(frames)
        if not frames:
            return []
        return self.strategy.select(frames, self.n_keyframes)
