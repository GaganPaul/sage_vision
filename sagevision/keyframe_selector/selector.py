"""Adaptive keyframe sampling using CLIP embeddings.

This implementation uses `laion/CLIP-ViT-B-32` by default to compute image
embeddings and selects a diverse subset using farthest-point sampling.
Models are lazy-loaded on first selection call to avoid heavy imports during
unit tests or simple imports.
"""
from typing import Iterable, Any, List, Optional

import math
import os
import numpy as np
import torch


class KeyframeSelector:
    """Select representative keyframes for a sequence of frames using CLIP.

    Args:
        model_name: Hugging Face CLIP model (default: laion/CLIP-ViT-B-32)
        device: Optional torch.device; if None auto-detects CUDA > MPS > CPU
        n_keyframes: number of keyframes to select when select() is called.
        token: optional Hugging Face token for gated models
        use_open_clip: if True, attempt to use the `open_clip` library and the
                       model id (supports `hf-hub:...` style ids)
    """

    def __init__(self, model_name: str = "laion/CLIP-ViT-B-32", device: Optional[torch.device] = None, n_keyframes: Optional[int] = None, token: Optional[str] = None, use_open_clip: bool = False):
        self.model_name = model_name
        self.device = device or self._auto_device()
        self.n_keyframes = n_keyframes
        # Hugging Face token (optional). If not provided, reads HUGGINGFACE_HUB_TOKEN from env.
        self.token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.use_open_clip = use_open_clip or self.model_name.startswith("hf-hub:")
        self._model = None
        self._processor = None
        # open_clip-specific fields
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

        # If requested, try using open_clip first (supports hf-hub: ids)
        if self.use_open_clip:
            try:
                import open_clip

                # Users may pass a model id like 'hf-hub:laion/CLIP-ViT-B-32...'
                repo = self.model_name
                self._oc_model, _, self._oc_preprocess = open_clip.create_model_and_transforms(repo)
                self._oc_model.eval()
                try:
                    self._oc_model.to(self.device)
                except Exception:
                    self._oc_model.to(torch.device("cpu"))
                self._using_open_clip = True
                return
            except Exception as e:
                # If open_clip fails (not installed or model gated), fall back
                print(f"Warning: open_clip unavailable or failed for '{self.model_name}' ({e}); falling back to transformers CLIP.")
                self._using_open_clip = False

        from transformers import CLIPModel, CLIPProcessor

        # Allow using an auth token for gated models; fall back to an accessible model
        load_kwargs = {}
        if self.token:
            load_kwargs["use_auth_token"] = self.token

        try:
            self._processor = CLIPProcessor.from_pretrained(self.model_name, **load_kwargs)
            self._model = CLIPModel.from_pretrained(self.model_name, **load_kwargs)
        except Exception as e:
            # If the configured model is not accessible (e.g., gated or private),
            # fall back to a widely available CLIP model.
            fallback = "openai/clip-vit-large-patch14"
            print(f"Warning: failed to load '{self.model_name}' ({e}). Falling back to '{fallback}'.")
            self.model_name = fallback
            self.token = None
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)

        try:
            self._model.to(self.device)
        except Exception:
            self.device = torch.device("cpu")
            self._model.to(self.device)

    def _compute_embeddings(self, frames: List[Any]) -> np.ndarray:
        """Compute normalized image embeddings for a list of frames."""
        self._load()
        if self._using_open_clip:
            # Use open_clip preprocess and model
            tensors = []
            for f in frames:
                img = f if not isinstance(f, np.ndarray) else Image.fromarray(f)
                t = self._oc_preprocess(img).unsqueeze(0)  # add batch dim
                tensors.append(t)
            batch = torch.cat(tensors, dim=0)
            batch = batch.to(self.device)
            with torch.no_grad():
                feats = self._oc_model.encode_image(batch)
            embs = feats.cpu().numpy()
        else:
            batch_embeddings = []
            # Process in small batches to reduce memory pressure
            batch_size = 16
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt", padding=True)
                # Move tensors to device if possible
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self._model.get_image_features(**inputs)
                feats = feats.cpu().numpy()
                batch_embeddings.append(feats)
            embs = np.vstack(batch_embeddings)
        # L2-normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return embs

    def _farthest_point_sampling(self, embs: np.ndarray, k: int) -> List[int]:
        n = embs.shape[0]
        if k >= n:
            return list(range(n))
        selected = [0]
        # initialize min distances to the first selected point
        min_dists = 1.0 - np.dot(embs, embs[0])
        for _ in range(1, k):
            idx = int(np.argmax(min_dists))
            selected.append(idx)
            dists = 1.0 - np.dot(embs, embs[idx])
            min_dists = np.minimum(min_dists, dists)
        return selected

    def select(self, frames: Iterable[Any]) -> List[int]:
        """Return indices of selected keyframes from `frames`.

        Args:
            frames: iterable of images (PIL / numpy arrays acceptable by CLIP processor)
        """
        frames = list(frames)
        n = len(frames)
        if n == 0:
            return []
        k = self.n_keyframes or max(1, min(5, n //  max(1, n // 5)))
        # Fallback: choose up to 5 frames or a small fraction of total
        k = min(k, n)
        embs = self._compute_embeddings(frames)
        sel = self._farthest_point_sampling(embs, k)
        sel.sort()
        return sel
