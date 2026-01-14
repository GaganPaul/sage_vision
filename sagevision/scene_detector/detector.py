"""Histogram-based scene / shot boundary detection.

This detector performs a simple histogram-difference heuristic to detect
scene boundaries. It is intentionally lightweight and suitable for CPU-only
machines.
"""
from typing import Iterable, Any, List, Optional
import numpy as np
from PIL import Image


class SceneDetector:
    """Detect scene boundaries using histogram differences.

    Args:
        threshold: float; histogram chi-square threshold to consider a scene
                   change (smaller -> more sensitive). Default 0.5.
        min_scene_len: minimum frames between boundaries to avoid short scenes.
    """

    def __init__(self, threshold: float = 0.5, min_scene_len: int = 5):
        self.threshold = threshold
        self.min_scene_len = max(1, int(min_scene_len))

    def _histogram(self, img: Image.Image) -> np.ndarray:
        # convert to RGB and compute per-channel histograms normalized
        img = img.convert("RGB")
        arr = np.array(img)
        hists = []
        for c in range(3):
            hist, _ = np.histogram(arr[..., c], bins=32, range=(0, 256), density=True)
            hists.append(hist)
        return np.concatenate(hists)

    def detect_scenes(self, frames: Iterable[Any]) -> List[int]:
        """Return a list of frame indices where a new scene starts (including 0).

        Args:
            frames: Iterable of PIL.Image or numpy arrays
        """
        frames = list(frames)
        n = len(frames)
        if n == 0:
            return []

        hists = []
        for f in frames:
            if not isinstance(f, Image.Image):
                f = Image.fromarray(np.array(f))
            hists.append(self._histogram(f))

        boundaries = [0]
        last_boundary = 0
        for i in range(1, n):
            # Chi-square distance
            h1 = hists[i - 1]
            h2 = hists[i]
            # avoid division by zero
            denom = (h1 + h2)
            denom[denom == 0] = 1.0
            chi2 = 0.5 * np.sum(((h1 - h2) ** 2) / denom)
            if chi2 >= self.threshold and (i - last_boundary) >= self.min_scene_len:
                boundaries.append(i)
                last_boundary = i
        return boundaries
