"""Video parser implementation using OpenCV.

Yields frames as PIL Images to be compatible with the VisionCaptioner and
CLIP processors. Supports optional frame skipping (sample every `step` frames)
for performance.
"""
from typing import Iterator, Any, Optional
from PIL import Image
import cv2
import numpy as np


class VideoParser:
    """OpenCV-backed video parser.

    Args:
        source: Path to a video file
        step: sample every `step` frames (default: 1, i.e., every frame)
    """

    def __init__(self, source: str, step: int = 1):
        self.source = source
        self.step = max(1, int(step))

    def frames(self) -> Iterator[Any]:
        """Yield decoded frames as PIL Image objects.

        Yields:
            PIL.Image instances in RGB mode.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Failed to open video source: {self.source}")

        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.step == 0:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame_rgb)
                    yield pil
                idx += 1
        finally:
            cap.release()
