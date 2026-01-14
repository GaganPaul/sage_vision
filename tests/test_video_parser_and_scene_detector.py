import os
from PIL import Image
import numpy as np

from sagevision.video_parser import VideoParser
from sagevision.scene_detector import SceneDetector


def _make_test_video(path: str, num_frames: int = 10, size=(64, 48)):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10.0, size)
    for i in range(num_frames):
        # alternate colors to create scene changes
        if i < num_frames // 2:
            frame = np.full((size[1], size[0], 3), (i * 2 % 255, 10, 10), dtype=np.uint8)
        else:
            frame = np.full((size[1], size[0], 3), (10, i * 3 % 255, 20), dtype=np.uint8)
        out.write(frame)
    out.release()


def test_video_parser_reads(tmp_path):
    p = tmp_path / "test.mp4"
    _make_test_video(str(p), num_frames=6)
    parser = VideoParser(str(p))
    frames = list(parser.frames())
    assert len(frames) == 6
    assert isinstance(frames[0], Image.Image)


def test_scene_detector_detects_boundaries(tmp_path):
    # Create simple frames: half red-ish then half green-ish
    frames = []
    for i in range(8):
        if i < 4:
            arr = np.full((48, 64, 3), (200, 10, 10), dtype=np.uint8)
        else:
            arr = np.full((48, 64, 3), (10, 200, 10), dtype=np.uint8)
        frames.append(Image.fromarray(arr))

    det = SceneDetector(threshold=0.1, min_scene_len=1)
    bounds = det.detect_scenes(frames)
    # Expect a boundary at index 0 and at index 4
    assert 0 in bounds
    assert 4 in bounds
