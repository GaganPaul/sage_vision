from PIL import Image

from sagevision.pipeline import Pipeline


class DummyCaptioner:
    def caption(self, image):
        return "a test"

    def caption_batch(self, images):
        return ["a test" for _ in images]


class DummySummarizer:
    def summarize(self, texts):
        # concatenate for predictability
        return " | ".join(texts)


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    # Create a short synthetic video file using OpenCV
    import cv2
    import numpy as np

    path = tmp_path / "demo.mp4"
    size = (64, 48)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(6):
        frame = np.full((size[1], size[0], 3), ((i * 30) % 255, 10, 10), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Run pipeline with dummy components
    p = Pipeline(captioner=DummyCaptioner(), summarizer=DummySummarizer())
    summary = p.run(str(path))
    assert isinstance(summary, str)
    assert len(summary) > 0
