from sagevision.pipeline import Pipeline


class DummyCaptioner:
    def caption(self, image):
        return "a"

    def caption_batch(self, images):
        return ["a" for _ in images]


class DummySummarizer:
    def summarize(self, texts):
        return " | ".join(texts)


def test_pipeline_emits_progress(tmp_path):
    # Create a short synthetic video file using OpenCV
    import cv2
    import numpy as np

    path = tmp_path / "demo.mp4"
    size = (32, 24)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(6):
        frame = np.full((size[1], size[0], 3), ((i * 30) % 255, 10, 10), dtype=np.uint8)
        out.write(frame)
    out.release()

    events = []

    def cb(stage, percent, message):
        events.append((stage, round(percent, 2), message))

    p = Pipeline(captioner=DummyCaptioner(), summarizer=DummySummarizer())
    res = p.run(str(path), progress_callback=cb)
    # Expect at least parsing, scene_detection, aggregation and finished events
    stages = {e[0] for e in events}
    assert "parsing" in stages
    assert "scene_detection" in stages
    assert "aggregation" in stages
    assert "finished" in stages
    assert isinstance(res, str)
