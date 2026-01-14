"""Simple orchestration pipeline that composes the components.

This pipeline uses placeholder implementations; replace with actual references
to the concrete module implementations when they are available.
"""
from typing import Any

from sagevision.video_parser import VideoParser
from sagevision.scene_detector import SceneDetector
from sagevision.keyframe_selector import KeyframeSelector
from sagevision.vision_captioner import VisionCaptioner
from sagevision.summarizer import Summarizer


class Pipeline:
    """High-level pipeline that runs the primary stages and returns a summary."""

    def __init__(self, *, captioner: VisionCaptioner = None, summarizer: Summarizer = None):
        self.captioner = captioner or VisionCaptioner()
        self.summarizer = summarizer or Summarizer()

    def run(self, video_path: str, progress_callback=None) -> str:
        """Run the end-to-end pipeline on `video_path` and return final summary.

        Args:
            video_path: path to the input video
            progress_callback: Optional callable(stage: str, percent: float, message: str)
                Called at key points to report progress. Percent is in [0.0, 1.0].
        """

        def _cb(stage: str, percent: float, message: str = ""):
            if callable(progress_callback):
                try:
                    progress_callback(stage, percent, message)
                except Exception:
                    # Don't let callback errors break the pipeline
                    pass

        # 1. Parse video -> frames
        _cb("parsing", 0.0, "opening video")
        parser = VideoParser(video_path)
        try:
            frames_iterator = parser.frames()
        except Exception as e:
            _cb("parsing", 1.0, f"failed: {e}")
            return "[Pipeline error] failed to open video"
        _cb("parsing", 1.0, "done")

        # 2. Scene detection (we collect frames into memory for now)
        _cb("scene_detection", 0.0, "collecting frames")
        frames = list(frames_iterator)
        detector = SceneDetector()
        try:
            boundaries = detector.detect_scenes(frames)
        except Exception as e:
            _cb("scene_detection", 1.0, f"failed: {e}")
            return "[Pipeline error] scene detection failed"
        _cb("scene_detection", 1.0, f"found {len(boundaries)} scene(s)")

        # 3. For each scene, select keyframes, caption and summarize
        selector = KeyframeSelector()
        scene_summaries = []
        total_scenes = max(1, len(boundaries))
        for i, start_idx in enumerate(boundaries):
            percent_base = i / total_scenes
            percent_step = 1.0 / total_scenes
            _cb(f"scene_{i}_start", percent_base, f"scene {i} starting at frame {start_idx}")

            end_idx = boundaries[i + 1] if (i + 1) < len(boundaries) else len(frames)
            scene_frames = frames[start_idx:end_idx]
            if not scene_frames:
                _cb(f"scene_{i}_done", percent_base + percent_step, "empty scene")
                continue
            # select keyframes from the scene
            try:
                key_idxs = selector.select(scene_frames)
            except Exception:
                # fallback: use the first frame
                key_idxs = [0]
            keyframes = [scene_frames[k] for k in key_idxs]
            # caption them
            captions = self.captioner.caption_batch(keyframes)
            # summarize scene
            scene_summary = self.summarizer.summarize(captions)
            scene_summaries.append(scene_summary)
            _cb(f"scene_{i}_done", percent_base + percent_step, f"scene {i} done")

        # 4. Final aggregation
        _cb("aggregation", 0.0, "aggregating scene summaries")
        if not scene_summaries:
            _cb("aggregation", 1.0, "no scenes")
            return ""
        final = self.summarizer.summarize(scene_summaries)
        _cb("aggregation", 1.0, "done")
        _cb("finished", 1.0, "pipeline complete")
        return final
