"""Basic tests to ensure modules import and classes instantiate."""

from sagevision.video_parser import VideoParser
from sagevision.scene_detector import SceneDetector
from sagevision.keyframe_selector import KeyframeSelector
from sagevision.vision_captioner import VisionCaptioner
from sagevision.summarizer import Summarizer
from sagevision.pipeline import Pipeline


def test_instantiation():
    assert VideoParser
    assert SceneDetector
    assert KeyframeSelector
    assert VisionCaptioner
    assert Summarizer
    Pipeline()
