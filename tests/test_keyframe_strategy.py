import numpy as np
from sagevision.keyframe_selector import (
    KeyframeSelector,
    KeyframeStrategy,
)

class DummyStrategy(KeyframeStrategy):
    def select(self, frames, n_keyframes):
        return [0] if frames else []

# Custom Strategy Check
def test_custom_strategy_used():
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    selector = KeyframeSelector(n_keyframes=3, strategy=DummyStrategy())
    result = selector.select(frames)
    assert result == [0]

# Default Strategy Check
def test_default_strategy_instance():
    selector = KeyframeSelector(n_keyframes=3)
    assert selector.strategy is not None
