from PIL import Image
import numpy as np

import types
import sys

from sagevision.keyframe_selector import KeyframeSelector


def test_openclip_path_monkeypatched(monkeypatch):
    # Create fake open_clip module
    fake = types.SimpleNamespace()

    class FakeModel:
        def __init__(self):
            pass
        def eval(self):
            pass
        def to(self, device):
            pass
        def encode_image(self, tensor):
            # Return deterministic embeddings: mean over pixels
            # shape: (batch, dim) -> return (batch, 4)
            bs = tensor.shape[0]
            import torch
            out = torch.arange(bs * 4, dtype=torch.float32).view(bs, 4)
            return out

    def fake_create(repo):
        # repo will be the hf-hub id passed through
        model = FakeModel()
        preprocess = lambda img: __import__('torch').from_numpy((np.array(img.resize((32,32))).mean(axis=2) / 255.0).astype('float32')).unsqueeze(0)
        return model, None, preprocess

    fake.create_model_and_transforms = fake_create

    monkeypatch.setitem(sys.modules, 'open_clip', fake)

    # Create simple frames
    frames = [Image.new('RGB', (32, 32), color=(i*20, 10, 10)) for i in range(3)]

    ks = KeyframeSelector(model_name='hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K', use_open_clip=True)
    idxs = ks.select(frames)
    # we expect it to return valid indices within range
    assert all(0 <= i < len(frames) for i in idxs)
    assert len(idxs) >= 1
