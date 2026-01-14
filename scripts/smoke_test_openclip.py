"""Smoke test for OpenCLIP-backed keyframe selection.

This script creates a few simple images and runs `KeyframeSelector` with
`use_open_clip=True` against the HF-hosted OpenCLIP model we downloaded.
"""
from PIL import Image
from sagevision.keyframe_selector import KeyframeSelector


def run():
    frames = [Image.new("RGB", (224, 224), color=(i * 40 % 255, 10, 10)) for i in range(5)]
    ks = KeyframeSelector(model_name="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", use_open_clip=True)
    idxs = ks.select(frames)
    print("Selected indices:", idxs)


if __name__ == "__main__":
    run()
