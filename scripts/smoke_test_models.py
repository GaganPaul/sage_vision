"""Smoke test: load BLIP, CLIP, and BART and run simple inferences.

Creates small PIL images and runs captioning, embedding-based selection, and summarization.
"""
import traceback
from PIL import Image

from sagevision.vision_captioner import VisionCaptioner
from sagevision.keyframe_selector import KeyframeSelector
from sagevision.summarizer import Summarizer


def run():
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    images = [image, Image.new("RGB", (224, 224), color=(255, 0, 0)), Image.new("RGB", (224, 224), color=(0, 255, 0))]

    print("=== Device availability ===")
    import torch

    print("torch.cuda.is_available():", torch.cuda.is_available())
    mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    print("torch.backends.mps.is_available():", bool(mps_available))

    # Vision captioner (BLIP)
    try:
        print("\n=== Running BLIP captioner ===")
        vc = VisionCaptioner()  # default model
        cap = vc.caption(image)
        print("Caption:", cap)
    except Exception:
        print("BLIP captioner failed:")
        traceback.print_exc()

    # CLIP embeddings + keyframe selection
    try:
        print("\n=== Running CLIP keyframe selector ===")
        ks = KeyframeSelector()
        idxs = ks.select(images)
        print("Selected indices:", idxs)
    except Exception:
        print("CLIP keyframe selection failed:")
        traceback.print_exc()

    # Summarizer (BART)
    try:
        print("\n=== Running BART summarizer ===")
        summ = Summarizer()
        out = summ.summarize(["A person walks into the room.", "They sit down and read a book.", "The scene is quiet."])
        print("Summary:", out)
    except Exception:
        print("Summarizer failed:")
        traceback.print_exc()


if __name__ == "__main__":
    run()
