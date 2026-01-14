"""Download an OpenCLIP model from the Hugging Face Hub.

This script reads `HUGGINGFACE_HUB_TOKEN` from the environment (recommended) or
relies on `huggingface-cli login` state. It attempts to import `open_clip` and
calls `open_clip.create_model_and_transforms(repo)` which will download the
model weights and preprocessing transforms into the HF cache.

Usage:

    export HUGGINGFACE_HUB_TOKEN="hf_xxx"
    python scripts/download_openclip.py --model hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K

Note: Do NOT paste your token into chat. If you already pasted a token, rotate
it on Hugging Face and revoke the exposed token for safety.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download OpenCLIP model from HF hub")
    parser.add_argument("--model", default="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                        help="OpenCLIP model id (use hf-hub:... for HF-hosted models)")
    parser.add_argument("--force", action="store_true", help="Redownload even if cached")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("Warning: HUGGINGFACE_HUB_TOKEN not found in environment.")
        print("If the model is gated, please set HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`.")

    try:
        import open_clip
    except Exception as e:
        print("Required package `open_clip_torch` is not installed.")
        print("Install with: pip install open_clip_torch")
        sys.exit(1)

    # Attempt to create model (this will download weights/transforms)
    repo = args.model
    try:
        print(f"Downloading OpenCLIP model: {repo}")
        # open_clip uses huggingface-hub internally; auth is picked up from env
        model, _, preprocess = open_clip.create_model_and_transforms(repo)
        print("Model downloaded and ready.")
    except Exception as e:
        print(f"Failed to download OpenCLIP model '{repo}': {e}")
        print("If this is a gated model, ensure your HUGGINGFACE_HUB_TOKEN is valid and try again.")
        sys.exit(2)


if __name__ == "__main__":
    main()
