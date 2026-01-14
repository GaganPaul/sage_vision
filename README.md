# Sage Vision
SageVision is a local-first, vision-centric video summarisation framework that generates meaningful summaries without relying on audio transcripts.

Unlike traditional video summarizers that depend on speech-to-text pipelines, SageVision treats visual understanding as the primary signal, making it suitable for silent videos, privacy-sensitive environments, and offline use.

**Key Features:**

```python
🧠 Vision-first summarization:  no transcript required
💻 Runs fully locally (CPU or GPU)
🔒 Privacy-preserving:  no cloud or external APIs needed
🎞️ Scene-aware & keyframe:  based processing
📉 Minimal LLM usage through hierarchical summarization
🧩 Modular & extensible open:   source architecture
```

## **🔍 Why SageVision?**
```python
Most existing video summarization tools follow this pipeline:

Video → Audio → Transcript → LLM → Summary
```

**SageVision instead follows:**

**Video → Visual Understanding → Semantic Compression → Summary**

**This makes SageVision especially useful for:**

```python
Silent or music-only videos
Educational videos with slides
Surveillance and CCTV footage
Accessibility use cases
Low-bandwidth or offline environments
```

## **🏗️ System Overview:**

**High-level Pipeline:**

```python
Video
  ↓
Scene Detection
  ↓
Keyframe Extraction
  ↓
Vision Captioning
  ↓
Scene Level Summaries
  ↓
Final Video Summary
```


Core Design Principles
Compress before reasoning
Scenes over frames
LLMs as aggregators, not perception engines
Local-first by default


## **🧩 Architecture:**

```python
sagevision/
├── video_parser/        ## Video decoding (FFmpeg / OpenCV)
├── scene_detector/      ## Shot & scene boundary detection
├── keyframe_selector/   ## Adaptive keyframe sampling
├── vision_captioner/    ## Image-to-text (Florence-2, BLIP, etc.)
├── summarizer/          ## Lightweight text summarization
├── pipeline/            ## End-to-end orchestration
├── cli/                 ## Command-line interface
└── utils/               ## Shared utilities
```

Each module is replaceable and configurable, enabling experimentation with different models and strategies.

## **🖥️ Local Execution Modes:**

Mode:	Description

CPU-only:	Fully offline, slower but accessible

GPU-accelerated:	Faster vision captioning & summarization

Research mode:	Plug in custom models & heuristics

SageVision is designed to scale down gracefully to low-resource machines.

---

## **🔁 Default Hugging Face Models (configurable)**

- **Vision captioning:** `Salesforce/blip-image-captioning-base` (default, CPU-friendly)
- **CLIP embeddings (keyframe selection):** `laion/CLIP-ViT-B-32`
- **Summarization:** `facebook/bart-large-cnn`

These defaults are chosen to balance quality and runtime on modern laptops (e.g. Apple M-series). Models are configurable in code and can be overridden for research or production.

### Device selection

The components auto-detect available compute in the following order: CUDA > MPS (Apple Silicon) > CPU. If MPS or CUDA is available, models will attempt to use them; otherwise the code runs on CPU.

### Accessing gated models on Hugging Face

Some model repositories on Hugging Face are gated or require authentication. If a configured model (e.g., `laion/CLIP-ViT-B-32`) is private, you can authenticate in one of two ways:

- Log in locally with the CLI: `huggingface-cli login` and provide your token when prompted.
- Set the token in your environment: `export HUGGINGFACE_HUB_TOKEN="hf_xxx"` (or add it to your shell profile).

The `KeyframeSelector` accepts an optional `token` parameter or reads `HUGGINGFACE_HUB_TOKEN` from the environment. If a model fails to load, SageVision will automatically fall back to a public CLIP model (`openai/clip-vit-large-patch14`).

### Optional: use OpenCLIP

You can use the `open_clip` library and HF-hosted OpenCLIP models directly (for example `hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) by enabling `use_open_clip=True` when creating `KeyframeSelector`.

Example:
```python
from sagevision.keyframe_selector import KeyframeSelector
selector = KeyframeSelector(model_name='hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K', use_open_clip=True)
```
If `open_clip` is not installed or the requested model is gated and you are not authenticated, the selector will fall back to the `transformers`-based CLIP path and then to a public fallback model.

### Pre-download OpenCLIP (safe)

If you want to pre-download a gated HF-hosted OpenCLIP model to your machine (so the selector can use it offline), follow these steps locally:

1. Install OpenCLIP: `pip install open_clip_torch`
2. Authenticate to Hugging Face (recommended):
   - `huggingface-cli login` or
   - `export HUGGINGFACE_HUB_TOKEN="hf_xxx"` (add to your shell profile)
3. Run the download helper (from the repo root):

```bash
python scripts/download_openclip.py --model hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

This will download the model weights and transforms into your HF cache. Do NOT paste or share your token in public chats; if you already did, rotate and revoke it on Hugging Face for safety.

### Quick code examples

```python
from sagevision.vision_captioner import VisionCaptioner
from sagevision.keyframe_selector import KeyframeSelector
from sagevision.summarizer import Summarizer

cap = VisionCaptioner()  # uses Salesforce/blip-image-captioning-base
cap.caption(pil_image)

selector = KeyframeSelector()
idxs = selector.select(list_of_frames)

summ = Summarizer()
summary = summ.summarize(["A man walks into a room.", "He sits down."])
```

### Dependencies

Add the following to your environment for best results:

```
transformers>=4.30
torch
accelerate  # optional, for large models and device mapping
```

For Apple Silicon / MPS users, install a PyTorch wheel with MPS support and ensure `torch.backends.mps.is_available()` returns True.

## **🚀 Getting Started (Planned):**

**Recommended (Conda, macOS-friendly)**

```bash
# Create an environment (example name: 'sv')
conda create -n sv python=3.10 -y
conda activate sv
# Install Tcl/Tk from conda-forge so Tkinter works reliably on macOS
conda install -c conda-forge tk -y
# Install runtime dependencies
pip install -r requirements.txt
```

If you prefer not to activate the environment, use `conda run` to execute commands in the `sv` environment. Examples (exact commands):

```bash
# Launch GUI via the CLI (example you provided):
conda run -n sv /bin/bash -lc "PYTHONPATH=. python -m sagevision.cli --gui"

# Launch GUI directly (no CLI wrapper):
conda run -n sv /bin/bash -lc "PYTHONPATH=. python -m sagevision.gui"

# Run the CLI on a video file from the environment:
conda run -n sv /bin/bash -lc "PYTHONPATH=. python -m sagevision.cli --input path/to/video.mp4"
```

**Alternative (pip / system Python)**

```bash
git clone https://github.com/GaganPaul/sage_vision
cd sagevision
pip install -r requirements.txt
```

### Quick start

- Run the CLI on a video file:

```bash
python -m sagevision.cli --input path/to/video.mp4
```

- Launch the simple Tkinter GUI via the CLI:

```bash
python -m sagevision.cli --gui
```

- Launch the GUI directly (without the CLI):

```bash
python -m sagevision.gui
```

or from any Python REPL / script:

```bash
python -c "from sagevision.gui import launch; launch()"
```

Note: If Tkinter is not available in your Python build on macOS, install a tcl/tk-enabled Python or use Homebrew to install the necessary libraries. The CLI performs a safe Tk init check (subprocess) before launching on macOS and will print diagnostic output if initialization fails.

### Running GUI vs CLI

- Use `python -m sagevision.cli --input path/to/video.mp4` when you want to run the pipeline non-interactively and just receive a textual summary.
- Use `python -m sagevision.cli --gui` when you prefer the simple desktop UI and do not want to run the pipeline from a script.
- Use `python -m sagevision.gui` if you want to launch the GUI directly (for example when embedding or debugging the UI) without invoking the CLI's subprocess Tk init check.

### GUI improvements & notes

- The GUI now reuses a single `Pipeline` instance to reduce per-run overhead and offers a small set of usability improvements: **Stop** button (best-effort if `Pipeline.stop()` is present), **Clear Output**, input validation (checks file exists), and more robust progress handling (accepts both 0..1 and 0..100 percent ranges).
- If you press **Stop** and the pipeline implements a `stop()` method it will be called; otherwise a stop request will be recorded and noted in the UI (best-effort cancellation).
- If you see issues launching the GUI on macOS, make sure your Python is linked against a Tcl/Tk build (or use a Python distribution that bundles it, e.g., official installers or Homebrew builds).

### GUI progress

The GUI displays a visual progress bar (ttk.Progressbar) in the main toolbar and shows per-stage messages in the output window via the pipeline's `progress_callback` API. The progress bar updates are marshaled safely to the GUI thread.

### OpenCV video parsing

SageVision now includes an OpenCV-based `VideoParser` (uses `opencv-python` / `opencv-python-headless`). The parser yields `PIL.Image` frames which are consumed by the rest of the pipeline. On macOS, install a PyTorch wheel with MPS support and ensure OpenCV is available in your environment.
## **🎯 Project Goals:**


Enable transcript-less video summarization

Reduce dependency on large multimodal LLMs

Support offline & edge deployments

Provide a clean, research-friendly codebase

Serve as a foundation for further work in visual understanding


## **🚫 Non-Goals:**


Real-time live video summarization

Emotion or intent-level reasoning

Replacing transcript-based summarizers

Cloud-first or API-dependent workflows


## **📚 Use Cases:**

Education & self-learning

Accessibility tools

Video archiving & indexing

Research & benchmarking

NGOs and low-connectivity regions

Privacy-sensitive video analysis

## **🧠 Research Alignment:**

**SageVision can be positioned as:**

A local-first, vision-centric video summarization system that minimizes LLM usage through adaptive scene-based compression.

## **The project is suitable for:**

Applied research

System papers

Open-source contributions

Academic demos and benchmarks

## **🤝 Contributing:**

Contributions are welcome!

## **You can help by:**

Improving keyframe selection strategies

Adding new vision captioning models

Optimizing performance for CPU-only setups

Improving documentation and examples

Contribution guidelines will be added soon.

## **📄 License:**

This project will be released under a permissive open-source license (TBD).

## **🌱 Project Status:**

**🟡 Active development**

Core architecture and pipeline design are complete.
Implementation is ongoing.

