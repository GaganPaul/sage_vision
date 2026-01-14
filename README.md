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
Scene level Summaries
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

## **🚀 Getting Started (Planned):**

```python
git clone https://github.com/GaganPaul/sage_vision
cd sagevision
pip install -r requirements.txt

```
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

