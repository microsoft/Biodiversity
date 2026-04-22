# What's new in PyTorch-Wildlife v1.3.0

## What we've been up to

Hey everyone! It's been a while since our last update — we hope you haven't forgotten about us! 😊

Over the past couple of months we've been thinking hard about the future of PyTorchWildlife and cooking up some exciting new features just for you.

After two years of community use, one thing has become super clear: most people prefer a nice graphical interface over writing code. People have been asking for a more seamless, unified experience that covers data management, processing, AI inference, analysis, and annotation all in one place.

So we built Sparrow Studio — a clean, unified UI built on top of PyTorchWildlife that brings exactly those tools together:

- Local and cloud-based data storage & management
- AI inference using the PyTorchWildlife model zoo
- Post-inference statistics and analysis
- Pre- and post-inference data annotation (easy bounding-box and category editing)
- Embedding visualization and feature retrieval tools

We're kicking things off with a beta test before the official release. The Windows MSI installer is available directly on Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress — reach out if you'd like to be on that list.

We've also expanded PyTorchWildlife itself into bioacoustics and overhead animal localization — both are out in this release:

- A dedicated [bioacoustics module](../bioacoustics.md) with several newly trained bioacoustics models
- [OWL](https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_owl.ipynb) (Overhead Wildlife Locator) — our new generalized, point-based detection model for overhead imagery. (publication on the way.)

Sparrow Studio already has dedicated support for both, so beta testers can run inference and annotate bioacoustics recordings or overhead images directly in the UI.

## The future of PyTorchWildlife

With Sparrow Studio stepping into the picture, PyTorchWildlife itself will gradually evolve into a clean, stable API + high‑quality model zoo layered on top of a general model inference engine — called PW-Engine, while Sparrow Studio becomes the intuitive, everything‑in‑one-place frontend.

**PW-Engine** (PyTorch-Wildlife Engine) is an inference core written in Rust. It is model-agnostic and targets the full PyTorch-Wildlife model zoo and future third party models (e.g. BioClip and Perch) through four consumption surfaces: an HTTP REST API, a single-binary CLI, Python bindings, and a native C library for desktop integration. All four surfaces are feature-complete today; a data-management layer and MLOps functionality are the next milestones. PW-Engine also powers Sparrow Studio under the hood, and the same surfaces are open to anyone building their own frontend. A short overview — what it is, how it fits alongside the current Python API and Sparrow Studio, and how to pilot it — is here: [PW-Engine Overview](../pw_engine_overview.md).

If you're interested in API or backend work, or you run an inference-heavy pipeline and want to pilot PW-Engine early, we'd love your help shaping the next chapter of PyTorchWildlife. We'll update our public task board later.

And one dream we've had for a long time: letting non‑coders fine‑tune their own models on their own data. Thanks to recent advances, we're finally close — and this will be a major focus for both PyTorchWildlife and Sparrow Studio next.

## Why "Sparrow Studio"?

Some of the UI features we needed for PyTorchWildlife also fit naturally as a frontend for Project Sparrow, another effort in our group focused on remote data-collection hardwares and edge computing. Since the name "Sparrow" already carried a warm, lively spirit — and the overlap between the projects made things simpler — we decided to call the UI Sparrow Studio. The name just reflects some shared roots and a bit of personality we liked.

Stay tuned! These updates are dropping very soon, and we'd genuinely love to have you in the Sparrow Studio beta. Drop us a message anytime — the more feedback the better! 🐦

![Sparrow Studio](https://zenodo.org/records/18870374/files/sparrow_studio.png)
