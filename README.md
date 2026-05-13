![A colorful banner illustrating various species of animals and plants in a natural environment, symbolizing biodiversity and the use of AI for conservation purposes.](https://zenodo.org/records/20044680/files/Biodiversity_Banner.png)

<div align="center">
<font size="6"> Open-source AI for camera traps, bioacoustics, and wildlife monitoring </font>
<br>
<hr>
<a href="https://github.com/microsoft/Biodiversity/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/Discord-Join_us-5865F2?logo=discord&logoColor=white" /></a>
<a href="https://microsoft.github.io/Biodiversity/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a>
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<br><br>
</div>

## 📣 Announcements

### What we've been up to

Hey everyone! It's been a while since our last update — we hope you haven't forgotten about us! 😊

Over the past couple of months we've been thinking hard about the future of our biodiversity AI ecosystem and cooking up some exciting new features just for you.

After two years of community use, one thing has become super clear: most people prefer a nice graphical interface over writing code. People have been asking for a more seamless, unified experience that covers data management, processing, AI inference, analysis, and annotation all in one place.

So we built **SPARROW Studio** — a clean, unified UI built on top of PyTorch-Wildlife that brings exactly those tools together and much more:

- Local and cloud-based data storage & management
- AI inference using the PyTorch-Wildlife model zoo
- Post-inference statistics and analysis
- Pre- and post-inference data annotation (easy bounding-box and category editing)
- Embedding visualization and feature retrieval tools

![image](https://zenodo.org/records/18870374/files/sparrow_studio.png)

We're kicking things off with a beta test before the official release. The Windows MSI installer is available directly on Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress — reach out if you'd like to be on that list.

We've also expanded PyTorch-Wildlife itself into bioacoustics and overhead animal localization — both are out in this release:

- A dedicated [MegaDetector-Acoustics](https://github.com/microsoft/MegaDetector-Acoustics) repository with several newly trained bioacoustic models
- [MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) — our generalized, point-based detection model for overhead imagery.

SPARROW Studio already has dedicated support for both, so beta testers can run inference and annotate acoustic recordings or overhead images directly in the UI.

### The future of PyTorch-Wildlife

With SPARROW Studio stepping into the picture, PyTorch-Wildlife itself will gradually evolve into a clean, stable API + high‑quality model zoo layered on top of a general model inference engine — called PW-Engine, while SPARROW Studio becomes the intuitive, everything‑in‑one-place frontend.

**PW-Engine** (PyTorch-Wildlife Engine) is an inference core written in Rust. It is model-agnostic and targets the full PyTorch-Wildlife model zoo and future third party models (e.g. BioClip and Perch) through four consumption surfaces: an HTTP REST API, a single-binary CLI, Python bindings, and a native C library for desktop integration. All four surfaces are feature-complete today; a data-management layer and MLOps functionality are the next milestones. PW-Engine also powers SPARROW Studio under the hood, and the same surfaces are open to anyone building their own frontend. A short overview — what it is, how it fits alongside the current Python API and SPARROW Studio, and how to pilot it — is here: [PW-Engine Overview](https://microsoft.github.io/Biodiversity/pw_engine_overview/).

If you're interested in API or backend work, or you run an inference-heavy pipeline and want to pilot PW-Engine early, we'd love your help shaping the next chapter of PyTorch-Wildlife.

And one dream we've had for a long time: letting non‑coders fine‑tune their own models on their own data. Thanks to recent advances, we're finally close — and this will be a major focus for both PyTorch-Wildlife and SPARROW Studio next.

### Why "SPARROW Studio"?

Some of the UI features we needed for PyTorch-Wildlife also fit naturally as a frontend for [SPARROW](https://github.com/microsoft/SPARROW), our edge device for remote data collection and biodiversity monitoring. Since the name "Sparrow" already carried a warm, lively spirit — and the overlap between the projects made things simpler — we decided to call the UI SPARROW Studio. The name reflects shared roots and a bit of personality we liked.

We'd genuinely love to have you in the SPARROW Studio beta. Drop us a message anytime — the more feedback the better! 🐦

#### Previous versions:
- [Release notes](https://microsoft.github.io/Biodiversity/releases/release_notes/)

## Projects

| Repo | What it is |
|---|---|
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | AI model for detecting animals, people, and vehicles in camera-trap imagery — where it all started |
| [microsoft/MegaDetector-Acoustics](https://github.com/microsoft/MegaDetector-Acoustics) | Bioacoustic AI for biodiversity monitoring — audio classification and species identification from sound |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | Overhead imagery detection — point-based wildlife localization from aerial views |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | The collaborative deep learning framework and model zoo for conservation AI |
| [microsoft/WildlifeClassification](https://github.com/microsoft/WildlifeClassification) | Species classification models, running downstream of MegaDetector detection |
| [microsoft/Bongo](https://github.com/microsoft/Bongo) | *(Owner-populated)* |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI edge device for remote field deployments |
| [/SPARROW-Studio](./SPARROW-Studio) | The all-in-one desktop UI for running every model in this ecosystem |

## Cite us

When citing work that uses any of the repositories under this umbrella, please cite:

- **Hernandez et al. 2024** — *Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation* — for any use of the PyTorch-Wildlife framework or models accessed through it
- **Beery, Morris, Yang 2019** — *Efficient Pipeline for Camera Trap Image Review* — for any use of MegaDetector specifically

A `citation.cff` file is included in this repository for automated citation tools.

## Contributing

We welcome community contributions. See our [Contribution Guidelines](https://microsoft.github.io/Biodiversity/contribute/#how-to-participate) for how to participate.

## Community

Have questions or want to connect with the team? Join us on Discord: [![Discord](https://img.shields.io/badge/Discord-Join_us-5865F2?logo=discord&logoColor=white)](https://discord.gg/TeEVxzaYtm)

A list of organizations using MegaDetector across global conservation work — six years of partnerships, from national parks to research universities to NGOs — is maintained on the [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) repository.

>[!IMPORTANT]
>If you would like to be added to this list or have any questions regarding MegaDetector and Pytorch-Wildlife, please [email us](zhongqimiao@microsoft.com) or join us in our Discord channel: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

## About

Maintained by [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/).