![Biodiversity banner featuring colorful wildlife illustrations including birds, mammals, and insects arranged in a natural landscape with green foliage and earth tones, representing the collaborative deep learning framework for conservation](https://zenodo.org/records/20044680/files/Biodiversity_Banner.png)

# Microsoft Biodiversity

**Open-source AI for biodiversity monitoring and conservation.**  
Microsoft AI for Good Lab — camera-trap detection, bioacoustic analysis, species classification, field deployment.
<div align="center"> 

<hr>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/pyversions/PytorchWildlife" /></a> 
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<!-- <a href="https://colab.research.google.com/drive/16-OjFVQ6nopuP-gfqofYBBY00oIgbcr1?usp=sharing"><img src="https://img.shields.io/badge/Colab-Video detection-blue?logo=GoogleColab" /></a> -->
<a href="https://github.com/microsoft/CameraTraps/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/PytorchWildlife" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord" /></a>
<a href="https://microsoft.github.io/Biodiversity/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<br><br>
</div>


---

## Why this repo exists

Our journey started with **MegaDetector** — a camera-trap animal detection model that became a widely adopted tool in the conservation community. Building on that foundation, we created **PyTorch-Wildlife** as a unified platform to host all of our AI for biodiversity work, bringing together detection, classification, and eventually much more.

Over time, our scope grew well beyond camera-trap imagery. We now have active work in bioacoustics, overhead animal detection, edge computing for remote field deployments, and a dedicated desktop UI — SPARROW Studio — that runs every model we produce. As the ecosystem expanded, it became clear that keeping everything inside a single repository was working against us. Code was harder to find, harder to maintain, and harder to extend.

So we made a deliberate decision: break the work into focused, dedicated repositories — one per project — where the code in each repo is concentrated, the ownership is clear, and future contributors know exactly where to go. This repository is the hub that ties them together.

If you're looking for PyTorch Wildlife, model zoo, or installation instructions, head to [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife). If you're here for MegaDetector specifically, [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) is its new canonical home. Everything else is linked in the table below.

---

## Projects

| Repo | What it is |
|---|---|
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | AI model for detecting animals, people, and vehicles in camera-trap imagery — where it all started |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | The collaborative deep learning framework and model zoo for conservation AI |
| [microsoft/Bioacoustics](https://github.com/microsoft/Bioacoustics) | Audio classification and species identification from sound recordings |
| [microsoft/WildlifeClassification](https://github.com/microsoft/WildlifeClassification) | Species classification models, running downstream of MegaDetector detection |
| [microsoft/Bongo](https://github.com/microsoft/Bongo) | *(Owner-populated)* |
| [microsoft/OwlBioacoustics](https://github.com/microsoft/OwlBioacoustics) | *(Owner-populated)* |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI edge device for remote field deployments |
| [/SPARROW-Studio](./SPARROW-Studio) | The all-in-one desktop UI for running every model in this ecosystem |

---

## Cite us

When citing work that uses any of the repositories in this ecosystem:

- **Hernandez et al. 2024** — *Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation* — for any use of the PyTorch-Wildlife framework or the models accessed through it  
  [![arXiv](https://img.shields.io/badge/arXiv-2405.12930-b31b1b.svg)](https://arxiv.org/abs/2405.12930)

- **Beery, Morris, Yang 2019** — *Efficient Pipeline for Camera Trap Image Review* — for any use of MegaDetector specifically  
  [![arXiv](https://img.shields.io/badge/arXiv-1907.06772-b31b1b.svg)](https://arxiv.org/abs/1907.06772)

A `citation.cff` file is included in this repository for automated citation tools.

---

## Contributing

We welcome community contributions across the ecosystem. Each child repository has its own contribution guidelines — start there for code contributions. For broader questions or ideas that span multiple repos, open a discussion here.

👉 [Contribution Guidelines](https://microsoft.github.io/CameraTraps/contribute/#how-to-participate)

---

## Community

Questions, feedback, or just want to connect with the team? Join us on Discord.

[![Discord](https://img.shields.io/badge/Discord-Join_us-5865F2?logo=discord&logoColor=white)](https://discord.gg/TeEVxzaYtm)

A list of organizations using MegaDetector across global conservation work — six years of partnerships spanning national parks, research universities, NGOs, and government agencies — is maintained at [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector).

---

## About

Maintained by [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/).
