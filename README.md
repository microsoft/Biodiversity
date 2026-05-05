![image](https://zenodo.org/records/15376499/files/Pytorch_Banner_transparentbk.png)

# Biodiversity

**Microsoft AI for Good Lab — open-source AI for biodiversity monitoring and conservation.**

[![PyPI version](https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen)](https://pypi.org/project/PytorchWildlife)
[![Downloads](https://static.pepy.tech/badge/pytorchwildlife)](https://pypi.org/project/PytorchWildlife)
[![Python versions](https://img.shields.io/pypi/pyversions/PytorchWildlife)](https://pypi.org/project/PytorchWildlife)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife)
[![Colab Demo](https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab)](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing)
[![License](https://img.shields.io/pypi/l/PytorchWildlife)](https://github.com/microsoft/Biodiversity/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord)](https://discord.gg/TeEVxzaYtm)

This umbrella repository is the documentation hub for the AI for Good Lab's biodiversity work. Open-source AI models, edge devices, and tools for camera-trap detection, bioacoustic analysis, species classification, and field deployment in conservation contexts. The repository links to dedicated child repos for each focus area. SPARROW Studio — the desktop companion app for the SPARROW edge device — is hosted at [/SPARROW-Studio](./SPARROW-Studio).

## What's in this ecosystem

| Repo | Purpose |
|---|---|
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | AI model for detecting animals, people, and vehicles in camera-trap imagery |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI edge device |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | Collaborative deep learning framework for conservation |
| [microsoft/Bioacoustics](https://github.com/microsoft/Bioacoustics) | Bioacoustic AI for biodiversity monitoring |
| [microsoft/WildlifeClassification](https://github.com/microsoft/WildlifeClassification) | Species classification models |
| [microsoft/Bongo](https://github.com/microsoft/Bongo) | (Owner-populated) |
| [microsoft/OwlBioacoustics](https://github.com/microsoft/OwlBioacoustics) | (Owner-populated) |
| [/SPARROW-Studio](./SPARROW-Studio) | SPARROW Studio — desktop installer, packaging, signed releases |

## What's new

We've recently restructured this repository. The PyTorch-Wildlife framework now lives at [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife). MegaDetector has its own dedicated home at [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector). Bioacoustic models, species classification, and the OWL overhead detection model have all been promoted to their own repositories. SPARROW Studio — the unified UI for biodiversity AI workflows — is hosted as a subfolder of this umbrella.

If you're looking for the framework code, model zoo, or installation instructions, head to [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife). If you're looking for MegaDetector specifically, [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) is the new canonical location. The old `microsoft/CameraTraps` URL automatically redirects here.

## Cite us

When citing work that uses any of the repositories under this umbrella, please cite:

- **Hernandez et al. 2024** — *Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation* — for any use of the PyTorch-Wildlife framework or models accessed through it
- **Beery, Morris, Yang 2019** — *Efficient Pipeline for Camera Trap Image Review* — for any use of MegaDetector specifically

A `citation.cff` file is included in this repository for automated citation tools.

## Contributing

We welcome community contributions. See our [Contribution Guidelines](https://microsoft.github.io/CameraTraps/contribute/#how-to-participate) for how to participate.

## Community

For questions about MegaDetector and PyTorch-Wildlife, reach the team via Discord: [![Discord](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

A list of organizations using MegaDetector across global conservation work — six years of partnerships, from national parks to research universities to NGOs — is maintained on the [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) repository.

## About

Maintained by [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/).
