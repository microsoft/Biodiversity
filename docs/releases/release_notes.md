---
description: "PyTorch-Wildlife v1.3.0 release notes: SPARROW Studio beta, MegaDetector-Acoustic, MegaDetector-Overhead, and PW-Engine preview."
tags:
  - PyTorch-Wildlife
  - release notes
  - SPARROW Studio
  - MegaDetector-Acoustic
  - MegaDetector-Overhead
  - PW-Engine
---

<a id="main-changes-and-additions"></a>

# PyTorch-Wildlife release notes

!!! note "PyTorch-Wildlife has ended"
    This page and [Past Releases](past_releases.md) are the release history of PyTorch-Wildlife, which ended with the `PytorchWildlife` 1.3.1 package and is now [SPARROW Playground](https://github.com/microsoft/SPARROW-Playground). 1.3.1 was the [final release](https://github.com/microsoft/SPARROW-Playground/releases/tag/v1.3.1): it adds a `FutureWarning` on import and no further releases or fixes are planned. Each SPARROW project now publishes its own releases; the [home page](../index.md#recent-releases) lists the latest ones.

### V 1.3.0

This release has three parts plus a preview of what's next.

#### Naming changes

| Previous name | New name | Notes |
|---|---|---|
| OWL / Overhead Wildlife Locator | MegaDetector-Overhead | Repo: [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) |
| PW_Bioacoustics / bioacoustics module | MegaDetector-Acoustic | Repo: [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) |
| microsoft/CameraTraps | microsoft/Biodiversity | GitHub repo rename — old URLs redirect automatically |
| Sparrow Studio | SPARROW Studio | Capitalization standardized (SPARROW is an acronym) |

#### SPARROW Studio beta

SPARROW Studio is our new graphical frontend — data management, inference, analysis, and annotation in one UI. The Windows MSI installer is available from Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress.

#### MegaDetector-Acoustic

MegaDetector-Acoustic is available at [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) with CLI scripts for dataset preparation, training, and inference on audio recordings, plus a pre-trained bird classifier (`MD_AudioBirds_V1`). See the [MegaDetector-Acoustic overview](../bioacoustics.md), the [model zoo catalogue](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#acoustics--detector), and the end-to-end demo at [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic).

#### MegaDetector-Overhead

A new generalized, point-based detection model for overhead imagery. Two variants are released — MegaDetector-Overhead-T and MegaDetector-Overhead-C — listed in the [overhead detectors](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#overhead--detector) section of the model zoo catalogue. Demo: [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead).

#### Looking ahead — PW-Engine

The future of PyTorchWildlife is **PW-Engine**, a Rust-based, model-agnostic inference core that powers SPARROW Studio and can also be consumed directly via HTTP, CLI, Python bindings, or a native C library. PW-Engine has since been released as [Sparrow Engine](https://github.com/microsoft/SPARROW-Engine) (`pip install sparrow-engine`).
