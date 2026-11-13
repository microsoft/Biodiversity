---
description: "PyTorch-Wildlife release history: final version 1.3.1 and the v1.3.0 announcement."
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

Published on 2026-04-22 as [PyTorch-Wildlife v1.3.0](https://github.com/microsoft/Biodiversity/releases/tag/pw_v1.3.0), this release had three parts plus a preview.

#### Naming changes

| Previous name | New name | Notes |
|---|---|---|
| OWL / Overhead Wildlife Locator | MegaDetector-Overhead | Repo: [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) |
| PW_Bioacoustics / bioacoustics module | MegaDetector-Acoustic | Repo: [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) |
| microsoft/CameraTraps | microsoft/Biodiversity | GitHub repo rename — old URLs redirect automatically |
| Sparrow Studio | SPARROW Studio | Capitalization standardized (SPARROW is an acronym) |

#### SPARROW Studio beta

The release introduced the SPARROW Studio beta, a graphical frontend for data management, inference, analysis, and annotation. It linked a signed [Windows MSI installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1); Mac and Linux builds were in progress at the time.

#### MegaDetector-Acoustic

MegaDetector-Acoustic is available at [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) with CLI scripts for dataset preparation, training, and inference on audio recordings, plus a pre-trained bird classifier (`MD_AudioBirds_V1`). See the [MegaDetector-Acoustic overview](../bioacoustics.md), the [model zoo catalogue](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#acoustics--detector), and the end-to-end demo at [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic).

#### MegaDetector-Overhead

A generalized, point-based detection model for overhead imagery. The release named two variants, OWL-T and OWL-C. The [MegaDetector-Overhead repository](https://github.com/microsoft/MegaDetector-Overhead) also lists OWL-D; see the [overhead detectors](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#overhead--detector) in the model zoo catalogue.

#### Looking ahead — PW-Engine

The release previewed **PW-Engine**, a Rust-based, model-agnostic inference core intended for the Python API and SPARROW Studio, with HTTP, CLI, Python and C interfaces. It has since been released as [Sparrow Engine](https://github.com/microsoft/SPARROW-Engine) (`pip install sparrow-engine`).
