---
description: "PyTorch-Wildlife v1.3.0 release notes: SPARROW Studio beta, MegaDetector-Acoustics, MegaDetector-Overhead, and PW-Engine preview."
tags:
  - PyTorch-Wildlife
  - release notes
  - SPARROW Studio
  - MegaDetector-Acoustics
  - MegaDetector-Overhead
  - PW-Engine
---

# Main changes and additions

### V 1.3.0

This release has three parts plus a preview of what's next.

#### SPARROW Studio beta

SPARROW Studio is our new graphical frontend — data management, inference, analysis, and annotation in one UI. The Windows MSI installer is available from Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress.

#### MegaDetector-Acoustics

MegaDetector-Acoustics is available at [microsoft/MegaDetector-Acoustics](https://github.com/microsoft/MegaDetector-Acoustics) with CLI scripts for dataset preparation, training, and inference on audio recordings, plus a pre-trained bird classifier (`MD_AudioBirds_V1`). See the [MegaDetector-Acoustics overview](../bioacoustics.md), the [model-zoo entry](../model_zoo/bioacoustics.md), and the end-to-end demo at [microsoft/MegaDetector-Acoustics](https://github.com/microsoft/MegaDetector-Acoustics).

#### MegaDetector-Overhead

A new generalized, point-based detection model for overhead imagery. Two variants are released — MegaDetector-Overhead-T and MegaDetector-Overhead-C — listed in the [Other Detectors](../model_zoo/other_detectors.md) model zoo. Demo: [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead).

#### Looking ahead — PW-Engine

The future of PyTorchWildlife is [**PW-Engine**](../pw_engine_overview.md), a Rust-based, model-agnostic inference core that powers SPARROW Studio and can also be consumed directly via HTTP, CLI, Python bindings, or a native C library. See the PW-Engine overview for what it is, how it fits alongside the current Python API, and how to pilot it.
