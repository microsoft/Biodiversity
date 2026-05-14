---
description: "MegaDetector-Acoustic: open-source bioacoustic AI for wildlife monitoring. Audio classification, bird detection, and species identification from passive acoustic recordings."
tags:
  - MegaDetector-Acoustic
  - bioacoustics
  - wildlife audio classification
  - bird detection
  - passive acoustic monitoring
---

# MegaDetector-Acoustic

**MegaDetector-Acoustic** provides training, inference, and dataset preparation for audio classification in wildlife monitoring. The module is maintained at [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) and builds on core APIs in `PytorchWildlife.data.bioacoustics` and `PytorchWildlife.models.bioacoustics`.

## What's included

- **CLI scripts** for dataset preparation (`prepare_dataset.py`), training (`train.py`), and inference (`inference.py`)
- **`ResNetClassifier`** — PyTorch Lightning module for spectrogram classification (binary and multiclass)
- **Mel-spectrogram preprocessing** with optional GPU acceleration
- **Annotation readers** (COCO-like JSON), including support for the PteroSet / Raven Pro format
- **`MD_AudioBirds_V1`** — a pre-trained bird classifier distributed as ONNX for direct inference

See the [MegaDetector-Acoustic model zoo](model_zoo/bioacoustics.md) for the released models.

## Demo

The end-to-end notebook at [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) walks through:

1. **Data exploration** — annotation counts, species distribution
2. **Inference** — run `MD_AudioBirds_V1` on real recordings, visualise predictions vs. ground truth
3. **Training** — build COCO-style annotations, binary classification (target vs. noise), multiclass classification

It uses recordings from the [PteroSet](https://zenodo.org/records/19137071) dataset.

## Projects using this module

- **[PteroSet](https://github.com/microsoft/PteroSet)** — Machine-learning pipeline for detecting and classifying tropical bird vocalisations from passive acoustic monitoring, with leave-one-project-out cross-validation.
- **[CookInlet_Belugas](https://github.com/microsoft/CookInlet_Belugas)** — Passive acoustic monitoring for endangered Cook Inlet beluga whales. A two-stage pipeline covering cetacean signal detection and multi-species classification (beluga, humpback, killer whale), plus an active-learning loop for domain adaptation.

## Install

```bash
pip install PytorchWildlife
pip install librosa soundfile pyyaml torchmetrics
```

See the [MegaDetector-Acoustic README](https://github.com/microsoft/MegaDetector-Acoustic) for full configuration options, training arguments, and output formats.
