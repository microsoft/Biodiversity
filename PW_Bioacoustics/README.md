# PW_Bioacoustics

Companion module for bioacoustics experiments using the PytorchWildlife core library.

## Overview

This module provides CLI scripts for training, inference, and dataset preparation for bioacoustics classification. The core functionality (models, datasets, utilities) is provided by the `PytorchWildlife` library.

## Quick Start

### 1. Installation

```bash
# Install PyTorch-Wildlife with bioacoustics dependencies
pip install -e .
pip install librosa soundfile pyyaml torchmetrics
```

### 2. Configuration

Create a YAML config file for your domain (see `template.yaml` as reference):

```yaml
name: "my_domain"
datasets:
  - "dataset_name_1"

class_names:
  0: "noise"
  1: "target_class"

paths:
  data_root: "${DATA_ROOT}"
  output_root: "${OUTPUT_ROOT}"
  spectrograms_dir: "${OUTPUT_ROOT}/mel_spectrograms"
  annotations_file: "annotations.json"

audio:
  sample_rate: 48000
  window_size_sec: 5.0
  overlap_sec: 4.0

spectrogram:
  n_fft: 2048
  hop_length: 512
  n_mels: 224

training:
  batch_size: 32
  lr: 0.0001
  epochs: 50
  backbone: "resnet18"
```

### 3. Prepare Dataset

```bash
# Full pipeline (stats, windows, spectrograms, splits)
python prepare_dataset.py --config config/my_domain.yaml

# Or run specific steps
python prepare_dataset.py --config config/my_domain.yaml --steps windows spectrograms
```

### 4. Train Model

```bash
# Binary classification
python train.py --config config/my_domain.yaml \
    --train_csv train_split.csv \
    --val_csv val_split.csv \
    --test_csv test_split.csv

# Multiclass classification
python train.py --config config/my_domain.yaml \
    --train_csv train_split.csv \
    --test_csv test_split.csv \
    --num_classes 4
```

### 5. Run Inference

```bash
python inference.py --config config/my_domain.yaml \
    --checkpoint model.ckpt \
    --audios_source /path/to/audio/folder \
    --dataset my_inference
```

## Demo

The recommended way to get started is the **end-to-end demo notebook** at [`demo/bioacoustics_demo.ipynb`](demo/bioacoustics_demo.ipynb). It walks through the full pipeline using real bird recordings from the [PteroSet](https://zenodo.org/records/19137071) dataset:

1. **Data Exploration** — annotation counts, species distribution
2. **Inference** — download `MD_AudioBirds_V1.onnx` from Zenodo, run ONNX inference on all 5 recordings, visualise predictions vs. ground-truth
3. **Train**
   - **3.0 Build COCO Annotations** — `PteroSetReader` converts Raven Pro TSV → COCO-like JSON
   - **3.1 Binary Classification** — AVEVOC vs. noise training with `ResNetClassifier`
   - **3.2 Multiclass Classification** — top-4 species vs. noise, species analysis bar chart, trains separate model

See [`demo/README.md`](demo/README.md) for setup instructions and expected runtimes.

## Module Structure

```
PW_Bioacoustics/
├── __init__.py
├── train.py              # Training CLI script
├── inference.py          # Inference CLI script
├── prepare_dataset.py    # Dataset preparation pipeline
├── template.yaml         # Template configuration file
└── demo/
    ├── bioacoustics_demo.ipynb   # End-to-end demo notebook
    ├── README.md
    ├── data/                     # Sample audio + annotations
    └── config/                   # Demo YAML configs
```

The CLI scripts (`train.py`, `prepare_dataset.py`, `inference.py`) can be used standalone or imported as modules (as the demo notebook does). `template.yaml` documents all configuration parameters.

## Core Library (PytorchWildlife)

This module uses the following components from `PytorchWildlife`:

### Models (`PytorchWildlife.models.bioacoustics`)
- `ResNetClassifier`: PyTorch Lightning module for spectrogram classification (binary and multiclass)
- `BaseBioacousticsClassifier`: Base class for bioacoustics models
- `load_model_from_checkpoint()`: Load a trained model from a `.ckpt` file for inference

### Datasets (`PytorchWildlife.data.bioacoustics.bioacoustics_datasets`)
- `BioacousticsDataset`: Training dataset for loading spectrograms from `.npy` files
- `BioacousticsInferenceDataset`: Inference dataset (no labels required)
- `SpectrogramAugmentations`: SpecAugment-style augmentations (time/frequency masking)
- `MixUpCollator`: Batch-level MixUp augmentation
- `PerSampleNormalize`, `ResizeTo`: Spectrogram transforms

### Annotations (`PytorchWildlife.data.bioacoustics.bioacoustics_annotations`)
- `BaseReader`: Abstract base class for converting annotation formats to COCO-like JSON
- `AnnotationCreator`: Builds COCO-like annotation files from `BaseReader` subclasses

### Configuration (`PytorchWildlife.data.bioacoustics.bioacoustics_configs`)
- `DomainConfig`: Nested dataclass for domain settings (paths, audio, spectrogram, training, splits)
- `load_config()`: YAML configuration loader with environment variable expansion
- `save_config()`: Serialize a `DomainConfig` back to YAML

### Windows (`PytorchWildlife.data.bioacoustics.bioacoustics_windows`)
- `build_windows()`: Generate training windows from annotations (sliding, balanced, or customized strategies)
- `build_inference_windows()`: Generate sliding windows for inference on raw audio files

### Spectrograms (`PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms`)
- `compute_mel_spectrograms_gpu()`: GPU-accelerated mel spectrogram computation, saves `.npy` files

## Projects Using PytorchWildlife Bioacoustics

- **[PteroSet](https://github.com/microsoft/PteroSet)** — A machine learning pipeline for detecting and classifying tropical bird vocalizations from passive acoustic monitoring recordings. Built on the PytorchWildlife bioacoustics core library, it demonstrates the full workflow: COCO annotation creation from Raven Pro labels, mel spectrogram preparation, binary ResNet training, and leave-one-project-out cross-validation.
- **[CookInlet_Belugas](https://github.com/microsoft/CookInlet_Belugas)** — An end-to-end passive acoustic monitoring pipeline for endangered Cook Inlet beluga whales. Implements spectrogram generation, a two-stage deep learning architecture for cetacean signal detection and multi-species classification (beluga, humpback, killer whale), and an active-learning loop for domain adaptation to novel soundscapes.

## Training Arguments

```
--config              # YAML config file (recommended)
--train_csv           # Path to training CSV
--val_csv             # Path to validation CSV (optional)
--test_csv            # Path to test CSV
--num_classes         # 2 for binary, >2 for multiclass (default: 2)
--backbone            # resnet18, resnet34, resnet50 (default: resnet18)
--batch_size          # Batch size (default: 32)
--lr                  # Learning rate (default: 1e-4)
--epochs              # Number of epochs (default: 5)
--use_specaug         # Enable spectrogram augmentations
--normalize           # Normalize spectrograms (default: True)
--freeze_backbone     # none, all, early, layer1, layer2, layer3
```

## Inference Arguments

```
--config              # YAML config file (recommended)
--audios_source       # Audio folder, JSON, or CSV with windows
--checkpoint          # Model checkpoint file (.ckpt)
--num_classes         # Number of classes (default: 2)
--class_names         # Class names for output columns
--window_size_sec     # Window size in seconds (default: 5.0)
--overlap_sec         # Overlap between windows (default: 4.0)
--sample_rate         # Target sample rate (default: 48000)
--batch_size          # Batch size (default: 64)
--temperature         # Temperature scaling (default: 1.0)
--dataset             # Dataset name for output directory
```

## Output Formats

### Binary Classification
Output CSV columns: `audio`, `start(s)`, `end(s)`, `prediction`, `probability`, `confidence`

### Multiclass Classification
Output CSV columns: `audio`, `start(s)`, `end(s)`, `prediction`, `{ClassName}_prob` for each class

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning
- librosa, torchaudio
- pandas, numpy
- Weights & Biases (optional, for experiment tracking)

See the main `requirements.txt` for full dependencies.
