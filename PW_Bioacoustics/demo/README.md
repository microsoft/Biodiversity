# PW_Bioacoustics Demo

End-to-end demo using 5 real bird recordings from the
[PteroSet](https://zenodo.org/records/19137071) dataset
(project PPA4, Putumayo, Colombia — CC BY 4.0).

## Files

```
demo/
├── bioacoustics_demo.ipynb   ← main demo notebook
├── data/
│   ├── audios/               ← 5 PteroSet WAV files (192 kHz, auto-downsampled to 48 kHz)
│   └── labels/               ← Raven Pro selection tables (.Table.1.selections.txt)
└── output/                   ← generated at runtime
    ├── spectrograms/         ← shared .npy mel spectrogram files
    ├── inference/            ← ONNX inference predictions and spectrograms
    ├── binary/               ← binary model, splits, logs, checkpoints
    └── multiclass/           ← multiclass model, splits, logs, checkpoints
```

## Running the notebook

```bash
# From the PW_Bioacoustics/ directory
cd PW_Bioacoustics/demo
jupyter notebook bioacoustics_demo.ipynb
```

The notebook must be run from `PW_Bioacoustics/demo/` — the first cell asserts this.

## What the notebook demonstrates

| Section | Description |
|---------|-------------|
| 0. Setup | Imports, path configuration, `%matplotlib inline` |
| 1. Data Exploration | Annotation counts, species distribution (derived automatically from data) |
| 2. Inference | Download `MD_AudioBirds_V1.onnx` from Zenodo, run ONNX inference on all 5 recordings, visualise predictions vs. ground-truth |
| 3. Train | — |
| 3.0 Build COCO Annotations | `PteroSetReader` converts Raven Pro TSV → COCO-like JSON (binary + multiclass) |
| 3.1 Binary Classification | AVEVOC vs. noise — `build_windows`, spectrograms, train, evaluate |
| 3.2 Multiclass Classification | Top-4 species vs. noise — species analysis bar chart, reuses spectrograms, trains separate model |

Every code cell is preceded by a markdown cell explaining what it does and its expected output.

## Pre-trained model

Section 2 downloads the pre-trained **MD_AudioBirds_V1** ONNX model from Zenodo:

```
https://zenodo.org/records/18177050/files/MD_AudioBirds_V1.onnx?download=1
```

The file is cached to `output/inference/MD_AudioBirds_V1.onnx` and skipped on subsequent runs.

## Using your own data

Swap in your own recordings by replacing the files in `data/audios/` and `data/labels/`
and updating `PPA4_FILES` in the Setup cell. For a different annotation format, subclass `BaseReader`
following the `PteroSetReader` pattern.

## Expected runtime

| Environment | Binary training | Multiclass training |
|-------------|----------------|---------------------|
| GPU (A100)  | ~2 min         | ~2 min              |
| CPU (16-core) | ~20–40 min   | ~20–40 min          |

The ONNX inference section (Section 2) runs in under a minute on CPU.

Reduce `epochs` in the config cells to speed up the demo.

## Data citation

> Ruiz, D., Ulloa, J. S., Miao, Z., Betancourt, N., Hernández, A., Demuro, B., Barona Cortés, E.,
> Toro Gómez, M. P., Mendoza-Henao, A. M., Sierra-Ricaurte, A. F., Pérez-Peña, S. C., Dodhia, R.,
> Arbelaez, P., & Lavista Ferres, J. M. (2026). PteroSet [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.19137071
