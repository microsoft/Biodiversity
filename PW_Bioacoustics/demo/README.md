# PW_Bioacoustics Demo

End-to-end demo using 5 real bird recordings from the
[Humboldt Aves dataset](https://zenodo.org/records/18563039)
(project PPA4, Putumayo, Colombia — CC BY 4.0).

## Files

```
demo/
├── bioacoustics_demo.ipynb   ← main demo notebook
├── data/
│   ├── audios/               ← 5 PPA4 WAV files (192 kHz, auto-downsampled to 48 kHz)
│   └── labels/               ← Raven Pro selection tables (.Table.1.selections.txt)
└── output/                   ← generated at runtime
    ├── spectrograms/         ← shared .npy mel spectrogram files
    ├── binary/               ← binary model, splits, logs, checkpoints
    ├── multiclass/           ← multiclass model, splits, logs, checkpoints
    └── inference/            ← inference predictions CSVs
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
| 1. Data Exploration | Annotation counts, species distribution, mel spectrogram visualization |
| 2. Build COCO Annotations | `PPA4Reader` converts Raven Pro TSV → COCO-like JSON (binary + multiclass) |
| 3. Binary Classification | AVEVOC vs. noise — `build_windows`, spectrograms, train, evaluate |
| 4. Multiclass Classification | 4 species vs. noise — reuses spectrograms, trains separate model |
| 5. Inference | Both models on a held-out file, probability timeline visualization |

## Using your own data

Swap in your own recordings by replacing the files in `data/audios/` and `data/labels/`
and updating `PPA4_FILES` in cell 2.  For a different annotation format, subclass `BaseReader`
following the `PPA4Reader` pattern.

## Expected runtime

| Environment | Binary training | Multiclass training |
|-------------|----------------|---------------------|
| GPU (A100)  | ~2 min         | ~2 min              |
| CPU (16-core) | ~20–40 min   | ~20–40 min          |

Reduce `epochs` in the config cells to speed up the demo.

## Data citation

> Hernandez-Beltran, V. et al. *Humboldt Aves: A Bioacoustics Dataset for Bird Species
> Classification in the Colombian Amazon*. Zenodo, 2025.
> https://zenodo.org/records/18563039 — CC BY 4.0
