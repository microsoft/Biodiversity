# Main changes and additions

### V 1.3.0

This release ships Sparrow Studio beta, a full bioacoustics module, OWL (overhead animal detection), and previews the next-generation **PW-Engine** inference core.

#### Sparrow Studio beta

A graphical frontend for data management, inference, analysis, and annotation in one UI. Signed Windows MSI available directly from Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1). Mac and Linux builds are in progress.

#### Bioacoustics module

A dedicated `PW_Bioacoustics/` module with CLI scripts for dataset preparation, training, and inference, plus a pre-trained bird classifier `MD_AudioBirds_V1` (ONNX). See the [bioacoustics overview](../bioacoustics.md), the [model-zoo entry](../model_zoo/bioacoustics.md), and the [end-to-end demo](https://github.com/microsoft/CameraTraps/tree/main/PW_Bioacoustics/demo).

#### OWL (Overhead Wildlife Locator)

A generalized, point-based detection model for overhead imagery. Two variants released — **OWL-T** and **OWL-C** — listed in the [Other Detectors](../model_zoo/other_detectors.md) model zoo. Publication on the way. Demo: [`image_detection_demo_owl.ipynb`](https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_owl.ipynb).

#### PW-Engine preview — the future of PyTorch-Wildlife

A Rust-based, model-agnostic inference core that will sit under both the Python API and Sparrow Studio. Four consumption surfaces: HTTP REST, a single-binary CLI, Python bindings, and a native C library for desktop integration. All four are feature-complete today; a data-management layer and MLOps functionality are next. Read the [PW-Engine Overview](../pw_engine_overview.md).

#### Install

```bash
pip install -U PytorchWildlife==1.3.0
```

#### Under the hood

- `version.txt` is now the single source of truth for the package version
- MIT license headers added to OWL model sources
- MkDocs `mkdocstrings` paths refreshed after the `localization/` reorganisation; docs build cleanly again
