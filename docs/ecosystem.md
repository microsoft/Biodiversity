---
title: "SPARROW project catalogue: which Microsoft biodiversity project to use"
description: "A guide to the Microsoft AI for Good Lab biodiversity projects: which SPARROW or MegaDetector repository fits camera traps, audio, aerial imagery, field devices or model sharing, and where to find the model zoo."
tags:
  - Microsoft biodiversity AI
  - open source conservation AI
  - wildlife monitoring AI tools
  - SPARROW
  - Sparrow Engine
  - MegaDetector
  - model zoo
---

# Project catalogue

The Microsoft AI for Good Lab keeps each biodiversity project in its own repository. This page
helps you pick a starting point, shows how the projects connect, and links the model zoo.
The [home page](index.md#projects) has the short list of repositories.

## Which project should I use?

| You want to | Start here | Also useful |
|---|---|---|
| Try the tools on your own images, videos or audio | [SPARROW Playground](https://github.com/microsoft/SPARROW-Playground): notebooks and scripts, no training | [Quickstart notebook](https://github.com/microsoft/SPARROW-Playground/blob/main/notebooks/00_quickstart.ipynb) |
| Find animals, people and vehicles in camera-trap images | [Sparrow Engine](https://github.com/microsoft/SPARROW-Engine) with a MegaDetector model from the zoo (`pip install sparrow-engine`) | [MegaDetector documentation](https://microsoft.github.io/MegaDetector/): model variants, output format and confidence thresholds |
| Name the species in camera-trap detections | Sparrow Engine with a [camera-trap classifier](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#camera-trap--classifier) from the zoo | [Detection and classification notebook](https://github.com/microsoft/SPARROW-Playground/blob/main/notebooks/03_detection_classification.ipynb) |
| Train or fine-tune a species classifier on your own data | [MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | |
| Detect or identify species in audio recordings | Sparrow Engine with an [acoustic model](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#acoustics--detector) from the zoo | [MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) (the lab's bioacoustic models); [bioacoustics notebook](https://github.com/microsoft/SPARROW-Playground/blob/main/notebooks/04_bioacoustics.ipynb) |
| Detect wildlife in drone or aerial images | [MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) (OWL models) | [Overhead models in the zoo](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#overhead--detector) |
| Run models from the command line, over HTTP, or from C, C++, Go, Rust or C# | Sparrow Engine: `spe` CLI, Docker HTTP server, C library | [User manual](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/user-manual.md) |
| Receive data from SPARROW devices, 4G e-mail cameras, the Reconyx cloud or FTPS cameras | [Sparrow Comm](https://github.com/microsoft/SPARROW-Comm) (`pip install sparrow-comm`) | [Field-data notebooks](https://github.com/microsoft/SPARROW-Playground#receiving-field-data-sparrow-comm) |
| Deploy a solar-powered AI device at a remote site | [SPARROW](https://github.com/microsoft/SPARROW) (AI edge device with 4G or satellite uplink) | [SPARROWmini](https://github.com/microsoft/SPARROWmini) (small nodes that relay over a radio mesh to a SPARROW gateway, or upload directly over Wi-Fi or 4G); [SPARROW documentation](https://microsoft.github.io/SPARROW/) |
| Add your model to the Sparrow Model Zoo | [SPARROW Model Zoo Uploader](https://github.com/microsoft/SPARROW-Model-Zoo-Uploader): convert, check outputs against the original, submit | |
| Use or read the old PyTorch-Wildlife code | The [`archive/pytorch-wildlife`](https://github.com/microsoft/SPARROW-Playground/tree/archive/pytorch-wildlife) branch of SPARROW Playground (unmaintained) | [From PyTorch-Wildlife notebook](https://github.com/microsoft/SPARROW-Playground/blob/main/notebooks/10_from_pytorch_wildlife.ipynb): maps old classes and calls to Sparrow Engine |

## How the projects fit together

1. **Capture.** A [SPARROW](https://github.com/microsoft/SPARROW) device, a
   [SPARROWmini](https://github.com/microsoft/SPARROWmini) node or a commercial camera trap records
   images or audio in the field.
2. **Receive.** [Sparrow Comm](https://github.com/microsoft/SPARROW-Comm) collects that data over the
   device API, e-mail, the Reconyx cloud or FTPS.
3. **Analyse.** [Sparrow Engine](https://github.com/microsoft/SPARROW-Engine) runs models from the
   Sparrow Model Zoo on it: MegaDetector to find animals, then a classifier for species, or acoustic
   models for sound.
4. **Explore.** [SPARROW Playground](https://github.com/microsoft/SPARROW-Playground) shows each step
   in a notebook you can copy into your own code.

Model development happens in the model repositories:
[MegaDetector](https://github.com/microsoft/MegaDetector),
[MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier),
[MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) and
[MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead).
[SPARROW Model Zoo Uploader](https://github.com/microsoft/SPARROW-Model-Zoo-Uploader) converts a
trained model, checks its outputs against the original, and submits it to the zoo that Sparrow
Engine runs.

## Model zoo

The Sparrow Model Zoo holds the lab's models and third-party models for camera traps, acoustics,
overhead and marine imagery, and general use. Each model keeps its own licence, so check it before
use, especially for commercial work.

- **Catalogue:** the
  [Sparrow Engine model zoo catalogue](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md)
  lists every model with its source, licence and citation. Sections:
  [camera-trap detectors](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#camera-trap--detector),
  [camera-trap classifiers](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#camera-trap--classifier),
  [acoustic detectors](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#acoustics--detector),
  [acoustic classifiers](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#acoustics--classifier),
  [overhead detectors](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#overhead--detector),
  [marine imagery detectors](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#marine-imagery--detector),
  [licensing at a glance](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md#licensing-at-a-glance).
- **Model files:** the [Sparrow Model Zoo on Zenodo](https://doi.org/10.5281/zenodo.20348978). This
  DOI always opens the latest version.
- **MegaDetector accuracy and model sizes:** the
  [MegaDetector model zoo page](https://microsoft.github.io/MegaDetector/model_zoo/).
- **Earlier weights:** the AI4G-Opossum classifier from PyTorch-Wildlife is not in the Sparrow Model
  Zoo. Its original weights (`OpossumClassification_v0.0.0.ckpt`) stay on
  [Zenodo record 10023414](https://zenodo.org/records/10023414).

The model zoo pages that used to live on this site (`model_zoo/megadetector/`,
`model_zoo/classifiers/`, `model_zoo/other_detectors/`, `model_zoo/bioacoustics/`) now redirect to
the matching sections of the catalogue.

## Get involved

Ask questions on [Discord](https://discord.gg/TeEVxzaYtm). Bug reports and feature requests go to
the repository of the project they concern; see [Contribute](contribute.md). The teams already
using these tools are on the [collaborators page](collaborators.md).
