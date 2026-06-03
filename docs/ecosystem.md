---
title: "The Microsoft Biodiversity Ecosystem: Open-Source Conservation AI Projects"
description: "A guide to the Microsoft AI for Good Lab biodiversity ecosystem: which open-source conservation AI project fits camera traps, audio, field devices, or model development."
tags:
  - Microsoft biodiversity AI
  - open source conservation AI
  - wildlife monitoring AI tools
  - AI for Good biodiversity
  - conservation AI
---

# The Microsoft Biodiversity Ecosystem

The Microsoft AI for Good Lab maintains a family of open-source projects for biodiversity
monitoring and conservation. Each one owns a distinct part of the workflow, from detecting
animals in imagery to classifying sounds, deploying field hardware, and building custom models.
This page is the directory for the ecosystem and a guide to picking the right starting point.

## Which project should I use?

Start from what you already have or what you are trying to build.

- **I have camera-trap photos and want to find the animals in them.** Use
  [MegaDetector](https://microsoft.github.io/MegaDetector/), the detector that locates animals,
  people, and vehicles in camera-trap images and clears the empty frames out of your review queue.
- **I have audio recordings and want to identify what is calling.** Use
  [MegaDetector-Acoustic](https://microsoft.github.io/MegaDetector-Acoustic/) for terrestrial
  bioacoustic classification and species identification from sound.
- **I am deploying a sensor in the field and need it to run on its own power.** Use
  [SPARROW](https://microsoft.github.io/SPARROW/), the solar-powered edge device that runs these
  models on site and sends results back over remote connectivity.
- **I want to build, train, or integrate conservation models in my own code.** Use
  [PyTorch-Wildlife](https://microsoft.github.io/Pytorch-Wildlife/), the deep learning framework
  and model zoo that ties the ecosystem together in Python.
- **I want to adapt a species classifier to my own region or dataset.** Use
  [MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) to fine-tune
  classification on top of detections.

## Projects

### Detection

- **[MegaDetector](https://microsoft.github.io/MegaDetector/)** is where the ecosystem started.
  It detects animals, people, and vehicles in camera-trap imagery and filters blank frames, which
  removes the bulk of manual review on large datasets. See the
  [camera-trap image detection documentation](https://microsoft.github.io/MegaDetector/) for run
  paths, the model zoo, and output format.
- **[MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead)** brings detection
  to aerial and drone imagery with point-based wildlife localization from overhead views.
- **[MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar)** extends detection to
  sidescan sonar imagery for underwater wildlife surveys.

### Audio

- **[MegaDetector-Acoustic](https://microsoft.github.io/MegaDetector-Acoustic/)** handles the
  audio side of monitoring: classifying calls and identifying terrestrial species from sound
  recordings, the bioacoustic counterpart to image-based detection.

### Classification

- **[MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier)** fine-tunes
  species classification on top of detections, so you can adapt a model to the species and regions
  in your own data.

### Field hardware

- **[SPARROW](https://microsoft.github.io/SPARROW/)**, the Solar-Powered Acoustic and Remote
  Recording Observation Watch, is the edge device for remote deployments. It captures imagery and
  audio, runs ecosystem models on site, and reports results back over satellite or cellular links.

### Framework

- **[PyTorch-Wildlife](https://microsoft.github.io/Pytorch-Wildlife/)** is the collaborative deep
  learning framework and model zoo for conservation AI. It loads detection and classification
  models in a few lines of Python and is the integration point for building on top of the ecosystem.

## How the projects fit together

A typical pipeline moves from capture to insight. A field unit such as SPARROW collects imagery
and audio in a remote site. MegaDetector finds the animals in the images and discards empty frames.
MegaDetector-Classifier or a model from the PyTorch-Wildlife zoo names the species in those
detections, while MegaDetector-Acoustic does the same for the audio. PyTorch-Wildlife is the common
framework that runs each model and lets you assemble these steps into your own workflow.

## Get involved

The projects share a home in the [Microsoft Biodiversity organization](https://github.com/microsoft/Biodiversity)
and a community on [Discord](https://discord.gg/TeEVxzaYtm). See the list of teams already using
these tools on the [collaborators page](collaborators.md), and the
[ecosystem documentation standards](ecosystem-standards.md) if you are bringing a new project into
the family.
