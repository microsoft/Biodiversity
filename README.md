![A colorful banner illustrating various species of animals and plants in a natural environment, symbolizing biodiversity and the use of AI for conservation purposes.][image](https://zenodo.org/records/20044680/files/Biodiversity_Banner.png)

<div align="center"> 
<font size="6"> A Collaborative Deep Learning Framework for Conservation </font>
<br>
<hr>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/pyversions/PytorchWildlife" /></a> 
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<!-- <a href="https://colab.research.google.com/drive/16-OjFVQ6nopuP-gfqofYBBY00oIgbcr1?usp=sharing"><img src="https://img.shields.io/badge/Colab-Video detection-blue?logo=GoogleColab" /></a> -->
<a href="https://github.com/microsoft/CameraTraps/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/PytorchWildlife" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord" /></a>
<a href="https://microsoft.github.io/Biodiversity/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<br><br>
</div>


## 📣 Announcements

### What we've been up to
Hey everyone! It’s been a while since our last update — we hope you haven’t forgotten about us! 😊

Over the past couple of months we’ve been thinking hard about the future of PyTorchWildlife and cooking up some exciting new features just for you.

After two years of community use, one thing has become super clear: most people prefer a nice graphical interface over writing code. People have been asking for a more seamless, unified experience that covers data management, processing, AI inference, analysis, and annotation all in one place.

So we built **Sparrow Studio** — a clean, unified UI built on top of PyTorchWildlife that brings exactly those tools together and much more:

- Local and cloud-based data storage & management
- AI inference using the PyTorchWildlife model zoo
- Post-inference statistics and analysis
- Pre- and post-inference data annotation (easy bounding-box and category editing)
- Embedding visualization and feature retrieval tools

![image](https://zenodo.org/records/18870374/files/sparrow_studio.png)

We’re kicking things off with a beta test before the official release. The Windows MSI installer is available directly on Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress — reach out if you’d like to be on that list.

We’ve also expanded PyTorchWildlife itself into bioacoustics and overhead animal localization — both are out in this release:

- A dedicated [bioacoustics module](./PW_Bioacoustics) with several newly trained bioacoustics models
- [OWL](./demo/image_detection_demo_owl.ipynb) (Overhead Wildlife Locator) — our new generalized, point-based detection model for overhead imagery. (publication on the way.)

Sparrow Studio already has dedicated support for both, so beta testers can run inference and annotate bioacoustics recordings or overhead images directly in the UI.

### The future of PyTorchWildlife

With Sparrow Studio stepping into the picture, PyTorchWildlife itself will gradually evolve into a clean, stable API + high‑quality model zoo layered on top of a general model inference engine — called PW-Engine, while Sparrow Studio becomes the intuitive, everything‑in‑one-place frontend.

**PW-Engine** (PyTorch-Wildlife Engine) is an inference core written in Rust. It is model-agnostic and targets the full PyTorch-Wildlife model zoo and future third party models (e.g. BioClip and Perch) through four consumption surfaces: an HTTP REST API, a single-binary CLI, Python bindings, and a native C library for desktop integration. All four surfaces are feature-complete today; a data-management layer and MLOps functionality are the next milestones. PW-Engine also powers Sparrow Studio under the hood, and the same surfaces are open to anyone building their own frontend. A short overview — what it is, how it fits alongside the current Python API and Sparrow Studio, and how to pilot it — is here: [PW-Engine Overview](https://microsoft.github.io/Biodiversity/pw_engine_overview/).

If you’re interested in API or backend work, or you run an inference-heavy pipeline and want to pilot PW-Engine early, we’d love your help shaping the next chapter of PyTorchWildlife. We’ll update our public task board later.

And one dream we’ve had for a long time: letting non‑coders fine‑tune their own models on their own data. Thanks to recent advances, we’re finally close — and this will be a major focus for both PyTorchWildlife and Sparrow Studio next.

### Why "Sparrow Studio"?
Some of the UI features we needed for PyTorchWildlife also fit naturally as a frontend for [Project SPARROW](https://github.com/microsoft/sparrow-client/), another effort in our group focused on remote data-collection, hardware and edge computing. Since the name “Sparrow” already carried a warm, lively spirit — and the overlap between the projects made things simpler — we decided to call the UI Sparrow Studio. The name just reflects some shared roots and a bit of personality we liked.

[SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed).

Stay tuned! These updates are dropping very soon, and we’d genuinely love to have you in the Sparrow Studio beta. Drop us a message anytime — the more feedback the better! 🐦




#### Previous versions:
- [What's New](https://microsoft.github.io/Biodiversity/releases/release_notes/)


## 👋 Welcome to Pytorch-Wildlife

**PyTorch-Wildlife** is an AI platform designed for the AI for Conservation community to create, modify, and share powerful AI conservation models. It allows users to directly load a variety of models including [MegaDetector](https://microsoft.github.io/Biodiversity/megadetector/), [DeepFaune](https://microsoft.github.io/Biodiversity/megadetector/), and [HerdNet](https://github.com/Alexandre-Delplanque/HerdNet) from our ever expanding [model zoo](https://microsoft.github.io/Biodiversity/model_zoo/megadetector/) for both animal detection and classification. In the future, we will also include models that can be used for applications, including underwater images and bioacoustics. We want to provide a unified and straightforward experience for both practitioners and developers in the AI for conservation field. Your engagement with our work is greatly appreciated, and we eagerly await any feedback you may have.

Explore the codebase, functionalities and user interfaces of **Pytorch-Wildlife** through our [documentation](https://microsoft.github.io/Biodiversity/), interactive [HuggingFace web app](https://huggingface.co/spaces/AndresHdzC/pytorch-wildlife) or local [demos and notebooks](./demo). 

## 🚀 Quick Start

👇 Here is a quick example on how to perform detection and classification on a single image using `PyTorch-wildlife`
```python
import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

img = np.random.randn(3, 1280, 1280)

# Detection
detection_model = pw_detection.MegaDetectorV6() # Model weights are automatically downloaded.
detection_result = detection_model.single_image_detection(img)

#Classification
classification_model = pw_classification.AI4GAmazonRainforest() # Model weights are automatically downloaded.
classification_results = classification_model.single_image_classification(img)
```
More models can be found in our [model zoo](https://microsoft.github.io/Biodiversity/model_zoo/megadetector/)

## ⚙️ Install Pytorch-Wildlife
```
pip install PytorchWildlife
```
Please refer to our [installation guide](https://microsoft.github.io/Biodiversity/installation/) for more installation information.

## 📃 Documentation
Please also go to our newly made documentation page for more information: [![](https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://microsoft.github.io/Biodiversity/)

## 🖼️ Examples

### Image detection using `MegaDetector`
<img src="https://zenodo.org/records/15376499/files/animal_det_1.JPG" alt="animal_det_1" width="300"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Image classification with `MegaDetector` and `AI4GAmazonRainforest`
<img src="https://zenodo.org/records/15376499/files/animal_clas_1.png" alt="animal_clas_1" width="300"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Opossum ID with `MegaDetector` and `AI4GOpossum`
<img src="https://zenodo.org/records/15376499/files/opossum_det.png" alt="opossum_det" width="300"/><br>
*Credits to the Agency for Regulation and Control of Biosecurity and Quarantine for Galápagos (ABG), Ecuador.*


## :fountain_pen: Cite us!
We have recently published a [summary paper on Pytorch-Wildlife](https://arxiv.org/abs/2405.12930). The paper has been accepted as an oral presentation at the [CV4Animals workshop](https://www.cv4animals.com/) at this CVPR 2024. Please feel free to cite us!

```
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation}, 
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

Also, don't forget to cite our original paper for MegaDetector: 

```
@misc{beery2019efficient,
      title={Efficient Pipeline for Camera Trap Image Review},
      author={Sara Beery and Dan Morris and Siyu Yang},
      year={2019}
      eprint={1907.06772},
      archivePrefix={arXiv},
}
```

## 🚀 We’re Open for Contributions!

We’re excited to announce that **Pytorch-Wildlife** is now open to community contributions!  
If you’d like to get involved and help improve the project, we’d love to have you on board.

👉 **Check out our Contribution Guidelines:**  

[📚 How to Participate](https://microsoft.github.io/Biodiversity/contribute/#how-to-participate)

You’ll find everything you need there — from how to pick an issue, to submitting your first pull request.  
Let’s build this together! 🐾🌱


## 🤝 Existing Collaborators and Contributors

The extensive collaborative efforts of Megadetector have genuinely inspired us, and we deeply value its significant contributions to the community. As we continue to advance with Pytorch-Wildlife, our commitment to delivering technical support to our existing partners on MegaDetector remains the same.

Here we list a few of the organizations that have used MegaDetector. We're only listing organizations who have given us permission to refer to them here or have posted publicly about their use of MegaDetector.

We are also building a list of contributors and will release in future updates! Thank you for your efforts!


## Cite us

When citing work that uses any of the repositories under this umbrella, please cite:

- **Hernandez et al. 2024** — *Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation* — for any use of the PyTorch-Wildlife framework or models accessed through it
- **Beery, Morris, Yang 2019** — *Efficient Pipeline for Camera Trap Image Review* — for any use of MegaDetector specifically

A `citation.cff` file is included in this repository for automated citation tools.

## Contributing

We welcome community contributions. See our [Contribution Guidelines](https://microsoft.github.io/CameraTraps/contribute/#how-to-participate) for how to participate.

## Community

For questions about MegaDetector and PyTorch-Wildlife, reach the team via Discord: [![Discord](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

A list of organizations using MegaDetector across global conservation work — six years of partnerships, from national parks to research universities to NGOs — is maintained on the [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) repository.

## About

Maintained by [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/).
