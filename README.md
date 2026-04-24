![image](https://zenodo.org/records/15376499/files/Pytorch_Banner_transparentbk.png)

<div align="center">
<font size="6"> MegaDetector & PyTorch Wildlife: AI for Conservation </font>
<br>
<hr>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/pyversions/PytorchWildlife" /></a>
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<a href="https://github.com/microsoft/CameraTraps/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/PytorchWildlife" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord" /></a>
<a href="https://microsoft.github.io/CameraTraps/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<br><br>
</div>


## MegaDetector

**MegaDetector** is an AI model that detects **animals**, **people**, and **vehicles** in camera trap images. It does not identify species — it finds the animals so you can spend less time scrolling through empty frames. MegaDetector is developed by the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/ai/ai-for-good) and is used by **80+ conservation organizations** worldwide.

The latest version, **MegaDetector V6**, uses modern architectures (YOLOv9, YOLOv10, RT-DETR) that are dramatically smaller and faster than V5 while maintaining comparable accuracy. The compact YOLOv10 variant has only **2% of the parameters** of MegaDetector V5.

**Get started in 3 lines:**

```python
pip install PytorchWildlife
```

```python
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()
results = model.single_image_detection("path/to/camera_trap_image.jpg")
```

For a comprehensive guide to MegaDetector, including benchmarks, version history, tips for large-scale processing, and more, see the full **[MegaDetector documentation](megadetector.md)**.


## SPARROW Studio

MegaDetector is the flagship model in **SPARROW Studio**, a unified platform for conservation AI built on top of **PyTorch Wildlife**. SPARROW Studio brings together everything you need for camera trap and wildlife monitoring workflows:

- **AI inference** using the full PyTorch Wildlife model zoo (MegaDetector, species classifiers, and more)
- **Local and cloud-based** data storage and management
- **Post-inference statistics** and analysis
- **Pre- and post-inference annotation** (bounding-box and category editing)
- **Embedding visualization** and feature retrieval tools
- **Bioacoustics support** for audio-based wildlife monitoring

The Windows MSI installer is available on Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress — reach out if you'd like to be on that list.

![image](https://zenodo.org/records/18870374/files/sparrow_studio.png)


## Quick Start

Here is a quick example showing detection and classification on a single image using PyTorch Wildlife:

```python
import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

img = np.random.randn(3, 1280, 1280)

# Detection with MegaDetector V6
detection_model = pw_detection.MegaDetectorV6()  # Weights are automatically downloaded
detection_result = detection_model.single_image_detection(img)

# Classification with a species classifier
classification_model = pw_classification.AI4GAmazonRainforest()  # Weights are automatically downloaded
classification_results = classification_model.single_image_classification(img)
```

More models can be found in our [model zoo](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/).


## Install PyTorch Wildlife

```
pip install PytorchWildlife
```

Please refer to our [installation guide](https://microsoft.github.io/CameraTraps/installation/) for more information, including GPU setup and conda environments.


## Model Zoo

PyTorch Wildlife provides a growing collection of models for conservation AI:

### Detection Models

| Model | Task | Architecture | Params | License |
| --- | --- | --- | --- | --- |
| **MegaDetector V6** (multiple variants) | Animal/person/vehicle detection in camera traps | YOLOv9, YOLOv10, RT-DETR | 2.3M - 76M | MIT, Apache-2.0, AGPL-3.0 |
| **MegaDetector V5** | Animal/person/vehicle detection in camera traps | YOLOv5 | 139.9M | AGPL-3.0 |
| **DeepFaune Detector** | Wildlife detection | YOLOv8s | — | CC BY-SA 4.0 |
| **HerdNet** | Animal detection in aerial/overhead imagery | — | — | CC BY-NC-SA 4.0 |
| **OWL** (CNN and Transformer) | Point-based overhead wildlife localization | CNN / Transformer | — | MIT |

### Classification Models

| Model | Task | Classes | License |
| --- | --- | --- | --- |
| **AI4G Amazon Rainforest** | Amazon wildlife species classification | 36 | MIT |
| **AI4G Snapshot Serengeti** | Serengeti wildlife classification | 10 | MIT |
| **AI4G Opossum** | Opossum binary classification | 2 | MIT |
| **DeepFaune Classifier** | European wildlife classification (multilingual) | 34 | CC BY-SA 4.0 |
| **DFNE** | New England wildlife classification | 23 | CC0 1.0 |

### Bioacoustics

| Model | Task | License |
| --- | --- | --- |
| **MD AudioBirds V1** | Bird species classification from audio | MIT |

See the full [model zoo documentation](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/) for benchmarks, download links, and usage examples.


## Documentation

Visit our documentation site for installation guides, API reference, tutorials, and more:

[![](https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://microsoft.github.io/CameraTraps/)


## Examples

### Image detection using MegaDetector
<img src="https://zenodo.org/records/15376499/files/animal_det_1.JPG" alt="MegaDetector animal detection example" width="300"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Image classification with MegaDetector and AI4G Amazon Rainforest
<img src="https://zenodo.org/records/15376499/files/animal_clas_1.png" alt="MegaDetector classification example" width="300"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Opossum ID with MegaDetector and AI4G Opossum
<img src="https://zenodo.org/records/15376499/files/opossum_det.png" alt="MegaDetector opossum detection example" width="300"/><br>
*Credits to the Agency for Regulation and Control of Biosecurity and Quarantine for Galapagos (ABG), Ecuador.*


## Who Uses MegaDetector?

The extensive collaborative efforts behind MegaDetector have genuinely inspired us, and we deeply value its significant contributions to the conservation community. Here we list organizations that have used MegaDetector and have given us permission to reference them here or have posted publicly about their use of MegaDetector:

**Government Agencies**
- [Arizona Department of Environmental Quality](http://azdeq.gov/)
- [Idaho Department of Fish and Game](https://idfg.idaho.gov/)
- [Oregon Department of Fish and Wildlife — Wildlife Research](https://www.dfw.state.or.us/wildlife/research/index.asp)
- [Michigan Department of Natural Resources — Wildlife Division](https://www.michigan.gov/dnr/about/contact/wildlife)
- [Banff National Park Resource Conservation, Parks Canada](https://www.pc.gc.ca/en/pn-np/ab/banff/nature/conservation)
- [Capitol Reef National Park / Utah Valley University](https://www.nps.gov/care/index.htm)
- [Santa Monica Mountains Recreation Area, National Park Service](https://www.nps.gov/samo/index.htm)
- [National Wildlife Refuge System, Southwest Region, U.S. Fish & Wildlife Service](https://www.fws.gov/about/region/southwest)
- [Kenai National Wildlife Refuge, U.S. Fish & Wildlife Service](https://www.fws.gov/refuge/kenai) ([story](https://www.peninsulaclarion.com/sports/refuge-notebook-new-technology-increases-efficiency-of-refuge-cameras/))
- Protected Areas Unit, Canadian Wildlife Service

**Conservation Organizations**
- [Island Conservation](https://www.islandconservation.org/)
- [The Nature Conservancy in Wyoming](https://www.nature.org/en-us/about-us/where-we-work/united-states/wyoming/)
- [The Nature Conservancy in California](https://www.nature.org/en-us/about-us/where-we-work/united-states/california/) ([Animl platform](https://github.com/tnc-ca-geo/animl-frontend))
- [Conservation X Labs](https://conservationxlabs.com/)
- [Wildlife Protection Solutions](https://wildlifeprotectionsolutions.org/) ([Microsoft story](https://customers.microsoft.com/en-us/story/1384184517929343083-wildlife-protection-solutions-nonprofit-ai-for-earth), [story](https://www.enterpriseai.news/2023/02/20/ai-helps-wildlife-protection-solutions-safeguard-endangered-species/))
- [Australian Wildlife Conservancy](https://www.australianwildlife.org/) ([blog](https://www.australianwildlife.org/cutting-edge-technology-delivering-efficiency-gains-in-conservation/), [blog](https://www.australianwildlife.org/efficiency-gains-at-the-cutting-edge-of-technology/))
- [SPEA (Portuguese Society for the Study of Birds)](https://spea.pt/en/)
- [Gola Forest Programme, Royal Society for the Protection of Birds (RSPB)](https://www.rspb.org.uk/our-work/conservation/projects/scientific-support-for-the-gola-forest-programme/)
- [Canadian Parks and Wilderness Society (CPAWS) Northern Alberta Chapter](https://cpawsnab.org/)
- [Felidae Conservation Fund](https://felidaefund.org/) ([WildePod platform](https://wildepod.org/)) ([blog post](https://abhaykashyap.com/blog/ai-powered-camera-trap-image-annotation-system/))
- [Upper Yellowstone Watershed Group](https://www.upperyellowstone.org/)
- [Irvine Ranch Conservancy](http://www.irconservancy.org/) ([story](https://www.ocregister.com/2022/03/30/ai-software-is-helping-researchers-focus-on-learning-about-ocs-wild-animals/))

**Universities & Research Labs**
- [Blumstein Lab, UCLA](https://blumsteinlab.eeb.ucla.edu/)
- [Quantitative Ecology Lab, University of Washington](https://depts.washington.edu/sefsqel/)
- [Wildlife Coexistence Lab, University of British Columbia](https://wildlife.forestry.ubc.ca/)
- [Applied Conservation Macro Ecology Lab, University of Victoria](http://www.acmelab.ca/)
- [McLoughlin Lab in Population Ecology, University of Saskatchewan](http://mcloughlinlab.ca/lab/)
- [Mammal Spatial Ecology and Conservation Lab, Washington State University](https://labs.wsu.edu/dthornton/)
- [Department of Fish and Wildlife Sciences, University of Idaho](https://www.uidaho.edu/cnr/departments/fish-and-wildlife-sciences)
- [Department of Wildlife Ecology and Conservation, University of Florida](https://wec.ifas.ufl.edu/)
- [Cross-Cultural Ecology Lab, Macquarie University](https://crossculturalecology.net/)
- [Centre for Ecosystem Science, UNSW Sydney](https://www.unsw.edu.au/research/)
- [School of Natural Sciences, University of Tasmania](https://www.utas.edu.au/natural-sciences) ([story](https://www.utas.edu.au/about/news-and-stories/articles/2022/1204-innovative-camera-network-keeps-close-eye-on-tassie-wildlife))
- [Czech University of Life Sciences Prague](https://www.czu.cz/en)
- [Borderlands Research Institute, Sul Ross State University](https://bri.sulross.edu/)
- [Graeme Shannon's Research Group, Bangor University](https://wildliferesearch.co.uk/group-1)
- [Institut des Sciences de la Foret Temperee (ISFORT), Universite du Quebec en Outaouais](https://isfort.uqo.ca/)
- [Lab of Dr. Bilal Habib, Wildlife Institute of India](https://bhlab.in/about)
- Department of Ecology, TU Berlin

**Museums & Zoos**
- [Center for Biodiversity and Conservation, American Museum of Natural History](https://www.amnh.org/research/center-for-biodiversity-conservation)
- [Northern Great Plains Program, Smithsonian](https://nationalzoo.si.edu/news/restoring-americas-prairie)
- [Snapshot USA, Smithsonian](https://emammal.si.edu/snapshot-usa)
- [Seattle Urban Carnivore Project, Woodland Park Zoo](https://www.zoo.org/seattlecarnivores)
- [San Diego Zoo Wildlife Alliance](https://science.sandiegozoo.org/) ([Animl R package](https://github.com/conservationtechlab/animl))
- [Taronga Conservation Society](https://taronga.org.au/)
- [Hamaarag, The Steinhardt Museum of Natural History, Tel Aviv University](https://hamaarag.org.il/)

**Other Organizations & Platforms**
- [TerrOiko](https://www.terroiko.fr/) ([OCAPI platform](https://www.terroiko.fr/ocapi))
- [Blackbird Environmental](https://blackbirdenv.com/)
- [Camelot](https://camelotproject.org/)
- [EcoLogic Consultants Ltd.](https://www.consult-ecologic.com/)
- [Estacion Biologica de Donana](http://www.ebd.csic.es/inicio)
- [Myall Lakes Dingo Project](https://carnivorecoexistence.info/myall-lakes-dingo-project/)
- [Point No Point Treaty Council](https://pnptc.org/)
- [Ramat Hanadiv Nature Park](https://www.ramat-hanadiv.org.il/en/)
- [Synthetaic](https://www.synthetaic.com/)
- [TrapTagger](https://wildeyeconservation.org/trap-tagger-about/)
- [DC Cat Count, Humane Rescue Alliance](https://hub.dccatcount.org/)
- [Ecology and Conservation of Amazonian Vertebrates Research Group, Federal University of Amapa](https://www.researchgate.net/lab/Fernanda-Michalski-Lab-4)
- [Serra dos Orgaos National Park, ICMBio](https://www.icmbio.gov.br/parnaserradosorgaos/)
- [Shan Shui Conservation Center](http://en.shanshui.org/) ([blog post](https://mp.weixin.qq.com/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs))
- [Road Ecology Center, UC Davis](https://roadecology.ucdavis.edu/) ([Wildlife Observer Network](https://wildlifeobserver.net/))
- [Alberta Biodiversity Monitoring Institute (ABMI)](https://www.abmi.ca/home.html) ([WildTrax platform](https://www.wildtrax.ca/)) ([blog post](https://wildcams.ca/blog/the-abmi-visits-the-zoo/))
- Ghost Cat Analytics

> [!IMPORTANT]
> If you would like to be added to this list or have any questions regarding MegaDetector and PyTorch Wildlife, please [email us](mailto:zhongqimiao@microsoft.com) or join us in our Discord channel: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)


## Cite Us

If you use PyTorch Wildlife or MegaDetector in your research, please cite both papers:

**PyTorch Wildlife:**
```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

**MegaDetector:**
```bibtex
@misc{beery2019efficient,
      title={Efficient Pipeline for Camera Trap Image Review},
      author={Sara Beery and Dan Morris and Siyu Yang},
      year={2019},
      eprint={1907.06772},
      archivePrefix={arXiv},
}
```


## Contributing

We're open to community contributions! Check out our [contribution guidelines](https://microsoft.github.io/CameraTraps/contribute/#how-to-participate) to get started.


## Announcements

### The Future of PyTorch Wildlife

With SPARROW Studio stepping into the picture, PyTorch Wildlife itself will gradually evolve into a clean, stable API + high-quality model zoo layered on top of a general model inference engine — called **PW-Engine** — while SPARROW Studio becomes the intuitive, everything-in-one-place frontend.

**PW-Engine** (PyTorch-Wildlife Engine) is an inference core written in Rust. It is model-agnostic and targets the full PyTorch Wildlife model zoo and future third-party models (e.g., BioClip and Perch) through four consumption surfaces: an HTTP REST API, a single-binary CLI, Python bindings, and a native C library for desktop integration. All four surfaces are feature-complete today; a data-management layer and MLOps functionality are the next milestones. PW-Engine also powers SPARROW Studio under the hood, and the same surfaces are open to anyone building their own frontend. A short overview is here: [PW-Engine Overview](https://microsoft.github.io/CameraTraps/pw_engine_overview/).

We've also expanded PyTorch Wildlife into bioacoustics and overhead animal localization:

- A dedicated [bioacoustics module](./PW_Bioacoustics) with several newly trained bioacoustics models
- [OWL](./demo/image_detection_demo_owl.ipynb) (Overhead Wildlife Locator) — a new generalized, point-based detection model for overhead imagery (publication on the way)

SPARROW Studio already has dedicated support for both, so beta testers can run inference and annotate bioacoustics recordings or overhead images directly in the UI.

### Why "Sparrow Studio"?

Some of the UI features we needed for PyTorch Wildlife also fit naturally as a frontend for Project Sparrow, another effort in our group focused on remote data-collection hardware and edge computing. Since the name "Sparrow" already carried a warm, lively spirit — and the overlap between the projects made things simpler — we decided to call the UI SPARROW Studio. The name reflects some shared roots and a bit of personality we liked.

### Previous Releases

- [What's New](https://microsoft.github.io/CameraTraps/releases/release_notes/)
