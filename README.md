![image](https://zenodo.org/records/15376499/files/Pytorch_Banner_transparentbk.png)

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
<a href="https://microsoft.github.io/CameraTraps/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<br><br>
</div>


## 📣 Announcements

### What we've been up to
Hey everyone! It’s been a while since our last update — we hope you haven’t forgotten about us! 😊

Over the past couple of months we’ve been thinking hard about the future of PyTorchWildlife and cooking up some exciting new features just for you.

After two years of community use, one thing has become super clear: most people prefer a nice graphical interface over writing code. People have been asking for a more seamless, unified experience that covers data management, processing, AI inference, analysis, and annotation all in one place.

So we built Sparrow Studio — a clean, unified UI built on top of PyTorchWildlife that brings exactly those tools together:

- Local and cloud-based data storage & management
- AI inference using the PyTorchWildlife model zoo
- Post-inference statistics and analysis
- Pre- and post-inference data annotation (easy bounding-box and category editing)
- Embedding visualization and feature retrieval tools

We’re kicking things off with a beta test before the official release. The Windows MSI installer is available directly on Zenodo: [SPARROW Studio Installer](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1) (signed). Mac and Linux builds are in progress — reach out if you’d like to be on that list.

We’ve also been expanding PyTorchWildlife itself into bioacoustics and overhead animal localization. Later this month we’ll release:

- A dedicated bioacoustics module + several newly trained bioacoustics models
- OWL (Overhead Wildlife Locator) — our new generalized, point-based detection model for overhead imagery

Even better, Sparrow Studio already has dedicated support for both, so beta testers will be able to run inference and annotate bioacoustics recordings or overhead images directly in the UI.

### A new inference engine under the hood
Alongside the Sparrow Studio beta, we’ve been building **PW-Engine** (PyTorch-Wildlife Engine) — a new inference core written in Rust. It is model-agnostic and runs the same model set (MegaDetector, DeepFaune, HerdNet, OWL, SpeciesNet, and the new bioacoustics models) through four surfaces: an HTTP REST API, a single-binary CLI, Python bindings, and a native C library for desktop integration. All four surfaces are feature-complete today; a data-management layer is the next milestone. PW-Engine will be released under the MIT license, with a standalone public repository to follow the Sparrow Studio beta. A short overview — what it is, how it fits alongside the Python API and Sparrow Studio, and how to pilot it — is here: [PW-Engine Overview](<PW_ENGINE_2PAGER_LINK>).

If you run an inference-heavy pipeline and want to pilot PW-Engine early, please get in touch.

### The future of PyTorchWildlife
With Sparrow Studio stepping into the picture, PyTorchWildlife will gradually evolve into a clean, stable API + high‑quality model zoo, Sparrow Studio becomes the intuitive, everything‑in‑one-place frontend, and PW-Engine becomes the shared inference core behind both.

If you’re interested in API or backend work, we’d love your help shaping the next chapter of PyTorchWildlife. We’ll update our public task board later this month.

And one dream we’ve had for a long time: letting non‑coders fine‑tune their own models on their own data. Thanks to recent advances, we’re finally close — and this will be a major focus for both PyTorchWildlife and Sparrow Studio next.

### Why "Sparrow Studio"?
Some of the UI features we needed for PyTorchWildlife also fit naturally as a frontend for Project Sparrow, another effort in our group focused on remote data-collection hardwares and edge computing. Since the name “Sparrow” already carried a warm, lively spirit — and the overlap between the projects made things simpler — we decided to call the UI Sparrow Studio. The name just reflects some shared roots and a bit of personality we liked.

Stay tuned! These updates are dropping very soon, and we’d genuinely love to have you in the Sparrow Studio beta. Drop us a message anytime — the more feedback the better! 🐦

![image](https://zenodo.org/records/18870374/files/sparrow_studio.png)


#### Previous versions:
- [What's New](https://microsoft.github.io/CameraTraps/releases/release_notes/)


## 👋 Welcome to Pytorch-Wildlife

**PyTorch-Wildlife** is an AI platform designed for the AI for Conservation community to create, modify, and share powerful AI conservation models. It allows users to directly load a variety of models including [MegaDetector](https://microsoft.github.io/CameraTraps/megadetector/), [DeepFaune](https://microsoft.github.io/CameraTraps/megadetector/), and [HerdNet](https://github.com/Alexandre-Delplanque/HerdNet) from our ever expanding [model zoo](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/) for both animal detection and classification. In the future, we will also include models that can be used for applications, including underwater images and bioacoustics. We want to provide a unified and straightforward experience for both practicioners and developers in the AI for conservation field. Your engagement with our work is greatly appreciated, and we eagerly await any feedback you may have.

Explore the codebase, functionalities and user interfaces of **Pytorch-Wildlife** through our [documentation](https://microsoft.github.io/CameraTraps/), interactive [HuggingFace web app](https://huggingface.co/spaces/AndresHdzC/pytorch-wildlife) or local [demos and notebooks](./demo). 

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
More models can be found in our [model zoo](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/)

## ⚙️ Install Pytorch-Wildlife
```
pip install PytorchWildlife
```
Please refer to our [installation guide](https://microsoft.github.io/CameraTraps/installation/) for more installation information.

## 📃 Documentation
Please also go to our newly made dofumentation page for more information: [![](https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://microsoft.github.io/CameraTraps/)

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

[📚 How to Participate](https://microsoft.github.io/CameraTraps/contribute/#how-to-participate)

You’ll find everything you need there — from how to pick an issue, to submitting your first pull request.  
Let’s build this together! 🐾🌱


## 🤝 Existing Collaborators and Contributors

The extensive collaborative efforts of Megadetector have genuinely inspired us, and we deeply value its significant contributions to the community. As we continue to advance with Pytorch-Wildlife, our commitment to delivering technical support to our existing partners on MegaDetector remains the same.

Here we list a few of the organizations that have used MegaDetector. We're only listing organizations who have given us permission to refer to them here or have posted publicly about their use of MegaDetector.

We are also building a list of contributors and will release in future updates! Thank you for your efforts!

<details>
<summary><font size="3">👉 Full list of organizations</font></summary>

<ul>
  <li>(Newly Added) <a href="https://www.terroiko.fr/">TerrOïko</a> (<a href="https://www.terroiko.fr/ocapi">OCAPI platform</a>)</li>
  <li><a href="http://azdeq.gov/">Arizona Department of Environmental Quality</a></li>
  <li><a href="https://blackbirdenv.com/">Blackbird Environmental</a></li>
  <li><a href="https://camelotproject.org/">Camelot</a></li>
  <li><a href="https://cpawsnab.org/">Canadian Parks and Wilderness Society (CPAWS) Northern Alberta Chapter</a></li>
  <li><a href="https://conservationxlabs.com/">Conservation X Labs</a></li>
  <li><a href="https://www.czu.cz/en">Czech University of Life Sciences Prague</a></li>
  <li><a href="https://www.consult-ecologic.com/">EcoLogic Consultants Ltd.</a></li>
  <li><a href="http://www.ebd.csic.es/inicio">Estación Biológica de Doñana</a></li>
  <li><a href="https://idfg.idaho.gov/">Idaho Department of Fish and Game</a></li>
  <li><a href="https://www.islandconservation.org/">Island Conservation</a></li>
  <li><a href="https://carnivorecoexistence.info/myall-lakes-dingo-project/">Myall Lakes Dingo Project</a></li>
  <li><a href="https://pnptc.org/">Point No Point Treaty Council</a></li>
  <li><a href="https://www.ramat-hanadiv.org.il/en/">Ramat Hanadiv Nature Park</a></li>
  <li><a href="https://spea.pt/en/">SPEA (Portuguese Society for the Study of Birds)</a></li>
  <li><a href="https://www.synthetaic.com/">Synthetaic</a></li>
  <li><a href="https://taronga.org.au/">Taronga Conservation Society</a></li>
  <li><a href="https://www.nature.org/en-us/about-us/where-we-work/united-states/wyoming/">The Nature Conservancy in Wyoming</a></li>
  <li><a href="https://wildeyeconservation.org/trap-tagger-about/">TrapTagger</a></li>
  <li><a href="https://www.upperyellowstone.org/">Upper Yellowstone Watershed Group</a></li>
  <li><a href="http://www.acmelab.ca/">Applied Conservation Macro Ecology Lab</a>, University of Victoria</li>
  <li><a href="https://www.pc.gc.ca/en/pn-np/ab/banff/nature/conservation">Banff National Park Resource Conservation</a>, <a href="https://www.pc.gc.ca/en/pn-np/ab/banff/nature/conservation">Parks Canada</a></li>
  <li><a href="https://blumsteinlab.eeb.ucla.edu/">Blumstein Lab</a>, UCLA</li>
  <li><a href="https://bri.sulross.edu/">Borderlands Research Institute</a>, Sul Ross State University</li>
  <li><a href="https://www.nps.gov/care/index.htm">Capitol Reef National Park</a> / Utah Valley University</li>
  <li><a href="https://www.amnh.org/research/center-for-biodiversity-conservation">Center for Biodiversity and Conservation</a>, American Museum of Natural History</li>
  <li><a href="https://www.unsw.edu.au/research/">Centre for Ecosystem Science</a>, UNSW Sydney</li>
  <li><a href="https://crossculturalecology.net/">Cross-Cultural Ecology Lab</a>, Macquarie University</li>
  <li><a href="https://hub.dccatcount.org/">DC Cat Count</a>, led by the Humane Rescue Alliance</li>
  <li><a href="https://www.uidaho.edu/cnr/departments/fish-and-wildlife-sciences">Department of Fish and Wildlife Sciences</a>, University of Idaho</li>
  <li><a href="https://wec.ifas.ufl.edu/">Department of Wildlife Ecology and Conservation</a>, University of Florida</li>
  <li><a href="https://www.researchgate.net/lab/Fernanda-Michalski-Lab-4">Ecology and Conservation of Amazonian Vertebrates Research Group</a>, Federal University of Amapá</li>
  <li><a href="https://www.rspb.org.uk/our-work/conservation/projects/scientific-support-for-the-gola-forest-programme/">Gola Forest Programme</a>, Royal Society for the Protection of Birds (RSPB)</li>
  <li><a href="https://wildliferesearch.co.uk/group-1">Graeme Shannon's Research Group</a>, Bangor University</li>
  <li><a href="https://hamaarag.org.il/">Hamaarag</a>, The Steinhardt Museum of Natural History, Tel Aviv University</li>
  <li><a href="https://isfort.uqo.ca/">Institut des Science de la Forêt Tempérée (ISFORT)</a>, Université du Québec en Outaouais</li>
  <li><a href="https://bhlab.in/about">Lab of Dr. Bilal Habib</a>, the Wildlife Institute of India</li>
  <li><a href="https://labs.wsu.edu/dthornton/">Mammal Spatial Ecology and Conservation Lab</a>, Washington State University</li>
  <li><a href="http://mcloughlinlab.ca/lab/">McLoughlin Lab in Population Ecology</a>, University of Saskatchewan</li>
  <li><a href="https://www.fws.gov/about/region/southwest">National Wildlife Refuge System, Southwest Region</a>, U.S. Fish & Wildlife Service</li>
  <li><a href="https://nationalzoo.si.edu/news/restoring-americas-prairie">Northern Great Plains Program</a>, Smithsonian</li>
  <li><a href="https://depts.washington.edu/sefsqel/">Quantitative Ecology Lab</a>, University of Washington</li>
  <li><a href="https://www.nps.gov/samo/index.htm">Santa Monica Mountains Recreation Area</a>, National Park Service</li>
  <li><a href="https://www.zoo.org/seattlecarnivores">Seattle Urban Carnivore Project</a>, Woodland Park Zoo</li>
  <li><a href="https://www.icmbio.gov.br/parnaserradosorgaos/">Serra dos Órgãos National Park</a>, ICMBio</li>
  <li><a href="https://emammal.si.edu/snapshot-usa">Snapshot USA</a>, Smithsonian</li>
  <li><a href="https://wildlife.forestry.ubc.ca/">Wildlife Coexistence Lab</a>, University of British Columbia</li>
  <li><a href="https://www.dfw.state.or.us/wildlife/research/index.asp">Wildlife Research</a>, Oregon Department of Fish and Wildlife</li>
  <li><a href="https://www.michigan.gov/dnr/about/contact/wildlife">Wildlife Division</a>, Michigan Department of Natural Resources</li>
  <li>Department of Ecology, TU Berlin</li>
  <li>Ghost Cat Analytics</li>
  <li>Protected Areas Unit, Canadian Wildlife Service</li>
  <li><a href="https://www.utas.edu.au/natural-sciences">School of Natural Sciences</a>, University of Tasmania (<a href="https://www.utas.edu.au/about/news-and-stories/articles/2022/1204-innovative-camera-network-keeps-close-eye-on-tassie-wildlife">story</a>)</li>
  <li><a href="https://www.fws.gov/refuge/kenai">Kenai National Wildlife Refuge</a>, U.S. Fish & Wildlife Service (<a href="https://www.peninsulaclarion.com/sports/refuge-notebook-new-technology-increases-efficiency-of-refuge-cameras/">story</a>)</li>
  <li><a href="https://www.australianwildlife.org/">Australian Wildlife Conservancy</a> (<a href="https://www.australianwildlife.org/cutting-edge-technology-delivering-efficiency-gains-in-conservation/">blog</a>, <a href="https://www.australianwildlife.org/efficiency-gains-at-the-cutting-edge-of-technology/">blog</a>)</li>
  <li><a href="https://felidaefund.org/">Felidae Conservation Fund</a> (<a href="https://wildepod.org/">WildePod platform</a>) (<a href="https://abhaykashyap.com/blog/ai-powered-camera-trap-image-annotation-system/">blog post</a>)</li>
  <li><a href="https://www.abmi.ca/home.html">Alberta Biodiversity Monitoring Institute (ABMI)</a> (<a href="https://www.wildtrax.ca/">WildTrax platform</a>) (<a href="https://wildcams.ca/blog/the-abmi-visits-the-zoo/">blog post</a>)</li>
  <li><a href="http://en.shanshui.org/">Shan Shui Conservation Center</a> (<a href="https://mp.weixin.qq.com/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs">blog post</a>) (<a href="https://mp-weixin-qq-com.translate.goog/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs&_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp">translated blog post</a>)</li>
  <li><a href="http://www.irconservancy.org/">Irvine Ranch Conservancy</a> (<a href="https://www.ocregister.com/2022/03/30/ai-software-is-helping-researchers-focus-on-learning-about-ocs-wild-animals/">story</a>)</li>
  <li><a href="https://wildlifeprotectionsolutions.org/">Wildlife Protection Solutions</a> (<a href="https://customers.microsoft.com/en-us/story/1384184517929343083-wildlife-protection-solutions-nonprofit-ai-for-earth">story</a>, <a href="https://www.enterpriseai.news/2023/02/20/ai-helps-wildlife-protection-solutions-safeguard-endangered-species/">story</a>)</li>
  <li><a href="https://roadecology.ucdavis.edu/">Road Ecology Center</a>, University of California, Davis (<a href="https://wildlifeobserver.net/">Wildlife Observer Network platform</a>)</li>
  <li><a href="https://www.nature.org/en-us/about-us/where-we-work/united-states/california/">The Nature Conservancy in California</a> (<a href="https://github.com/tnc-ca-geo/animl-frontend">Animl platform</a>)</li>
  <li><a href="https://science.sandiegozoo.org/">San Diego Zoo Wildlife Alliance</a> (<a href="https://github.com/conservationtechlab/animl">Animl R package</a>)</li>
</ul>

</details><br>


>[!IMPORTANT]
>If you would like to be added to this list or have any questions regarding MegaDetector and Pytorch-Wildlife, please [email us](zhongqimiao@microsoft.com) or join us in our Discord channel: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

