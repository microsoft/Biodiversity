# MegaDetector

MegaDetector is an open-source AI model developed by the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/ai/ai-for-good) that automatically detects **animals**, **people**, and **vehicles** in camera trap images. It is used by more than 80 conservation organizations worldwide. MegaDetector does not identify species — it locates animals in images so researchers can spend less time reviewing empty frames and more time doing science.

MegaDetector is part of [PyTorch Wildlife](https://microsoft.github.io/CameraTraps/), a collaborative deep learning framework for conservation, and the [SPARROW Studio](https://github.com/microsoft/CameraTraps) ecosystem. It is free, open-source, and available under a permissive license.

---

## Table of Contents

- [What is MegaDetector?](#what-is-megadetector)
- [Getting Started](#getting-started)
- [MegaDetector V6](#megadetector-v6)
- [Organizations Using MegaDetector](#organizations-using-megadetector)
- [Performance and Speed](#performance-and-speed)
- [Desktop and Web Interfaces](#desktop-and-web-interfaces)
- [Working with Detection Output](#working-with-detection-output)
- [Species Classification](#species-classification)
- [GPU Setup](#gpu-setup)
- [Processing at Scale](#processing-at-scale)
- [Maximizing Accuracy](#maximizing-accuracy)
- [Known Limitations](#known-limitations)
- [Version History](#version-history)
- [Previous Versions and Archive](#previous-versions-and-archive)
- [Citing MegaDetector](#citing-megadetector)
- [Accuracy Evaluation](#accuracy-evaluation)
- [Training Data](#training-data)
- [Contact](#contact)

---

## What is MegaDetector?

Camera traps are a critical tool in conservation biology and wildlife ecology. Researchers deploy thousands of motion-triggered cameras in forests, grasslands, and other habitats to passively monitor wildlife populations. The result is millions of images — and the vast majority are empty frames triggered by wind, vegetation, or temperature changes. Manually reviewing these images is one of the biggest bottlenecks in camera trap research.

MegaDetector solves this problem. It is a deep-learning object detection model trained on **several million camera trap images** from ecosystems around the world. When you run MegaDetector on your camera trap images, it draws bounding boxes around three categories of objects:

| Class | Label | What it detects |
| --- | --- | --- |
| 0 | **animal** | Any animal — mammals, birds, reptiles, insects |
| 1 | **person** | Humans |
| 2 | **vehicle** | Cars, trucks, ATVs, boats |

Each detection comes with a **confidence score** between 0 and 1. You set a threshold (commonly 0.15–0.3) and anything above that threshold is flagged as a detection. The output is typically a JSON file listing every detection in every image, which you can then use to sort images, filter blanks, or feed into downstream species classifiers.

MegaDetector is intentionally a **detector**, not a **classifier**. It tells you *where* the animals are, not *what species* they are. This is a deliberate design choice: a general-purpose animal detector trained on data from hundreds of ecosystems generalizes far better than any single species classifier. You can then pair MegaDetector with a species classifier trained for your specific region — see [Species Classification](#species-classification).

### What makes MegaDetector different?

- **Generalization**: Trained on images from hundreds of camera trap deployments across every continent. It works on your data even if it has never seen your specific camera setup, habitat, or species. This breadth of training data is MegaDetector's core advantage — most custom-trained detectors overfit to one ecosystem, one camera brand, or one set of lighting conditions. MegaDetector is designed to be a reliable first pass across any deployment.
- **Microsoft-backed**: Developed and maintained by the Microsoft AI for Good Lab, with long-term institutional support. MegaDetector has been in continuous development since 2018, with regular version updates, bug fixes, and community support.
- **Community-tested**: Used by 80+ organizations (see the [full list](#organizations-using-megadetector)) across government agencies, universities, conservation NGOs, and private companies. This level of real-world validation across diverse ecosystems and workflows is unmatched by any other camera trap AI model.
- **Free and open-source**: Available under permissive licenses (MIT and Apache 2.0 options for V6). There is no cost, no usage limit, and no requirement to share your data.
- **Easy to use**: Install with `pip install PytorchWildlife` and run detection in three lines of Python. No machine learning expertise required. If you can run a Python script, you can run MegaDetector.
- **Part of a larger ecosystem**: MegaDetector is one model in the [PyTorch Wildlife](https://microsoft.github.io/CameraTraps/) framework, which also includes species classifiers, bioacoustics models, and the [SPARROW Studio](https://github.com/microsoft/CameraTraps) desktop application. You can start with MegaDetector for detection and expand into species classification, audio monitoring, or aerial imagery analysis as your needs grow.


## Getting Started

### Prerequisites

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **pip** (comes with Python)
- A computer with a reasonably modern CPU. A GPU is not required but will dramatically speed up processing — see [GPU Setup](#gpu-setup).

### Installation

```bash
pip install PytorchWildlife
```

This installs MegaDetector along with the full PyTorch Wildlife framework. Model weights are downloaded automatically the first time you load a model.

For conda users:

```bash
conda create -n megadetector python=3.10 -y
conda activate megadetector
pip install PytorchWildlife
```

For detailed environment setup, including GPU configuration and platform-specific notes, see the [installation guide](https://microsoft.github.io/CameraTraps/installation/).

### Running MegaDetector on a single image

```python
from PytorchWildlife.models import detection as pw_detection

# Load MegaDetector V6 (weights download automatically on first run)
model = pw_detection.MegaDetectorV6()

# Run detection on a single image
results = model.single_image_detection("path/to/your/image.jpg")

# results contains bounding boxes, class labels, and confidence scores
print(results)
```

### Running MegaDetector on a folder of images

```python
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()

# Process an entire folder
results = model.batch_image_detection("path/to/your/image_folder/")
```

### Running MegaDetector on video

```python
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()

# Process a video file
results = model.single_video_detection("path/to/your/video.mp4")
```

### Choosing a model variant

MegaDetector V6 comes in multiple variants optimized for different use cases. The default (`MegaDetectorV6()`) loads the Ultralytics YOLOv9 compact model. To load a specific variant:

```python
# Load the extra-large YOLOv10 variant for maximum accuracy
model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")

# Load the MIT-licensed compact variant
model = pw_detection.MegaDetectorV6(version="MDV6-mit-yolov9-c")

# Load the Apache-licensed RT-DETR extra variant (best overall accuracy)
model = pw_detection.MegaDetectorV6(version="MDV6-apa-rtdetr-e")
```

See the [full model table](#megadetector-v6-model-variants) for all variants, their sizes, accuracy, and licenses.

### Interactive demos

If you want to try MegaDetector without writing code:

- **Hugging Face Space**: [ai-for-good-lab/pytorch-wildlife](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) — upload images and see results in your browser
- **Google Colab notebook**: [Open in Colab](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing) — run MegaDetector in a free cloud GPU environment
- **SPARROW Studio**: A full desktop application with a graphical interface — see [Desktop and Web Interfaces](#desktop-and-web-interfaces)


## MegaDetector V6

MegaDetector V6 is a major generational update focused on **computational efficiency**, **modern architectures**, and **licensing flexibility**. We trained multiple new models using YOLOv9, YOLOv10, and RT-DETR architectures, optimized for both high-end servers and low-budget edge devices.

### What's new in V6

- **Dramatically smaller models**: The compact YOLOv10 variant (MDV6-yolov10-c) has only **2.3 million parameters** — just 2% of MegaDetector V5's 139.9 million — while maintaining comparable detection accuracy.
- **Multiple architecture choices**: Choose the variant that best fits your hardware, accuracy requirements, and licensing constraints.
- **Permissive licensing options**: V6 introduces MIT-licensed and Apache-2.0-licensed variants. V5 was AGPL-3.0 only.
- **Modern architectures**: YOLOv9 and YOLOv10 bring architectural improvements in efficiency and accuracy. RT-DETR (Real-Time Detection Transformer) offers a transformer-based option.
- **No NMS required for YOLOv10 variants**: YOLOv10 models use a one-to-one assignment strategy, eliminating the need for non-maximum suppression post-processing.

### MegaDetector V6 model variants

#### Ultralytics variants (AGPL-3.0 license)

| Model | Architecture | Parameters | mAR (Animal Recall) | mAP50 | Image Size |
| --- | --- | --- | --- | --- | --- |
| MDV6-yolov9-c | YOLOv9 Compact | 25.5M | 78.4% | 87.9% | 1280px |
| MDV6-yolov9-e | YOLOv9 Extra | 58.1M | 82.1% | 88.6% | 1280px |
| MDV6-yolov10-c | YOLOv10 Compact | 2.3M | 76.8% | 87.2% | 1280px |
| MDV6-yolov10-e | YOLOv10 Extra | 29.5M | 82.8% | 92.8% | 1280px |
| MDV6-rtdetr-c | RT-DETR Compact | 31.9M | 81.6% | 89.9% | 1280px |

#### MIT-licensed variants

| Model | Architecture | Parameters | mAR (Animal Recall) | mAP50 | Image Size |
| --- | --- | --- | --- | --- | --- |
| MDV6-mit-yolov9-c | YOLOv9 Compact | 9.7M | 74.8% | 87.6% | 640px |
| MDV6-mit-yolov9-e | YOLOv9 Extra | 51M | 76.1% | 71.5% | 640px |

#### Apache-2.0-licensed variants

| Model | Architecture | Parameters | mAR (Animal Recall) | mAP50 | Image Size |
| --- | --- | --- | --- | --- | --- |
| MDV6-apa-rtdetr-c | RT-DETR Compact | 20M | 81.1% | 91.0% | 640px |
| MDV6-apa-rtdetr-e | RT-DETR Extra | 76M | 82.9% | 94.1% | 640px |

> [!TIP]
> **Which variant should I use?**
> - **Best overall accuracy**: MDV6-apa-rtdetr-e (82.9% mAR, 94.1% mAP50, Apache-2.0 license)
> - **Best compact model**: MDV6-yolov10-c (2.3M params — runs well on laptops and edge devices)
> - **Best balance of speed and accuracy**: MDV6-yolov10-e (29.5M params, 82.8% mAR)
> - **Need MIT license?** MDV6-mit-yolov9-c (9.7M params, 74.8% mAR)
> - **Need Apache license with high accuracy?** MDV6-apa-rtdetr-e

### Understanding the metrics

We emphasize **Animal Recall (mAR)** as the primary performance metric rather than mAP alone. In camera trap workflows, the cost of a **false negative** (missing an animal) is much higher than the cost of a **false positive** (flagging a blank image as containing an animal). A missed animal is gone from your dataset forever; a false positive just means a human reviewer looks at one extra image. Optimizing for recall ensures the fewest possible animals are missed, even if it means slightly more false positives to review.

### Architecture

MegaDetector V6 models detect three classes — animal, person, and vehicle — using one-stage object detection architectures:

- **YOLOv9 / YOLOv10**: The latest in the YOLO family. YOLOv10 introduces one-to-one head assignment, removing the need for non-maximum suppression. This simplifies post-processing and reduces latency.
- **RT-DETR (Real-Time Detection Transformer)**: A hybrid CNN-transformer architecture that brings transformer-based detection to real-time speeds. The Apache-licensed RT-DETR extra variant achieves the highest accuracy of any MegaDetector model to date.

All V6 models are trained on the same multi-million-image camera trap dataset used for previous MegaDetector versions, augmented with additional training data collected since V5's release.


## Organizations Using MegaDetector

MegaDetector is used by conservation organizations, government agencies, universities, and technology companies around the world. We're listing organizations who have given us permission to reference them here or have posted publicly about their use of MegaDetector.

**Government Agencies and National Parks**
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

**Universities and Research Labs**
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

**Museums, Zoos, and Research Institutions**
- [Center for Biodiversity and Conservation, American Museum of Natural History](https://www.amnh.org/research/center-for-biodiversity-conservation)
- [Northern Great Plains Program, Smithsonian](https://nationalzoo.si.edu/news/restoring-americas-prairie)
- [Snapshot USA, Smithsonian](https://emammal.si.edu/snapshot-usa)
- [Seattle Urban Carnivore Project, Woodland Park Zoo](https://www.zoo.org/seattlecarnivores)
- [San Diego Zoo Wildlife Alliance](https://science.sandiegozoo.org/) ([Animl R package](https://github.com/conservationtechlab/animl))
- [Taronga Conservation Society](https://taronga.org.au/)
- [Hamaarag, The Steinhardt Museum of Natural History, Tel Aviv University](https://hamaarag.org.il/)

**Platforms and Technology Partners**
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

If your organization uses MegaDetector and you'd like to be listed here, please [email us](mailto:zhongqimiao@microsoft.com) or reach out on [Discord](https://discord.gg/TeEVxzaYtm).


## Performance and Speed

Processing speed depends on your hardware, the model variant you choose, and image resolution. Here are rough benchmarks to set expectations:

| Hardware | Model Variant | Approximate Speed |
| --- | --- | --- |
| Modern NVIDIA GPU (e.g., RTX 3090) | MDV6-yolov10-c (2.3M params) | ~100-200 images/sec |
| Modern NVIDIA GPU (e.g., RTX 3090) | MDV6-yolov10-e (29.5M params) | ~30-60 images/sec |
| Modern NVIDIA GPU (e.g., RTX 3090) | MDV6-apa-rtdetr-e (76M params) | ~15-30 images/sec |
| Modern CPU (no GPU) | MDV6-yolov10-c (2.3M params) | ~2-5 images/sec |
| Modern CPU (no GPU) | MDV6-yolov10-e (29.5M params) | ~0.5-2 images/sec |
| Google Colab (free tier GPU) | Any V6 variant | ~10-50 images/sec |

These are approximate ranges. Actual throughput depends on image resolution, batch size, CPU/GPU model, available memory, and I/O speed. JPEG decoding and disk I/O can become bottlenecks at high GPU throughput — if you find the GPU is waiting for data, try loading images from an SSD rather than a network drive or spinning disk.

**For reference**: A typical camera trap study generates 100,000–1,000,000 images. At 50 images/sec on a GPU, one million images takes about 5.5 hours. On a CPU with the compact model at 3 images/sec, one million images takes about 3.9 days.

**A note on MegaDetector V5 vs. V6 speed**: MegaDetector V5 uses a YOLOv5 architecture with 139.9M parameters. Even the largest V6 variant (MDV6-apa-rtdetr-e at 76M params) is roughly half the size of V5. The compact V6 variants are 2-60x smaller than V5 and correspondingly faster. If you were able to run V5 on your hardware, every V6 variant will be faster.

If you don't have a GPU, the compact models (MDV6-yolov10-c, MDV6-mit-yolov9-c) are your best bet — they're small enough to run on a laptop CPU at usable speeds. For large-scale processing, see [Processing at Scale](#processing-at-scale).


## Desktop and Web Interfaces

Several graphical interfaces are available for running MegaDetector without writing code:

### SPARROW Studio (by Microsoft AI for Good)

[SPARROW Studio](https://github.com/microsoft/CameraTraps) is a unified desktop application built on top of PyTorch Wildlife that includes:

- AI inference using the full model zoo (MegaDetector, species classifiers, bioacoustics models)
- Local and cloud-based data management
- Post-inference statistics and analysis
- Bounding-box and category annotation editing
- Embedding visualization

The Windows installer is available on [Zenodo](https://zenodo.org/records/19687738/files/SPARROW%20Studio%20Installer.msi?download=1). Mac and Linux builds are in progress.

### AddaxAI (formerly EcoAssist)

[AddaxAI](https://addaxdatascience.com/addaxai/) is a third-party desktop application that provides a user-friendly interface for running MegaDetector and other models. It supports Windows, macOS, and Linux, and includes features for batch processing, annotation, and results visualization.

### Hugging Face Space

Our [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) lets you upload images and run MegaDetector directly in your browser. No installation required. Best for quick tests on a handful of images.

### Timelapse

[Timelapse](http://saul.cpsc.ucalgary.ca/timelapse/) is a free image analysis tool by Saul Greenberg at the University of Calgary. It integrates with MegaDetector output files and provides a rich interface for reviewing camera trap images with detection data overlaid.


## Working with Detection Output

MegaDetector outputs detections as structured data containing bounding boxes, class labels, and confidence scores. Here's how to work with the results in common workflows:

### Programmatic access

```python
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()
results = model.single_image_detection("image.jpg")

# results["detections"] contains a supervision.Detections object with:
#   .xyxy — bounding box coordinates (Nx4 array)
#   .confidence — confidence scores (N array)
#   .class_id — class IDs: 0=animal, 1=person, 2=vehicle (N array)
```

### Saving annotated images

```python
import supervision as sv
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()
results = model.single_image_detection("image.jpg")

# Draw bounding boxes on the image and save
annotated = sv.BoxAnnotator().annotate(
    scene=results["img"].copy(),
    detections=results["detections"]
)
sv.plot_image(annotated)
```

### Filtering by confidence

Not all detections are real. Use a confidence threshold to filter out low-confidence detections:

```python
# Keep only detections with confidence >= 0.2
confident = results["detections"][results["detections"].confidence >= 0.2]
```

Common threshold values:
- **0.15**: Aggressive — catches almost everything but more false positives
- **0.2**: Balanced — good default for most use cases
- **0.3**: Conservative — fewer false positives but may miss some animals
- **0.5+**: Very conservative — use only when false positives are costly

### Sorting images

A common workflow is to separate images into "animal" and "blank" folders:

```python
import shutil
from pathlib import Path
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()
threshold = 0.2

for img_path in Path("my_images/").glob("*.jpg"):
    results = model.single_image_detection(str(img_path))
    has_animal = any(
        conf >= threshold
        for conf, cls in zip(
            results["detections"].confidence,
            results["detections"].class_id
        )
        if cls == 0  # class 0 = animal
    )
    dest = Path("animals" if has_animal else "blanks")
    dest.mkdir(exist_ok=True)
    shutil.copy2(img_path, dest / img_path.name)
```

### JSON output for integration with other tools

MegaDetector results can be exported as JSON files compatible with tools like Timelapse and other camera trap management platforms. See the [batch processing demos](https://github.com/microsoft/CameraTraps/tree/main/demo) for examples.


## Species Classification

MegaDetector finds animals — it does not identify species. This is by design. A single animal detector generalizes across ecosystems far better than any species classifier, because "animal vs. background" is a much more universal visual concept than distinguishing between hundreds of species that vary by continent.

For species identification, you run a two-stage pipeline:

1. **MegaDetector** detects and crops animals from images
2. **A species classifier** identifies the species in each crop

PyTorch Wildlife includes several species classifiers in its [model zoo](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/):

| Classifier | Region / Scope | Classes | License |
| --- | --- | --- | --- |
| AI4G Amazon Rainforest | Amazon basin | 36 species | MIT |
| AI4G Snapshot Serengeti | East Africa | 10 species | MIT |
| AI4G Opossum | Galapagos Islands | 2 (opossum / not) | MIT |
| DeepFaune Classifier | Europe | 34 species (multilingual) | CC BY-SA 4.0 |
| DFNE (Deep Fauna New England) | Northeastern U.S. | 23 species | CC0 1.0 |

### Running detection + classification together

```python
import supervision as sv
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

# Load both models
detector = pw_detection.MegaDetectorV6()
classifier = pw_classification.AI4GAmazonRainforest()

# Detect animals
det_results = detector.single_image_detection("image.jpg")

# Classify each detected animal
for xyxy in det_results["detections"].xyxy:
    cropped = sv.crop_image(image=det_results["img"], xyxy=xyxy)
    cls_result = classifier.single_image_classification(cropped)
    print(f"Species: {cls_result['prediction']}, Confidence: {cls_result['confidence']:.2f}")
```

If none of the existing classifiers cover your region, you can fine-tune a custom classifier on your own labeled data using PyTorch Wildlife's [fine-tuning tools](https://microsoft.github.io/CameraTraps/).

Additionally, Google's [SpeciesNet](https://github.com/google/cameratrapai) is a species classifier designed to work with MegaDetector outputs and covers a broad range of species globally.


## GPU Setup

MegaDetector runs on CPUs, but a GPU will speed things up 10-50x depending on the model variant and batch size.

### Requirements

- An NVIDIA GPU with CUDA support
- CUDA Toolkit (12.1 recommended; 11.3+ supported)
- PyTorch installed with CUDA support

### Setting up CUDA with PyTorch

If you installed PyTorch Wildlife with `pip install PytorchWildlife`, PyTorch may have installed without GPU support. To install PyTorch with CUDA:

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verifying GPU access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Using a specific GPU

```python
from PytorchWildlife.models import detection as pw_detection

# Use GPU 0 (default)
model = pw_detection.MegaDetectorV6(device="cuda:0")

# Use GPU 1 on a multi-GPU system
model = pw_detection.MegaDetectorV6(device="cuda:1")

# Force CPU even if GPU is available
model = pw_detection.MegaDetectorV6(device="cpu")
```

### Apple Silicon (M1/M2/M3)

On macOS with Apple Silicon, PyTorch supports the MPS (Metal Performance Shaders) backend:

```python
model = pw_detection.MegaDetectorV6(device="mps")
```

MPS support is generally functional but less mature than CUDA. If you encounter issues, fall back to CPU.


## Processing at Scale

If you're processing hundreds of thousands or millions of images, here are strategies to maximize throughput:

### Choose the right model

For batch processing at scale, the compact models (MDV6-yolov10-c at 2.3M params, or MDV6-mit-yolov9-c at 9.7M params) offer the best throughput-per-watt. The accuracy difference vs. the extra-large models is typically 3-5 percentage points on recall — often acceptable for a 5-10x speed improvement.

### Use batch processing

Always use `batch_image_detection()` instead of calling `single_image_detection()` in a loop. Batch processing leverages GPU parallelism:

```python
model = pw_detection.MegaDetectorV6(device="cuda:0")
results = model.batch_image_detection("path/to/folder/", batch_size=16)
```

Increase `batch_size` until GPU memory is fully utilized. Start with 16 and increase until you see out-of-memory errors, then back off.

### Use multiple GPUs

If you have access to multiple GPUs, run separate processes on each GPU with different subsets of your images.

### Use Google Colab for free GPU access

If you don't have local GPU access, Google Colab provides free (limited) GPU instances. Our [Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing) is pre-configured for MegaDetector.

### Save results incrementally

For very large jobs, save results periodically rather than keeping everything in memory:

```python
import json
from pathlib import Path
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6(device="cuda:0")
image_paths = list(Path("images/").glob("**/*.jpg"))

# Process in chunks and save periodically
chunk_size = 1000
for i in range(0, len(image_paths), chunk_size):
    chunk = image_paths[i:i+chunk_size]
    results = model.batch_image_detection([str(p) for p in chunk])
    with open(f"results_chunk_{i}.json", "w") as f:
        json.dump(results, f)
    print(f"Processed {min(i + chunk_size, len(image_paths))} / {len(image_paths)}")
```

### Cloud processing

For truly massive datasets (tens of millions of images), consider using cloud GPU instances (AWS, Azure, GCP). The compact models are especially cost-effective in cloud environments due to their small memory footprint.


## Maximizing Accuracy

### Use the right confidence threshold

The default threshold matters more than the model variant for most users. Start with 0.2, review a sample of your results, and adjust:

- If you're seeing too many missed animals: lower the threshold to 0.1-0.15
- If you're seeing too many false positives: raise it to 0.3-0.5
- For presence/absence surveys where every detection matters: use 0.1

### Use the extra-large models for critical work

If accuracy is more important than speed (e.g., endangered species surveys where missing a detection is costly), use MDV6-apa-rtdetr-e or MDV6-yolov10-e.

### Understand camera-specific quirks

MegaDetector is trained on data from hundreds of camera trap models, but some setups produce more false positives than others:

- **Infrared images**: Generally work well. Color and grayscale IR images are both supported.
- **Time-lapse mode**: Works, but images without motion triggers tend to have lower animal prevalence, so you may see more false positives proportionally.
- **Very close range**: Animals extremely close to the camera (filling most of the frame) may occasionally be missed. This is uncommon.
- **Extreme weather**: Heavy rain, snow, or fog can trigger false detections.

### Review a sample before bulk processing

Before processing your entire dataset, run MegaDetector on a representative sample (500-1,000 images) and review the results to calibrate your expectations and tune your threshold.


## Known Limitations

No model is perfect. Here are MegaDetector's known limitations:

### Things MegaDetector does not do

- **Species identification**: MegaDetector detects animals, people, and vehicles. It does not identify species. Use a species classifier as a second step (see [Species Classification](#species-classification)).
- **Counting**: MegaDetector draws individual bounding boxes, but it doesn't count animals. Overlapping animals or herds may produce merged or missed boxes.
- **Behavior classification**: MegaDetector does not classify animal behavior (feeding, resting, moving, etc.).

### Scenarios where accuracy degrades

- **Very small animals at long range**: Tiny animals (small birds, insects, mice at a distance) may fall below the detection threshold.
- **Highly camouflaged animals**: Animals that blend extremely well with their background (e.g., a brown deer in brown leaf litter) are harder to detect. MegaDetector still catches most of these, but accuracy is lower than for high-contrast subjects.
- **Dense vegetation occlusion**: Animals partially hidden behind dense vegetation may not be detected if too little of the animal is visible.
- **Non-standard camera trap images**: MegaDetector is trained on camera trap images. It will work on other image types (phone photos, DSLR images, drone imagery) but with reduced accuracy. For drone/aerial imagery, consider [OWL (Overhead Wildlife Locator)](https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_owl.ipynb) instead.
- **Underwater images**: MegaDetector is trained on terrestrial camera trap images. It is not designed for underwater use.

### False positive patterns

Common sources of false positives:

- Moving vegetation (swaying branches, waving grass)
- Lens flare or reflections
- Shadows, especially at dawn/dusk
- Water surfaces with reflections
- Snow or rain

These can typically be managed by adjusting the confidence threshold upward. In our experience, a threshold of 0.2-0.3 handles most of these cases well.


## Version History

### V6.0 (current — 2024)

The latest release, with multiple model variants across three license types (AGPL-3.0, MIT, Apache-2.0). Uses YOLOv9, YOLOv10, and RT-DETR architectures. See [MegaDetector V6](#megadetector-v6) for full details and benchmarks.

```python
from PytorchWildlife.models import detection as pw_detection
model = pw_detection.MegaDetectorV6()
```

### V5.0 (2022)

The previous major release, based on YOLOv5. Two sub-versions:

| Model | Parameters | mAR | mAP50 | License |
| --- | --- | --- | --- | --- |
| MegaDetector V5a | 139.9M | 81.7% | 92.0% | AGPL-3.0 |
| MegaDetector V5b | 139.9M | 80.9% | 90.1% | AGPL-3.0 |

V5 remains available in PyTorch Wildlife:

```python
from PytorchWildlife.models import detection as pw_detection
model = pw_detection.MegaDetectorV5(version="a")
```

### V4.1 (2020)

Based on Faster R-CNN with an InceptionResNetV2 backbone, trained in TensorFlow. V4.1 added a "vehicle" class. This version is no longer maintained but model weights are available in the [archive branch](https://github.com/microsoft/CameraTraps/tree/archive).

### Earlier versions (V1-V3)

MegaDetector V1 through V3 used various architectures including Faster R-CNN and SSD. These versions are superseded and not recommended for new projects. Historical model weights can be found in the [archive branch](https://github.com/microsoft/CameraTraps/tree/archive).


## Previous versions and archive

Looking for MegaDetector V5 or older? The previous MegaDetector codebase (V5 and earlier), which was primarily developed by Dan Morris during his time at Microsoft, is available on the [archive branch](https://github.com/microsoft/CameraTraps/tree/archive) of this repository.

We recommend migrating to MegaDetector V6, which offers:
- Comparable or better accuracy
- Dramatically smaller and faster models
- Permissive license options (MIT, Apache-2.0)
- Active maintenance and development
- Integration with the full PyTorch Wildlife model zoo and SPARROW Studio


## Citing MegaDetector

If you use MegaDetector in your research, please cite both papers:

### PyTorch Wildlife (the framework MegaDetector runs on)

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

Paper: [arxiv.org/abs/2405.12930](https://arxiv.org/abs/2405.12930) (Oral presentation at CV4Animals workshop, CVPR 2024)

### MegaDetector (the original model)

```bibtex
@misc{beery2019efficient,
      title={Efficient Pipeline for Camera Trap Image Review},
      author={Sara Beery and Dan Morris and Siyu Yang},
      year={2019},
      eprint={1907.06772},
      archivePrefix={arXiv},
}
```

Paper: [arxiv.org/abs/1907.06772](https://arxiv.org/abs/1907.06772)

You can also use GitHub's built-in "Cite this repository" button (in the repo sidebar) to get a citation from our [CITATION.cff](CITATION.cff) file.


## Accuracy Evaluation

MegaDetector is evaluated on a held-out validation set of camera trap images from diverse ecosystems. The primary metric is **Animal Recall (mAR)** — the percentage of actual animals that MegaDetector successfully detects. We also report mAP50 (mean average precision at IoU threshold 0.5).

The best V6 variant (MDV6-apa-rtdetr-e) achieves **82.9% mAR** and **94.1% mAP50** on our validation set. The best V5 variant achieves 81.7% mAR and 92.0% mAP50, so V6 matches or exceeds V5 accuracy with dramatically fewer parameters.

We emphasize recall over precision because in camera trap workflows, false negatives (missed animals) are more costly than false positives (extra images flagged for review). A false positive costs a reviewer a few seconds; a false negative loses data permanently.

Performance on your specific data may vary depending on camera model, habitat, species composition, and image quality. We strongly recommend running MegaDetector on a representative sample of your data and evaluating the results before committing to full-scale processing.

Several external studies have independently evaluated MegaDetector. Organizations including the Australian Wildlife Conservancy, the Smithsonian, and multiple university research labs have published reports confirming MegaDetector's effectiveness on their specific datasets. See the [Organizations Using MegaDetector](#organizations-using-megadetector) section for links to published evaluations and blog posts.

The full benchmark tables for all model variants are in the [MegaDetector V6 model variants](#megadetector-v6-model-variants) section above, and additional details are available in the [model zoo documentation](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/).

### How does MegaDetector compare across ecosystems?

MegaDetector's training data spans tropical forests, temperate woodlands, grasslands, deserts, alpine environments, and urban/suburban habitats across all continents except Antarctica. While performance is generally consistent, there are some patterns:

- **Temperate and savanna ecosystems**: Typically the highest accuracy, as these habitats are well-represented in the training data.
- **Tropical forests**: Good accuracy, though dense vegetation and high species diversity can reduce recall slightly.
- **Arctic and alpine environments**: Accuracy is comparable to other ecosystems, though snow cover can increase false positive rates.
- **Urban/suburban environments**: Works well for common urban wildlife (deer, raccoons, coyotes), though human detection rates are naturally higher in these settings.

Regardless of your ecosystem, we recommend evaluating on a sample of your own data before processing at scale.


## Training Data

MegaDetector is trained on several million camera trap images from a wide variety of ecosystems. The training data includes images contributed by dozens of partner organizations. Due to data sharing agreements with these partners, we cannot release the full training dataset.

However, many large camera trap datasets are publicly available and are excellent resources for training, evaluation, and research:

- [Lila.science](https://lila.science/) — the Labeled Information Library of Alexandria, a repository of large labeled camera trap datasets. This is the best single resource for public camera trap data.
- [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti/) — millions of annotated images from the Serengeti
- [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/) — a benchmark dataset for camera trap research

If you have a large camera trap dataset and are interested in contributing to MegaDetector's training data, please [contact us](mailto:zhongqimiao@microsoft.com). Contributing training data from underrepresented ecosystems or camera setups directly improves MegaDetector's generalization for the entire community.

### A note on data and bias

Like all machine learning models, MegaDetector's performance reflects its training data. Ecosystems and camera setups that are well-represented in the training data tend to have higher detection accuracy. If you work in an underrepresented region or with unusual camera configurations and notice lower-than-expected accuracy, please reach out — we are actively working to broaden MegaDetector's training data to cover more of the world's ecosystems.


## Contact

- **Email**: [zhongqimiao@microsoft.com](mailto:zhongqimiao@microsoft.com)
- **Discord**: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)
- **GitHub Issues**: [microsoft/CameraTraps/issues](https://github.com/microsoft/CameraTraps/issues)
- **GitHub Discussions**: [microsoft/CameraTraps/discussions](https://github.com/microsoft/CameraTraps/discussions)
