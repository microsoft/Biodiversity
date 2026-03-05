### Pytorch-Wildlife Version 1.2.1

#### SpeciesNet is available in Pytorch-Wildlife for testing! 
- We have added SpeciesNet into our model zoo, which is compatible with all detection models provided by Pytorch-Wildlife. Please refer to [this document](https://github.com/microsoft/CameraTraps/blob/SppNet_TF/PytorchWildlife/models/classification/speciesnet_base/sppnet_readme.md) for more details!

#### Deepfaune in Our Model Zoo!! 
- We are excited to announce the release of the Deepfaune models—both the detector and classifier—in PyTorch-Wildlife, adding to our growing model zoo. A huge thank you to the Deepfaune team for your support! Deepfaune is one of the most comprehensive models focused on the European ecosystem for both detection and classification. It serves as a great complement to MegaDetector, which has primarily been trained on datasets from North America, South America, and Africa. The Deepfaune detector is also our first third-party camera trap detection model integrated into PyTorch-Wildlife!
- To use the model, you just need to load them as any other Pytorch-Wildife models: 
```
detection_model = pw_detection.DeepfauneDetector(device=DEVICE)
classification_model = pw_classification.DeepfauneClassifier(device=DEVICE)
```
- You can also use the `detection_classification_pipeline_demo.py` script in the demo folder to test the whole detection + classification pipeline. 
- Please also take a look at the original [Deepfaune website](https://www.deepfaune.cnrs.fr/en/) and give them a star! 

#### Deepfaune-New-England in Our Model Zoo Too!!
- Besides the original Deepfaune mode, there is another fine-tuned Deepfaune model developed by USGS for the Northeastern NA area called Deepfaune-New-England (DFNE). It can also be loaded with `classification_model = pw_classification.DFNE(device=DEVICE)`
- Please take a look at the orignal [DFNE repo](https://code.usgs.gov/vtcfwru/deepfaune-new-england/-/tree/main?ref_type=heads) and give them a star! 

### Pytorch-Wildlife Version 1.2.0
- In this version of Pytorch-Wildlife, we are happy to release our [detection fine-tuning module](PW_FT_detection), with which users can fine-tune their own detection model from any released pre-trained MegaDetectorV6 models. Besides, this module also has functionalities that help users to prepare their datasets for the fine-tuning, just as our classification fine-tuning modules. For more details, please check the [readme](https://github.com/microsoft/CameraTraps/blob/main/PW_FT_detection). Currently the fine-tuning is based on [Ultralytics](https://www.ultralytics.com/) with AGPL. We will release MIT versions in the future. Here is the [release page](https://github.com/microsoft/CameraTraps/releases).
- We have also released additional MegaDetectorV6 models based on Yolo-v10 and RtDetr. We have skipped Yolo-v11 models because of limited performance and architectural gains. Most of the MIT and Apache versions have also finished training but are waiting for internal review before they can be released.
- We have also updated our AI4G-Amazon model with bigger datasets and it has a better performance compared to previous iterations. Please feel free to test it or fine-tune on it. 
- We will also make a new roadmap for 2025 in the next couple of updates.
- Special thanks to [José Díaz](https://github.com/jdiaz97) for his great cross-platform app, [BoquilaHUB](https://github.com/boquila/boquilahub), that is even working on ios and android! Please check his repo out! In the future, we will create a project gallery showcasing projects that use or are build upon Pytorch-Wildlife. If you want your projects to be included, please feel free to reach out to us on [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

<img src="https://github.com/boquila/boquilahub/blob/main/readme.jpg?raw=true" alt="animal_det_1" width="300"/><br>
