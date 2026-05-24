---
description: "PyTorch-Wildlife license information: MIT for the core package and utilities. Model-specific licenses (AGPL-3.0, CC BY-SA) are listed in the model zoo."
tags:
  - PyTorch-Wildlife license
  - MIT license
  - open source wildlife AI
  - model licensing
---

# License

## :bangbang: Model licensing 

The **PyTorch-Wildlife** package is under MIT, however some of the models in the model zoo are not. For example, MegaDetectorV5, which is trained using the Ultralytics package, a package under AGPL-3.0, and is not for closed-source commercial uses if they are using updated 'ultralytics' packages. 

There may be a confusion because YOLOv5 was initially released before the establishment of the AGPL-3.0 license. According to the official [Ultralytics-Yolov5](https://github.com/ultralytics/yolov5) package, it is under AGPL-3.0 now, and the maintainers have discussed how their licensing policy has evolved over time in their issues section. 

We want to make PyTorch-Wildlife a platform where different models with different licenses can be hosted and want to enable different use cases. To reduce user confusions, in our [model zoo](model_zoo/megadetector.md) section, we list all existing and planned future models in our model zoo, their corresponding license, and release schedules. 

In addition, since the **PyTorch-Wildlife** package is under MIT, all the utility functions, including data pre-/post-processing functions and model fine-tuning functions in this packages are under MIT as well.


```
--8<-- "LICENSE"
```