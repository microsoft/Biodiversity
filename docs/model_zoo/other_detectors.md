---
description: "PyTorch-Wildlife model zoo: other detection models including MegaDetector-Overhead, HerdNet, and Deepfaune for aerial and camera-trap wildlife detection."
tags:
  - MegaDetector-Overhead
  - HerdNet
  - Deepfaune
  - wildlife detection
  - overhead detection
  - aerial wildlife survey
---

# Other detection models

|Models|Version Names|Licence|Release|Reference|
|---|---|---|---|---|
|Deepfaune-detection|-|CC BY-SA 4.0|Released|[Deepfaune](https://www.deepfaune.cnrs.fr/en/)|
|HerdNet-general|general|CC BY-NC-SA-4.0|Released|[Alexandre et. al. 2023](https://github.com/Alexandre-Delplanque/HerdNet)|
|HerdNet-ennedi|ennedi|CC BY-NC-SA-4.0|Released|[Alexandre et. al. 2023](https://github.com/Alexandre-Delplanque/HerdNet)|
|MegaDetector-Overhead-T (Transformer)|T|MIT|Released|[MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead)|
|MegaDetector-Overhead-C (CNN)|C|MIT|Released|[MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead)|

>[!TIP]
>Some models, such as MegaDetectorV6, HerdNet, and AI4G-Amazon, have different versions, and they are loaded by their corresponding version names. Here is an example: `detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")`.
