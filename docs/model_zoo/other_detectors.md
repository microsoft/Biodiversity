# Other detection models 

|Models|Version Names|Licence|Release|Reference|
|---|---|---|---|---|
|Deepfaune-detection|-|CC BY-SA 4.0|Released|[Deepfaune](https://www.deepfaune.cnrs.fr/en/)|
|HerdNet-general|general|CC BY-NC-SA-4.0|Released|[Alexandre et. al. 2023](https://github.com/Alexandre-Delplanque/HerdNet)|
|HerdNet-ennedi|ennedi|CC BY-NC-SA-4.0|Released|[Alexandre et. al. 2023](https://github.com/Alexandre-Delplanque/HerdNet)|
|OWL-T (Overhead Wildlife Locator, Transformer)|T|MIT|Released|[OWL demo](https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_owl.ipynb)|
|OWL-C (Overhead Wildlife Locator, CNN)|C|MIT|Released|[OWL demo](https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_owl.ipynb)|

>[!TIP]
>Some models, such as MegaDetectorV6, HerdNet, and AI4G-Amazon, have different versions, and they are loaded by their corresponding version names. Here is an example: `detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")`.