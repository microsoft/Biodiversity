# Main changes and additions

### V 1.2.4

The inference code for the MIT YOLO and Apache RT‑DETR models is now available! To use either one, just load it like any other PyTorch‑Wildlife model:

```python
from pw_detection import MegaDetectorV6MIT, MegaDetectorV6Apache

# MIT YOLO
detector = MegaDetectorV6MIT(
    device=DEVICE,
    pretrained=True,
    version="MDV6-mit-yolov9-e"
)

# Apache RT‑DETR
detector = MegaDetectorV6Apache(
    device=DEVICE,
    pretrained=True,
    version="MDV6-apa-rtdetr-e"
)
```
Valid versions:
- MDV6-mit-yolov9-c
- MDV6-mit-yolov9-e
- MDV6-apa-rtdetr-c
- MDV6-apa-rtdetr-e

You can also try out the full pipeline using the `detection_classification_pipeline_demo.py` script in the demo folder.


