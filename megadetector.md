# MegaDetector

> **MegaDetector** is Microsoft's open-source AI model for camera-trap conservation — it detects animals, people, and vehicles in camera-trap imagery and filters out blank images, reducing manual review across large datasets. Developed and maintained by the **Microsoft AI for Good Lab**.

> [!IMPORTANT]
> **This page has moved. MegaDetector's canonical home is now [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector).**
> Source code, releases, issues, the model zoo, the full version history (V1–V6), benchmarks, and the list of organizations using MegaDetector all live there. This page remains only as a pointer to the canonical repository.

## Where things live

| You want… | Go to |
|---|---|
| MegaDetector source, releases, issues, V1–V6 history, benchmarks | **[microsoft/MegaDetector](https://github.com/microsoft/MegaDetector)** |
| The PyTorch-Wildlife framework that hosts MegaDetector and classifiers | [microsoft/Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife) |
| The biodiversity ecosystem umbrella (all repos, one hub) | [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) |
| The AI-enabled edge device that runs MegaDetector in the field | [microsoft/SPARROW](https://github.com/microsoft/SPARROW) |

## Run MegaDetectorV6 quickly

The current generation is **MegaDetectorV6** (smaller, faster architectures including YOLOv9, YOLOv10, and RT-DETR; the compact variant is roughly 2% of the V5 parameter count at comparable accuracy). Try it without writing code via the [Hugging Face Space](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife), or load it through PyTorch-Wildlife:

```python
from PytorchWildlife.models import detection as pw_detection
detection_model = pw_detection.MegaDetectorV6()
```

Full usage, model selection, and the fine-tuning pipeline are documented in **[microsoft/MegaDetector](https://github.com/microsoft/MegaDetector)**.

## MegaDetectorV5 and earlier

For `MegaDetectorV5` weights and the original repository — primarily developed by Dan Morris during his time at Microsoft — see the [archive branch](https://github.com/microsoft/Biodiversity/tree/archive) of this repository (formerly `microsoft/CameraTraps`). Dan continues to actively maintain a community fork at [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector), which remains a valuable resource for the community.

## Citing MegaDetector

A `citation.cff` is maintained on [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) for automated citation tools. Cite **Beery, Morris, Yang 2019 — *Efficient Pipeline for Camera Trap Image Review*** for MegaDetector specifically, and **Hernandez et al. 2024 — *Pytorch-Wildlife*** for the framework.

## Contact

Questions about MegaDetector or PyTorch-Wildlife: open an issue on [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector/issues) or join the [Discord](https://discord.gg/TeEVxzaYtm).
