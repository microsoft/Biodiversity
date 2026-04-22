# PW-Engine Overview

*A model-agnostic inference core for the PyTorch-Wildlife model zoo.*

**Status:** Preview. Inference surfaces are feature-complete; a data-management layer is the next milestone. Released under the MIT license.

**In one sentence:** PW-Engine (full name: PyTorch-Wildlife Engine) is a Rust-based inference engine and HTTP service that runs the PyTorch-Wildlife model set, powers Sparrow Studio, and can be embedded as the backend of any inference-heavy application.

---

## Why

The Python PyTorch-Wildlife stack is well-suited for research and fine-tuning but has friction as a deployment target: multi-second cold starts, multi-GB Docker images, single-process concurrency limits, and no native-binding story for desktop applications. PW-Engine addresses the deployment shape, not raw inference speed.

| Prior deployment shape | Cause | PW-Engine response |
|---|---|---|
| Multi-second cold start | Python interpreter + inference-server initialization | Sub-second cold start on GPU |
| Multi-GB Docker images | Multi-process Python stack + CUDA bloat | CPU image ~163 MB; GPU image ~4 GB |
| GIL-bound concurrency | Single-process Python worker | Async HTTP server (axum/tokio); per-model serialization, multi-model concurrent |
| Adding a model needs code changes | Hardcoded model adapters | Drop an ONNX file + a manifest entry into the model directory |
| No native desktop binding | No FFI in the Python stack | C header generated from Rust; consumed by Sparrow Studio Local via C# P/Invoke |

**Honest caveat:** the rewrite does not target faster raw inference. Both stacks call the same ONNX Runtime C library. The measurable gains are cold-start, memory footprint, deployment size, and preprocessing throughput.

---

## What

PW-Engine is a Rust core library with four consumption surfaces:

```
                    +---------------------------+
                    |        PW-Engine          |
                    |    (Rust core library)    |
                    +---+--------+-------+------+
                        |        |       |
             +----------+        |       +----------+
             |                   |                  |
      +------+------+     +------+------+    +------+------+
      |  HTTP       |     |   CLI       |    |  Python     |
      |  server     |     |  (single    |    |  bindings   |
      |  (REST,     |     |   binary,   |    |  (PyO3)     |
      |  axum)      |     |   ~35 MB)   |    |             |
      +------+------+     +------+------+    +------+------+
             |                   |                  |
             v                   v                  v
        Docker /             Lab / CLI          Python apps
        Sparrow Web          users              incl. PW SDK

             +---- C FFI (generated header) ----+
             v                                  v
        Sparrow Studio                      Other native
        Local (C#/P-Invoke)                 integrators
```

**Runtime:** ONNX Runtime (CPU or GPU). No PyTorch at inference time.

**Phase-1 model set (6 models):** MegaDetector v6, DeepFaune, HerdNet, OWL-T, SpeciesNet, MD_AudioBirds_V1. Vision (detection and classification) and audio classification.

**Model-agnostic:** adding a new model is a manifest change plus an ONNX file in the model directory. No engine code changes required.

**Model coverage:** the Phase-1 set is a subset of the full PyTorch-Wildlife model zoo. Additional models (e.g., AI4G Amazon Rainforest variants, Deepfaune-NE) will be onboarded in later phases via the same manifest path.

---

## How to adopt

Pick the surface that matches your stack.

| You are a… | Surface you use | Notes |
|---|---|---|
| Conservation user | No direct use — you interact via Sparrow Studio | Install the MSI; PW-Engine runs underneath |
| Existing Python user (`import PytorchWildlife`) | PyO3 bindings | Same API shape; drop-in |
| Web/cloud deployer running an inference server | Docker HTTP container | `docker run`; call `/v1/detect` |
| Laptop researcher | CLI — single static binary, ~35 MB | Invoke the CLI against a local image/audio file |
| Desktop app developer (Windows/.NET first; Mac/Linux ports in progress) | C FFI / C# bindings | Same integration path Sparrow Studio Local uses |
| Institutional / platform owner | Any combination | One inference implementation across desktop, server, and embedded |

**Custom models:** drop an ONNX file plus a manifest entry into the model directory. No engine code change.

**The Python PyTorch-Wildlife package keeps working.** PW-Engine is opt-in. Existing scripts and imports do not need to change; migrating to the PW-Engine Python bindings is a later, optional step.

---

## Status & roadmap

| Layer | Status |
|---|---|
| Core library + ONNX Runtime integration | Complete |
| HTTP REST server + Docker images | Complete |
| CLI + Python bindings | Complete |
| Utilities + model catalog | Complete |
| Data-management layer (SQLite-backed annotations and queries) | Planned |
| Multi-GPU scale-out | Not yet benchmarked |

Reliability hardening for long-running GPU workloads is in progress.

**Next milestone:** data-management sidecar.

**Release path:** preview today via the Sparrow Studio beta; a standalone release with a public source repository will follow. License: MIT.

---

## FAQ

**Does this replace PyTorch-Wildlife?**
No. The Python PyTorch-Wildlife package remains the user-facing API for training, fine-tuning, and research workflows. PW-Engine is the deployment and integration backend.

**When can I try it?**
Through the Sparrow Studio beta (Windows MSI) today. A standalone release with the public source repository will follow.

**Will my existing Python code break?**
No. PW-Engine is opt-in. The current PyTorch-Wildlife API is unchanged.

**Why is it called "PyTorch-Wildlife Engine" if it doesn't use PyTorch at runtime?**
PyTorch-Wildlife is the platform brand. PW-Engine uses ONNX Runtime; PyTorch is not in the inference path.

---

## Pilot

If you run an inference-heavy pipeline and want to pilot PW-Engine before the standalone release, reach out via the PyTorch-Wildlife Discord or email `zhongqimiao@microsoft.com`.
