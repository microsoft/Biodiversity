# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bioacoustics models for PytorchWildlife."""

from .base_bioacoustics import BaseBioacousticsClassifier
from .resnet_classifier import ResNetClassifier

__all__ = [
    "BaseBioacousticsClassifier",
    "ResNetClassifier",
]
