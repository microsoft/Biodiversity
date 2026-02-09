"""
Annotations infrastructure for PytorchWildlife.

This module provides tools for creating and reading standardized bioacoustics annotations
in a COCO-inspired JSON format.
"""

from .annotation_creator import AnnotationCreator
from .base_reader import BaseReader

__all__ = ["AnnotationCreator", "BaseReader"]
