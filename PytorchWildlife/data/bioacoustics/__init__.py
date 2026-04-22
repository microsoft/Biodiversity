"""
Bioacoustics infrastructure for PytorchWildlife.

This module provides tools for bioacoustics data processing, including annotations,
spectrograms, windowing, datasets, and configuration management.
"""

from .bioacoustics_annotations import AnnotationCreator, BaseReader
from .bioacoustics_configs import *
from .bioacoustics_datasets import *
from .bioacoustics_spectrograms import *
from .bioacoustics_windows import *
