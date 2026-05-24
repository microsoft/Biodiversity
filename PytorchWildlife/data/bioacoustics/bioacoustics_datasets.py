# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Bioacoustics datasets and data augmentation transforms for spectrogram classification.

This module provides:
- BioacousticsDataset: Dataset for loading spectrograms from .npy files
- SpectrogramAugmentations: Spectrogram-specific augmentation techniques
- MixUpCollator: Batch-level MixUp augmentation
- Utility transforms for normalization and resizing
"""

import os
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import librosa
except ImportError:
    librosa = None


__all__ = [
    "BioacousticsDataset",
    "BioacousticsInferenceDataset",
    "SpectrogramAugmentations",
    "MixUpCollator",
    "PerSampleNormalize",
    "ResizeTo",
    "mixup_criterion",
]


# --------------- Transforms ---------------

class PerSampleNormalize(nn.Module):
    """
    Normalize each sample x:[C,H,W] to zero-mean / unit-std (per entire tensor).
    Keeps scale stable without relying on dataset-wide stats.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std().clamp_min(1e-6)
        return (x - mean) / std


class ResizeTo:
    """
    Resize a spectrogram tensor to (H, W).

    Args:
        size_hw: Target size as [height, width].
    """
    def __init__(self, size_hw: List[int]):
        self.size_hw = size_hw

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResizeTo expects [C,H,W], got {tuple(x.shape)}")
        h, w = self.size_hw
        if (x.shape[-2], x.shape[-1]) == (h, w):
            return x
        # Crop width if needed (common for spectrograms)
        return x[:, :, :w]


class SpectrogramAugmentations:
    """
    Spectrogram-specific augmentation techniques.

    Includes horizontal/vertical shifts, random occlusions, Gaussian noise,
    buffer simulation, and color jitter.

    Args:
        horizontal_shift_prob: Probability of applying horizontal shift.
        horizontal_shift_range: Fraction of time dimension to shift.
        vertical_shift_prob: Probability of applying vertical shift.
        vertical_shift_range: Fraction of frequency dimension to shift.
        occlusion_prob: Probability of applying random occlusion.
        occlusion_max_lines: Max number of occlusion lines.
        occlusion_line_width: Width of occlusion lines as fraction.
        noise_prob: Probability of adding Gaussian noise.
        noise_std: Std of Gaussian noise.
        buffer_prob: Probability of buffer corruption.
        buffer_max_ratio: Max corruption ratio for buffer simulation.
        color_jitter_prob: Probability of applying color jitter.
        brightness: Brightness adjustment factor.
        contrast: Contrast adjustment factor.
    """

    def __init__(
        self,
        horizontal_shift_prob: float = 0.1,
        horizontal_shift_range: float = 0.1,
        vertical_shift_prob: float = 0.1,
        vertical_shift_range: float = 0.1,
        occlusion_prob: float = 0.15,
        occlusion_max_lines: int = 2,
        occlusion_line_width: float = 0.05,
        noise_prob: float = 0.1,
        noise_std: float = 0.02,
        buffer_prob: float = 0.05,
        buffer_max_ratio: float = 0.1,
        color_jitter_prob: float = 0.1,
        brightness: float = 0.1,
        contrast: float = 0.1,
    ):
        self.horizontal_shift_prob = horizontal_shift_prob
        self.horizontal_shift_range = horizontal_shift_range
        self.vertical_shift_prob = vertical_shift_prob
        self.vertical_shift_range = vertical_shift_range
        self.occlusion_prob = occlusion_prob
        self.occlusion_max_lines = occlusion_max_lines
        self.occlusion_line_width = occlusion_line_width
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.buffer_prob = buffer_prob
        self.buffer_max_ratio = buffer_max_ratio
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast

    def horizontal_shift(self, spec):
        if torch.rand(1) < self.horizontal_shift_prob:
            _, _, time_dim = spec.shape
            shift_pixels = int(torch.randint(
                -int(time_dim * self.horizontal_shift_range),
                int(time_dim * self.horizontal_shift_range) + 1, (1,)
            ))
            if shift_pixels != 0:
                mean_val = spec.mean()
                spec = torch.roll(spec, shifts=shift_pixels, dims=2)
                if shift_pixels > 0:
                    spec[:, :, :shift_pixels] = mean_val
                else:
                    spec[:, :, shift_pixels:] = mean_val
        return spec

    def vertical_shift(self, spec):
        if torch.rand(1) < self.vertical_shift_prob:
            _, freq_dim, _ = spec.shape
            shift_bins = int(torch.randint(
                -int(freq_dim * self.vertical_shift_range),
                int(freq_dim * self.vertical_shift_range) + 1, (1,)
            ))
            if shift_bins != 0:
                spec = torch.roll(spec, shifts=shift_bins, dims=1)
                mean_val = spec.mean()
                if shift_bins > 0:
                    spec[:, :shift_bins, :] = mean_val
                else:
                    spec[:, shift_bins:, :] = mean_val
        return spec

    def add_occlusions(self, spec):
        if torch.rand(1) < self.occlusion_prob:
            _, freq_dim, time_dim = spec.shape
            num_lines = torch.randint(1, self.occlusion_max_lines + 1, (1,)).item()
            mean_val = spec.mean()
            for _ in range(num_lines):
                if torch.rand(1) < 0.5:
                    freq_start = torch.randint(0, freq_dim, (1,)).item()
                    line_width = torch.randint(1, int(freq_dim * self.occlusion_line_width), (1,)).item()
                    freq_end = min(freq_start + line_width, freq_dim)
                    spec[:, freq_start:freq_end, :] = mean_val
                else:
                    time_start = torch.randint(0, time_dim, (1,)).item()
                    line_width = torch.randint(1, int(time_dim * self.occlusion_line_width), (1,)).item()
                    time_end = min(time_start + line_width, time_dim)
                    spec[:, :, time_start:time_end] = mean_val
        return spec

    def add_gaussian_noise(self, spec):
        if torch.rand(1) < self.noise_prob:
            noise = torch.randn_like(spec) * self.noise_std
            spec = spec + noise
        return spec

    def add_buffer_simulation(self, spec):
        if torch.rand(1) < self.buffer_prob:
            _, freq_dim, time_dim = spec.shape
            downsample_factor = 1.0 - torch.rand(1) * self.buffer_max_ratio
            new_time_dim = max(1, int(time_dim * downsample_factor))
            new_freq_dim = max(1, int(freq_dim * downsample_factor))
            spec_down = F.interpolate(
                spec.unsqueeze(0),
                size=(new_freq_dim, new_time_dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            spec = F.interpolate(
                spec_down.unsqueeze(0),
                size=(freq_dim, time_dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return spec

    def color_jitter(self, spec):
        if torch.rand(1) < self.color_jitter_prob:
            if self.brightness > 0:
                brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
                spec = spec * brightness_factor
            if self.contrast > 0:
                mean_val = spec.mean()
                contrast_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
                spec = (spec - mean_val) * contrast_factor + mean_val
        return spec

    def __call__(self, spec, is_training=True):
        if not is_training:
            return spec
        augmentations = [
            self.horizontal_shift,
            self.vertical_shift,
            self.add_occlusions,
            self.add_gaussian_noise,
            self.add_buffer_simulation,
            self.color_jitter,
        ]
        num_to_apply = len(augmentations)
        selected = random.sample(augmentations, num_to_apply)
        random.shuffle(selected)
        for aug in selected:
            spec = aug(spec)
        return spec


# --------------- MixUp ---------------

class MixUpCollator:
    """
    A collate function that applies MixUp augmentation at the batch level.

    Args:
        mixup_prob: Probability of applying MixUp.
        mixup_alpha: Alpha parameter for Beta distribution sampling.
    """

    def __init__(self, mixup_prob: float = 0.2, mixup_alpha: float = 0.2):
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

    def __call__(self, batch):
        """
        Apply MixUp to a batch of (spectrogram, label, path) tuples.

        Returns:
            specs: Tensor of mixed spectrograms [B, C, H, W].
            labels: Dict with info for MixUp loss calculation.
            paths: List of original paths.
        """
        specs, labels, paths = zip(*batch)
        specs = torch.stack(specs)
        labels = torch.tensor(labels)
        batch_size = specs.size(0)

        if torch.rand(1) < self.mixup_prob and batch_size > 1:
            noise_intensity = self.mixup_alpha
            lam = torch.clamp(
                1.0 - torch.abs(torch.randn(1) * noise_intensity), 0.0, 1.0
            ).item()

            indices = torch.randperm(batch_size)
            mixed_specs = lam * specs + (1 - lam) * specs[indices]

            mixed_labels = {
                'original_labels': labels,
                'shuffled_labels': labels[indices],
                'lambda': lam,
                'is_mixed': True
            }

            return mixed_specs, mixed_labels, list(paths)
        else:
            mixed_labels = {
                'original_labels': labels,
                'shuffled_labels': None,
                'lambda': 1.0,
                'is_mixed': False
            }
            return specs, mixed_labels, list(paths)


def mixup_criterion(criterion, pred, targets):
    """
    Compute the MixUp loss.

    Args:
        criterion: Loss function (e.g., nn.BCEWithLogitsLoss()).
        pred: Model predictions.
        targets: Dictionary containing mixed label information.

    Returns:
        Mixed loss value.
    """
    if targets['is_mixed']:
        y_a, y_b = targets['original_labels'].float(), targets['shuffled_labels'].float()
        lam = targets['lambda']
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    else:
        return criterion(pred, targets['original_labels'].float())


# --------------- Dataset ---------------

class BioacousticsDataset(Dataset):
    """
    Dataset that reads spectrograms from .npy files whose paths are listed in a CSV.

    Args:
        csv_path: Path to the CSV file.
        root: Root folder to prepend to spec_name when it's a relative path.
        x_col: Column containing the path to the .npy file.
        y_col: Column containing the label.
        target_size: [H, W] to resize spectrograms to; if None, keep original size.
        transform: Callable applied for transformations.
        is_training: Whether this is training mode (affects augmentations).
        normalize: Whether to apply per-sample normalization.
        pcen: Whether to apply PCEN transformation.
        num_classes: Number of classes; if None, inferred from data.
    """

    def __init__(
        self,
        csv_path: str,
        root: Optional[str] = None,
        x_col: str = "spec_name",
        y_col: str = "label",
        target_size: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False,
        normalize: bool = True,
        pcen: bool = False,
        num_classes: int = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform
        self.is_training = is_training
        self.pcen = pcen

        # Resolved paths
        self.paths: List[str] = []
        for p in self.df[self.x_col].astype(str).tolist():
            self.paths.append(os.path.join(self.root, p) if self.root else p)

        self._resize = ResizeTo(target_size) if target_size is not None else None
        self._normalize = PerSampleNormalize() if normalize else None

        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = int(self.df[self.y_col].max()) + 1

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, idx: int):
        path = self.paths[idx]
        try:
            arr = np.load(path)
        except (EOFError, ValueError) as e:
            print(f"ERROR loading file at index {idx}: {path}")
            print(f"File size: {os.path.getsize(path) if os.path.exists(path) else 'FILE NOT FOUND'} bytes")
            raise e
        return arr, path

    def _apply_pcen(self, S_db, sr=24000):
        """Apply Per-Channel Energy Normalization."""
        if librosa is None:
            raise ImportError("librosa is required for PCEN. Install with: pip install librosa")

        S_lin = librosa.db_to_power(S_db)
        S_pcen = librosa.pcen(
            S_lin * (2**31),
            sr=sr,
            gain=1.0,
            bias=10.0,
            power=0.5,
            time_constant=0.3,
        )
        S_pcen = librosa.power_to_db(S_pcen)
        return S_pcen.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        arr, path = self._load_npy(idx)
        arr = arr.astype(np.float32, copy=False)

        if self.pcen:
            arr = self._apply_pcen(arr)

        # Shape to [C, H, W]
        if arr.ndim == 2:
            arr = arr[None, ...]  # [1, H, W]
        elif arr.ndim == 3:
            if arr.shape[0] not in (1, 2, 3) and arr.shape[-1] in (1, 2, 3):
                arr = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Unexpected .npy shape {arr.shape} at index {idx}")

        x = torch.from_numpy(arr)

        if self._normalize is not None:
            x = self._normalize(x)

        if self._resize is not None:
            x = self._resize(x)

        if self.transform is not None:
            x = self.transform(x, self.is_training)

        y = int(self.df.iloc[idx][self.y_col])

        return x, y, path


class BioacousticsInferenceDataset(Dataset):
    """
    Dataset that reads spectrograms from .npy files whose paths are listed in a dataframe.

    Unlike :class:`BioacousticsDataset`, this class does **not** require a
    label column and returns ``(tensor, path)`` pairs suitable for inference.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing at least the column specified by `x_col`.
    x_col : str
        Column containing the path to the .npy file (default: "spec_name").
    target_size : Optional[List[int]]
        [H, W] to resize spectrograms to; if None, keep original size.
    normalize : bool
        Whether to apply per-sample normalization.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        x_col: str = "spec_name",
        target_size: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.df = dataframe
        self.x_col = x_col
        self.paths = self.df[self.x_col].astype(str).tolist()
        self._resize = ResizeTo(target_size) if target_size is not None else None
        self._normalize = PerSampleNormalize() if normalize else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, idx: int):
        path = self.paths[idx]
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR loading file at index {idx}:")
            print(f"Path: {path}")
            print(f"Exception: {e}")
            print(f"{'='*80}\n")
            raise
        return arr, path

    def __getitem__(self, idx: int):
        arr, path = self._load_npy(idx)
        arr = arr.astype(np.float32, copy=False)

        # shape to [C,H,W]
        if arr.ndim == 2:
            arr = arr[None, ...]  # [1,H,W]
        elif arr.ndim == 3:
            if arr.shape[0] not in (1, 2, 3) and arr.shape[-1] in (1, 2, 3):
                arr = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Unexpected .npy shape {arr.shape} at index {idx}")

        x = torch.from_numpy(arr)  # [C,H,W]

        if self._normalize is not None:
            x = self._normalize(x)

        if self._resize is not None:
            x = self._resize(x)

        return x, path
