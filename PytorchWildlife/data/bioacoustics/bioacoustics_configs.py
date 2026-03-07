# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Bioacoustics configuration schema for PytorchWildlife.

This module provides configuration dataclasses and loader/saver functions
for bioacoustics experiments.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


__all__ = [
    "PathConfig",
    "AudioConfig",
    "SpectrogramConfig",
    "TrainingConfig",
    "SplitsConfig",
    "DomainConfig",
    "load_config",
    "save_config",
]


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


@dataclass
class PathConfig:
    """Paths configuration with environment variable expansion."""
    data_root: str = ""
    output_root: str = ""
    spectrograms_dir: str = ""
    annotations_file: str = "annotations.json"
    windows_json: str = "windows_annotations.json"

    def __post_init__(self):
        """Expand environment variables and resolve paths."""
        self.data_root = os.path.expandvars(self.data_root)
        self.output_root = os.path.expandvars(self.output_root)
        self.spectrograms_dir = os.path.expandvars(self.spectrograms_dir)

    @property
    def annotations_path(self) -> str:
        """Full path to annotations file."""
        return os.path.join(self.data_root, self.annotations_file)


@dataclass
class AudioConfig:
    """Audio processing parameters."""
    sample_rate: int = 48000
    window_size_sec: float = 5.0
    overlap_sec: float = 4.0
    window_strategy: str = "sliding"  # "sliding", "balanced", or "customized"
    negative_proportion: float = 0.5  # For "balanced" strategy
    windows_csv: str = ""  # Path to pre-built CSV for "customized" strategy
    windows_json: str = ""  # Filename for saving/loading the windows JSON file
    multiclass: bool = False  # Use category_id labels instead of binary 0/1
    min_overlap_sec: float = 0  # Minimum overlap (s) to label a window positive

    @property
    def hop_size_sec(self) -> float:
        """Hop size in seconds (window_size - overlap)."""
        return self.window_size_sec - self.overlap_sec


@dataclass
class SpectrogramConfig:
    """Mel spectrogram generation parameters."""
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 224
    top_db: float = 80.0
    f_min: float = 0.0  # Minimum frequency (Hz) for the mel filterbank
    mono_channel: str = "left"  # "left", "right", or "mean" for stereo→mono
    fill_highfreq: bool = True
    fill_mean_below_sr: bool = False  # Fill with mean instead of noise when orig_sr < target_sr
    noise_db_std: float = 3.0
    storage_dtype: str = "float32"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    backbone: str = "resnet18"
    num_classes: int = 2  # 2 = binary mode, >2 = multiclass
    label_smoothing: float = 0.0
    target_size: List[int] = field(default_factory=lambda: [224, 469])
    x_col: str = "spec_name"
    y_col: str = "label"
    normalize: bool = True
    use_specaug: bool = False
    pos_weight: float = 1.0  # Binary only
    conf_threshold: float = 0.5  # Binary only
    freeze_backbone: str = "none"
    backbone_lr_ratio: float = 1.0


@dataclass
class SplitsConfig:
    """Data split parameters."""
    test_size: float = 0.15
    val_size: float = 0.15
    n_splits: int = 5
    random_state: int = 42
    custom_splits_folder: Optional[str] = None


@dataclass
class DomainConfig:
    """Complete domain-specific configuration."""
    name: str = ""
    datasets: List[str] = field(default_factory=list)
    class_names: Dict[int, str] = field(default_factory=dict)
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary classification task."""
        return self.training.num_classes == 2


def load_config(config_path: str) -> DomainConfig:
    """
    Load configuration from a YAML file.

    Environment variables in the format ${VAR_NAME} are expanded.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        DomainConfig object with all settings.

    Example:
        config = load_config("config/birds.yaml")
        print(config.audio.sample_rate)  # 48000
    """
    if yaml is None:
        raise ImportError("PyYAML is required for config loading. Install with: pip install pyyaml")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Expand environment variables
    data = _expand_env_vars(data)

    # Build nested config objects
    paths = PathConfig(**data.get('paths', {}))
    audio = AudioConfig(**data.get('audio', {}))
    # If windows_json is set in audio but not in paths, propagate it
    if audio.windows_json and not data.get('paths', {}).get('windows_json'):
        paths.windows_json = audio.windows_json
    spectrogram = SpectrogramConfig(**data.get('spectrogram', {}))
    training = TrainingConfig(**data.get('training', {}))
    splits = SplitsConfig(**data.get('splits', {}))

    return DomainConfig(
        name=data.get('name', ''),
        datasets=data.get('datasets', []),
        class_names=data.get('class_names', {}),
        paths=paths,
        audio=audio,
        spectrogram=spectrogram,
        training=training,
        splits=splits,
    )


def save_config(config: DomainConfig, config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: DomainConfig object to save.
        config_path: Path to save the YAML file.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for config saving. Install with: pip install pyyaml")

    from dataclasses import asdict

    data = {
        'name': config.name,
        'datasets': config.datasets,
        'class_names': config.class_names,
        'paths': asdict(config.paths),
        'audio': asdict(config.audio),
        'spectrogram': asdict(config.spectrogram),
        'training': asdict(config.training),
        'splits': asdict(config.splits),
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
