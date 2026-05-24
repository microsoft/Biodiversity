"""
Unified training script for PW_Bioacoustics.

Supports both binary classification (num_classes=2) and multiclass (num_classes>2).

Usage:
    # Binary classification (default)
    python train.py --train_csv train.csv --val_csv val.csv --test_csv test.csv

    # Multiclass classification
    python train.py --train_csv train.csv --test_csv test.csv --num_classes 4

    # With YAML config
    python train.py --config config/template.yaml --train_csv train.csv --test_csv test.csv
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier
from PytorchWildlife.data.bioacoustics.bioacoustics_datasets import (
    BioacousticsDataset,
    SpectrogramAugmentations,
    MixUpCollator,
)
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config


@dataclass
class DataModuleConfig:
    """Configuration for the SpectrogramDataModule."""
    train_csv: str = "train_split.csv"
    val_csv: str = "val_split.csv"
    test_csv: str = "test_split.csv"
    root: Optional[str] = "mel_spectrograms"
    x_col: str = "spec_name"
    y_col: str = "label"
    target_size: list = field(default_factory=lambda: [224, 469])
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    use_specaug: bool = False
    normalize: bool = False
    pcen: bool = False
    num_classes: Optional[int] = None
    use_mixup: bool = True
    # Transform params
    horizontal_shift_prob: float = 0.5
    horizontal_shift_range: float = 0.2
    vertical_shift_prob: float = 0.5
    vertical_shift_range: float = 0.1
    occlusion_prob: float = 0.5
    occlusion_max_lines: int = 3
    occlusion_line_width: float = 0.05
    noise_prob: float = 0.5
    noise_std: float = 0.02
    buffer_prob: float = 0.5
    buffer_max_ratio: float = 0.2
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.2
    color_jitter_prob: float = 0.5
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3


class SpectrogramDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for spectrogram classification."""

    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.eval_transform = None

        if cfg.use_specaug:
            self.train_transform = SpectrogramAugmentations(
                horizontal_shift_prob=self.cfg.horizontal_shift_prob,
                horizontal_shift_range=self.cfg.horizontal_shift_range,
                vertical_shift_prob=self.cfg.vertical_shift_prob,
                vertical_shift_range=self.cfg.vertical_shift_range,
                occlusion_prob=self.cfg.occlusion_prob,
                occlusion_max_lines=self.cfg.occlusion_max_lines,
                occlusion_line_width=self.cfg.occlusion_line_width,
                noise_prob=self.cfg.noise_prob,
                noise_std=self.cfg.noise_std,
                buffer_prob=self.cfg.buffer_prob,
                buffer_max_ratio=self.cfg.buffer_max_ratio,
                color_jitter_prob=self.cfg.color_jitter_prob,
                brightness=self.cfg.color_jitter_brightness,
                contrast=self.cfg.color_jitter_contrast,
            )
        else:
            self.train_transform = None

    def setup(self, stage: Optional[str] = None):
        dataset_kwargs = dict(
            root=self.cfg.root,
            x_col=self.cfg.x_col,
            y_col=self.cfg.y_col,
            target_size=self.cfg.target_size,
            normalize=self.cfg.normalize,
        )
        if hasattr(self.cfg, 'pcen'):
            dataset_kwargs['pcen'] = self.cfg.pcen
        if self.cfg.num_classes is not None:
            dataset_kwargs['num_classes'] = self.cfg.num_classes

        if self.cfg.train_csv is not None:
            self.train_ds = BioacousticsDataset(
                csv_path=self.cfg.train_csv,
                transform=self.train_transform,
                is_training=True,
                **dataset_kwargs
            )
        if self.cfg.val_csv is not None:
            self.val_ds = BioacousticsDataset(
                csv_path=self.cfg.val_csv,
                transform=self.eval_transform,
                is_training=False,
                **dataset_kwargs
            )
        self.test_ds = BioacousticsDataset(
            csv_path=self.cfg.test_csv,
            transform=self.eval_transform,
            is_training=False,
            **dataset_kwargs
        )

    @property
    def num_classes(self) -> int:
        return self.test_ds.num_classes

    @property
    def in_channels(self) -> int:
        x0, _, _ = self.test_ds[0]
        return x0.shape[0]

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2

    def train_dataloader(self) -> DataLoader:
        if self.is_binary and self.cfg.use_mixup:
            collate_fn = MixUpCollator(
                mixup_prob=self.cfg.mixup_prob,
                mixup_alpha=self.cfg.mixup_alpha
            )
        else:
            collate_fn = None

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle_train,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )


def main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser(description="Unified training for bioacoustics classification")

    # Config file (optional)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data arguments
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--x_col", type=str, default="spec_name")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 469])

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--class_names", type=str, nargs="+", default=None)
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--monitor_metric", type=str, default="val/f1")
    parser.add_argument("--finetune", type=lambda x: (str(x).lower() == 'true'), default=False)

    # Preprocessing
    parser.add_argument("--normalize", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--pcen", action="store_true")

    # Binary-specific
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Freezing
    parser.add_argument("--freeze_backbone", type=str, default="none",
                        choices=["none", "all", "early", "layer1", "layer2", "layer3"])
    parser.add_argument("--backbone_lr_ratio", type=float, default=1.0)

    # Augmentation
    parser.add_argument("--use_specaug", action="store_true")
    parser.add_argument("--mixup_prob", type=float, default=0)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        cfg = load_config(args.config)
        # Apply config defaults where CLI args weren't explicitly set
        if args.num_classes == 2 and cfg.training.num_classes != 2:
            args.num_classes = cfg.training.num_classes
        if args.x_col == "spec_name":
            args.x_col = cfg.training.x_col
        if args.target_size == [224, 469]:
            args.target_size = cfg.training.target_size
        if args.class_names is None:
            args.class_names = list(cfg.class_names.values())
        if args.batch_size == 32:
            args.batch_size = cfg.training.batch_size
        if args.num_workers == 4:
            args.num_workers = cfg.training.num_workers
        if args.lr == 1e-4:
            args.lr = cfg.training.lr
        if args.weight_decay == 1e-4:
            args.weight_decay = cfg.training.weight_decay
        if args.epochs == 5:
            args.epochs = cfg.training.epochs
        if args.backbone == "resnet18":
            args.backbone = cfg.training.backbone

    # Create DataModule config
    dm_cfg = DataModuleConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root=args.root,
        x_col=args.x_col,
        target_size=args.target_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_specaug=args.use_specaug,
        normalize=args.normalize,
        pcen=args.pcen,
        num_classes=args.num_classes if args.num_classes != 2 else None,
        use_mixup=(args.num_classes == 2),
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
    )

    dm = SpectrogramDataModule(dm_cfg)
    dm.setup()

    num_classes = args.num_classes if args.num_classes != 2 else dm.num_classes

    model = ResNetClassifier(
        num_classes=num_classes,
        in_channels=dm.in_channels,
        backbone=args.backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        T_max=args.epochs,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight,
        conf_threshold=args.conf_threshold,
        freeze_backbone=args.freeze_backbone,
        backbone_lr_ratio=args.backbone_lr_ratio,
        class_names=args.class_names,
    )

    print(f"\nClassification mode: {'Binary' if num_classes == 2 else f'Multiclass ({num_classes} classes)'}")
    print(summary(model, input_size=(args.batch_size, dm.in_channels, *args.target_size)))

    # Callbacks & logging
    mode = "min" if args.monitor_metric == "val/loss" else "max"

    ckpt_cb = ModelCheckpoint(
        monitor=args.monitor_metric,
        mode=mode,
        save_top_k=1,
        save_last=True,
        filename="resnet-finetune-{epoch:02d}" if args.finetune else "resnet-{epoch:02d}",
    )

    early_cb = None
    if not args.finetune:
        early_cb = EarlyStopping(monitor=args.monitor_metric, mode=mode, patience=20)

    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[0],
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        callbacks=[cb for cb in [ckpt_cb, lr_cb, early_cb] if cb is not None],
        logger=False,
    )

    if args.ckpt_path is None:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")
        print("Best ckpt:", ckpt_cb.best_model_path)
        print("Best score:", ckpt_cb.best_model_score)
    else:
        model = ResNetClassifier.load_from_checkpoint(args.ckpt_path)

        if args.temperature != 1.0:
            model.temperature = torch.tensor(args.temperature, device=model.device)
            print(f"Using manual temperature: {args.temperature}")

        if model.is_binary:
            model.hparams.conf_threshold = args.conf_threshold

        if args.finetune:
            model.hparams.lr = args.lr
            model.hparams.weight_decay = args.weight_decay
            model.hparams.label_smoothing = args.label_smoothing
            model.hparams.T_max = args.epochs
            model.hparams.batch_size = args.batch_size
            model.hparams.freeze_backbone = args.freeze_backbone
            model.hparams.backbone_lr_ratio = args.backbone_lr_ratio

            print(f"Finetuning from checkpoint: {args.ckpt_path}")
            model._apply_freezing_strategy()

            trainer.fit(model, datamodule=dm)
            trainer.test(model, datamodule=dm, ckpt_path='best')
            print("Finetune completed.")
            print("Best ckpt:", ckpt_cb.best_model_path)
            print("Best score:", ckpt_cb.best_model_score)
        else:
            trainer.test(model, datamodule=dm)
            print(f"Test completed from checkpoint {args.ckpt_path}")


if __name__ == "__main__":
    main()
