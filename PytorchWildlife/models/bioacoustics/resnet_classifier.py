# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ResNet-based classifier for bioacoustics spectrogram classification.

Supports both binary classification (num_classes=2) and multiclass (num_classes>2).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm

import pytorch_lightning as pl

# Binary metrics
from torchmetrics.classification import (
    BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score,
    BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryPrecisionRecallCurve
)
# Multiclass metrics
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassConfusionMatrix
)

from sklearn.metrics import average_precision_score

try:
    from torch_uncertainty.losses import BCEWithLogitsLSLoss
except ImportError:
    BCEWithLogitsLSLoss = None


__all__ = ["ResNetClassifier"]


class ResNetClassifier(pl.LightningModule):
    """
    Unified ResNet classifier supporting both binary and multiclass classification.

    When num_classes=2: Binary mode with BCEWithLogitsLoss, sigmoid activation.
    When num_classes>2: Multiclass mode with CrossEntropyLoss, softmax activation.

    Args:
        num_classes: Number of output classes (2 for binary, >2 for multiclass).
        in_channels: Number of input channels (1 for grayscale spectrograms).
        backbone: ResNet backbone variant ("resnet18", "resnet34", "resnet50").
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        label_smoothing: Label smoothing factor.
        T_max: Cosine annealing T_max parameter.
        batch_size: Batch size for logging.
        pos_weight: Positive class weight (binary only).
        conf_threshold: Confidence threshold for predictions (binary only).
        freeze_backbone: Freezing strategy ("none", "all", "early", "layer1-3").
        backbone_lr_ratio: Learning rate ratio for backbone vs classifier.
        class_names: List of class names for multiclass labeling.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        backbone: str = "resnet18",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        T_max: int = 100,
        batch_size: int = 32,
        pos_weight: float = 1.0,
        conf_threshold: float = 0.5,
        freeze_backbone: str = "none",
        backbone_lr_ratio: float = 1.0,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.is_binary = (num_classes == 2)

        # Build backbone with ImageNet weights
        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet34":
            net = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Adapt first conv to match spectrogram channels (typically 1)
        old_conv = net.conv1
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            net.conv1 = new_conv

        # Output head depends on classification mode
        in_feats = net.fc.in_features
        if self.is_binary:
            net.fc = nn.Linear(in_feats, 1)
        else:
            net.fc = nn.Linear(in_feats, num_classes)
        self.net = net

        # Apply freezing strategy
        self._apply_freezing_strategy()

        # Initialize loss and metrics based on mode
        if self.is_binary:
            self._init_binary_loss_and_metrics()
        else:
            self._init_multiclass_loss_and_metrics(num_classes)

        # Storage for test predictions
        self.test_logits = []
        self.test_targets = []
        self.test_paths = []
        self.test_preds = []

    def _init_binary_loss_and_metrics(self):
        """Initialize loss and metrics for binary classification."""
        if BCEWithLogitsLSLoss is not None:
            self.criterion = BCEWithLogitsLSLoss(label_smoothing=self.hparams.label_smoothing)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.train_auprc = BinaryAveragePrecision()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auprc = BinaryAveragePrecision()

        self.test_prec = BinaryPrecision()
        self.test_rec = BinaryRecall()
        self.test_cm = BinaryConfusionMatrix()
        self.test_prcurve = BinaryPrecisionRecallCurve(thresholds=None)

    def _init_multiclass_loss_and_metrics(self, num_classes: int):
        """Initialize loss and metrics for multiclass classification."""
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

        self.test_prec = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.test_rec = MulticlassRecall(num_classes=num_classes, average='macro')
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

        self.test_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.test_prec_per_class = MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_rec_per_class = MulticlassRecall(num_classes=num_classes, average=None)

    def _apply_freezing_strategy(self):
        """Apply layer freezing based on freeze_backbone parameter."""
        freeze_until = None

        if self.hparams.freeze_backbone == "none":
            return
        elif self.hparams.freeze_backbone == "all":
            freeze_until = "fc"
        elif self.hparams.freeze_backbone == "early":
            freeze_until = "layer3"
        elif self.hparams.freeze_backbone in ["layer1", "layer2", "layer3"]:
            freeze_until = self.hparams.freeze_backbone
        else:
            raise ValueError(f"Invalid freeze_backbone: {self.hparams.freeze_backbone}")

        if freeze_until:
            trainable_params = 0
            frozen_params = 0
            freeze = True

            for name, param in self.net.named_parameters():
                if freeze_until in name:
                    freeze = False
                param.requires_grad = not freeze
                if param.requires_grad:
                    trainable_params += param.numel()
                else:
                    frozen_params += param.numel()

            print(f"\n{'='*60}")
            print(f"Freezing strategy: {self.hparams.freeze_backbone}")
            print(f"Frozen parameters: {frozen_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Frozen ratio: {frozen_params/(frozen_params+trainable_params)*100:.1f}%")
            print(f"{'='*60}\n")

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, logits, y):
        """Compute loss handling both regular and MixUp batches."""
        # Import here to avoid circular dependency
        from PytorchWildlife.data.bioacoustics_datasets import mixup_criterion

        if self.is_binary:
            logits = logits.squeeze(1)
            if isinstance(y, dict) and y.get('is_mixed', False):
                loss = mixup_criterion(self.criterion, logits, y)
                targets = y['original_labels'].int()
            else:
                targets = y['original_labels'].int() if isinstance(y, dict) else y.int()
                loss = self.criterion(logits, targets.float())
            preds = (logits > 0).int()
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            targets = y

        return loss, preds, targets, logits

    def training_step(self, batch, batch_idx):
        x, y, path = batch
        logits = self(x)
        loss, preds, targets, logits_processed = self._compute_loss(logits, y)

        self.train_acc.update(preds, targets)
        self.train_f1.update(preds, targets)
        if self.is_binary:
            self.train_auprc.update(logits_processed, targets)

        self.log("train/loss", loss, batch_size=self.hparams.batch_size, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        f1 = self.train_f1.compute()
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)

        if self.is_binary:
            auprc = self.train_auprc.compute()
            self.log("train/auprc", auprc, prog_bar=True)
            self.train_auprc.reset()

        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y, path = batch
        logits = self(x)

        if self.is_binary:
            logits = logits.squeeze(1)
            loss = self.criterion(logits, y.float())
            preds = (logits > 0).int()
            targets = y.int()
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            targets = y

        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)
        if self.is_binary:
            self.val_auprc.update(logits, targets)

        self.log("val/loss", loss, batch_size=self.hparams.batch_size, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)

        if self.is_binary:
            auprc = self.val_auprc.compute()
            self.log("val/auprc", auprc, prog_bar=True)
            self.val_auprc.reset()

        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y, path = batch
        logits = self(x)

        if hasattr(self, "temperature"):
            logits = logits / self.temperature

        if self.is_binary:
            logits = logits.squeeze(1)
            loss = self.criterion(logits, y.float())
            prob = torch.sigmoid(logits)
            preds = (prob > self.hparams.conf_threshold).int()
            targets = y.int()
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            targets = y

        self.test_acc.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_prec.update(preds, targets)
        self.test_rec.update(preds, targets)
        self.test_cm.update(preds, targets)

        if self.is_binary:
            self.test_auprc.update(logits, targets)
            self.test_prcurve.update(logits, targets)
        else:
            self.test_f1_per_class.update(preds, targets)
            self.test_prec_per_class.update(preds, targets)
            self.test_rec_per_class.update(preds, targets)

        self.test_logits.append(logits.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.test_paths.extend(path)
        self.test_preds.append(preds.detach().cpu())

        self.log("test/loss", loss, batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        if self.is_binary:
            self._on_test_epoch_end_binary()
        else:
            self._on_test_epoch_end_multiclass()

    def _on_test_epoch_end_binary(self):
        """Test epoch end processing for binary classification."""
        f1 = self.test_f1.compute().item()
        auprc = self.test_auprc.compute().item()
        prec = self.test_prec.compute().item()
        rec = self.test_rec.compute().item()

        self.log("test/f1", f1)
        self.log("test/auprc", auprc)
        self.log("test/precision", prec)
        self.log("test/recall", rec)

        cm = self.test_cm.compute().cpu().numpy()
        acc_neg = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0.0
        acc_pos = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0.0
        self.log("test/acc_neg", acc_neg)
        self.log("test/acc_pos", acc_pos)

        self._reset_test_state()

    def _on_test_epoch_end_multiclass(self):
        """Test epoch end processing for multiclass classification."""
        num_classes = self.hparams.num_classes

        logits = torch.cat(self.test_logits, dim=0)
        targets = torch.cat(self.test_targets, dim=0).numpy()

        if hasattr(self, "temperature"):
            temp = self.temperature.cpu() if self.temperature.is_cuda else self.temperature
            probs = torch.softmax(logits / temp, dim=1).numpy()
        else:
            probs = torch.softmax(logits, dim=1).numpy()

        acc = self.test_acc.compute().item()
        f1 = self.test_f1.compute().item()
        prec = self.test_prec.compute().item()
        rec = self.test_rec.compute().item()

        macro_ap = np.nan
        if len(np.unique(targets)) == num_classes:
            macro_ap = average_precision_score(
                y_true=pd.get_dummies(targets),
                y_score=probs,
                average="macro"
            )

        self.log_dict({
            "test/acc_micro": acc,
            "test/f1": f1,
            "test/precision": prec,
            "test/recall": rec,
            "test/macro_average_precision": macro_ap,
        })

        self._reset_test_state()

    def _reset_test_state(self):
        """Reset test metrics and storage."""
        self.test_cm.reset()
        self.test_f1.reset()
        self.test_acc.reset()
        self.test_prec.reset()
        self.test_rec.reset()

        if self.is_binary:
            self.test_auprc.reset()
            self.test_prcurve.reset()
        else:
            self.test_f1_per_class.reset()
            self.test_prec_per_class.reset()
            self.test_rec_per_class.reset()

        self.test_logits.clear()
        self.test_targets.clear()
        self.test_paths.clear()
        self.test_preds.clear()

    def configure_optimizers(self):
        if self.hparams.backbone_lr_ratio != 1.0:
            backbone_params = []
            classifier_params = []

            for name, param in self.net.named_parameters():
                if not param.requires_grad:
                    continue
                if 'fc' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)

            backbone_lr = self.hparams.lr * self.hparams.backbone_lr_ratio
            classifier_lr = self.hparams.lr

            param_groups = []
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': backbone_lr})
            if classifier_params:
                param_groups.append({'params': classifier_params, 'lr': classifier_lr})

            print(f"Using discriminative LR: backbone={backbone_lr:.2e}, classifier={classifier_lr:.2e}")
            opt = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.T_max)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
