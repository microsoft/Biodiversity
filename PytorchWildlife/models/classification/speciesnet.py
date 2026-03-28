import timm
import torch.nn as nn

from .base_classifier import BaseClassifier


class SpeciesNet(BaseClassifier):
    """SpeciesNet classification model using a ResNet‑50 backbone.

    This model follows the same interface as other classification models in the
    library, allowing it to be instantiated via the configuration system and
    used interchangeably with existing models.
    """

    def __init__(self, num_cls, pretrained=True, **kwargs):
        """Create a ResNet‑50 model with a custom classification head.

        Args:
            num_cls (int): Number of output classes.
            pretrained (bool, optional): Whether to load ImageNet‑pretrained
                weights for the backbone. Defaults to True.
            **kwargs: Additional keyword arguments passed to ``timm.create_model``.
        """
        super().__init__()
        # ``timm`` handles the creation of the backbone and the classification head.
        self.backbone = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_cls, **kwargs
        )

    def forward(self, x):
        """Forward pass through the ResNet‑50 backbone.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Logits for each class.
        """
        return self.backbone(x)


__all__ = ["SpeciesNet"]
