# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base bioacoustics classifier class."""

import torch.nn as nn


__all__ = ["BaseBioacousticsClassifier"]


class BaseBioacousticsClassifier(nn.Module):
    """
    Base class for bioacoustics classifiers.

    This class provides utility methods for loading models, generating results,
    and performing single and batch audio classifications on spectrograms.
    """

    # Placeholder class-level attributes to be defined in derived classes
    SAMPLE_RATE = None
    WINDOW_SIZE_SEC = None
    N_MELS = None
    CLASS_NAMES = None

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the base bioacoustics classifier.

        Args:
            weights (str, optional): Path to model weights. Defaults to None.
            device (str, optional): Device for inference. Defaults to "cpu".
            url (str, optional): URL to fetch model weights. Defaults to None.
        """
        super(BaseBioacousticsClassifier, self).__init__()
        self.device = device

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load model weights.

        Args:
            weights (str, optional): Path to model weights. Defaults to None.
            device (str, optional): Device for inference. Defaults to "cpu".
            url (str, optional): URL to fetch model weights. Defaults to None.

        Raises:
            Exception: If weights are not provided.
        """
        pass

    def results_generation(self, preds, audio_id: str, id_strip: str = None) -> dict:
        """
        Generate results for classification based on model predictions.

        Args:
            preds: Model predictions (logits or probabilities).
            audio_id (str): Audio identifier.
            id_strip (str, optional): Strip specific characters from audio_id.

        Returns:
            dict: Dictionary containing audio ID, predictions, and labels.
        """
        pass

    def single_audio_classification(
        self, spectrogram, audio_path=None, conf_threshold=0.5, id_strip=None
    ) -> dict:
        """
        Perform classification on a single spectrogram.

        Args:
            spectrogram: Spectrogram tensor or ndarray.
            audio_path (str, optional): Audio path or identifier.
            conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.
            id_strip (str, optional): Characters to strip from audio_id.

        Returns:
            dict: Classification results.
        """
        pass

    def batch_audio_classification(
        self, dataloader, conf_threshold: float = 0.5, id_strip: str = None
    ) -> list[dict]:
        """
        Perform classification on a batch of spectrograms.

        Args:
            dataloader (DataLoader): DataLoader containing spectrogram batches.
            conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.
            id_strip (str, optional): Characters to strip from audio_id.

        Returns:
            list[dict]: List of classification results for all audio samples.
        """
        pass
