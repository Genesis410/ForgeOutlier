"""Complete deepfake detection model combining backbone and classifier."""

import numpy as np
import torch
import torch.nn as nn

from forgeoutlier.core.exceptions import ExecutionFailureError
from forgeoutlier.models.backbone import MobileNetV3SmallBackbone
from forgeoutlier.models.classifier import BinaryClassifierHead


class DeepfakeModel(nn.Module):
    """Integrated MobileNetV3-Small backbone and classification head.

    Adheres strictly to EncoderProtocol.
    Input images must be normalized: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    """

    def __init__(self, feature_dim: int = 576, num_classes: int = 2) -> None:
        super().__init__()
        self.features = MobileNetV3SmallBackbone(feature_dim=feature_dim)
        self.classifier = BinaryClassifierHead(in_features=feature_dim, num_classes=num_classes)
        self.eval()

    @property
    def feature_dim(self) -> int:
        return self.features.feature_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, features)."""
        z = self.features(x)
        logits = self.classifier(z)
        return logits, z

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract penultimate feature embeddings.

        Args:
            x: np.ndarray of shape (N, C, H, W) or (C, H, W), float32.

        Returns:
            np.ndarray of shape (N, D), float32.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)

        if not np.all(np.isfinite(x)):
            raise ExecutionFailureError("Non-finite values (NaN or Inf) detected in input array.")

        x_clean = x.astype(np.float32)
        if x_clean.ndim == 3:
            x_clean = np.expand_dims(x_clean, axis=0)
        elif x_clean.ndim != 4:
            raise ExecutionFailureError(f"Input array must be 3D or 4D, got shape {x.shape}")

        if x_clean.shape[1] != 3:
            raise ExecutionFailureError(
                f"Expected 3 channels (RGB) at dim 1, got shape {x_clean.shape}"
            )

        device = next(self.parameters()).device
        tensor_x = torch.from_numpy(x_clean).to(device)

        with torch.no_grad():
            self.eval()
            z = self.features(tensor_x)
            feats = z.cpu().numpy().astype(np.float32)

        return feats

    def predict_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Predict softmax probabilities over classes [REAL, FAKE].

        Args:
            x: np.ndarray of shape (N, C, H, W) or (C, H, W), float32.

        Returns:
            np.ndarray of shape (N, 2), float32.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)

        if not np.all(np.isfinite(x)):
            raise ExecutionFailureError("Non-finite values (NaN or Inf) detected in input array.")

        x_clean = x.astype(np.float32)
        if x_clean.ndim == 3:
            x_clean = np.expand_dims(x_clean, axis=0)
        elif x_clean.ndim != 4:
            raise ExecutionFailureError(f"Input array must be 3D or 4D, got shape {x.shape}")

        device = next(self.parameters()).device
        tensor_x = torch.from_numpy(x_clean).to(device)

        with torch.no_grad():
            self.eval()
            logits, _ = self.forward(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

        return probs
