"""Classification head for binary deepfake detection."""

import torch
import torch.nn as nn


class BinaryClassifierHead(nn.Module):
    """Linear classification head mapping pre-logits to binary classes (0: REAL, 1: FAKE)."""

    def __init__(self, in_features: int = 576, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pre-logits z to raw logits."""
        return self.classifier(self.dropout(z))
