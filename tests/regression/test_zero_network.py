"""Regression test: Enforce absolute zero network connectivity during inference."""

import socket

import numpy as np
import pytest
from PIL import Image

from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.ood.cosine import CosineCentroidScorer
from forgeoutlier.pipeline.inference import InferencePipeline


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch socket.socket to forbid any outgoing network calls."""

    def guarded_socket(*args, **kwargs):
        raise RuntimeError(
            "CRITICAL SECURITY VIOLATION: Network socket access detected in offline pipeline!"
        )

    monkeypatch.setattr(socket, "socket", guarded_socket)


def test_zero_network_pipeline_inference(tmp_path) -> None:
    """Validate full pipeline executes without network attempts."""
    dim = 576
    model = DeepfakeModel(feature_dim=dim)
    scorer = CosineCentroidScorer(mu=np.zeros(dim, dtype=np.float32), threshold=0.5)
    pipeline = InferencePipeline(model=model, ood_scorer=scorer)

    img_path = tmp_path / "probe.png"
    Image.new("RGB", (224, 224), color=(50, 50, 50)).save(img_path)

    # Must complete cleanly while network is blocked
    result = pipeline.predict_image(str(img_path))
    assert result.status == "SUCCESS"
    assert result.decision is not None
