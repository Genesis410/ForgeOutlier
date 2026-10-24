"""Global pytest configuration and fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def setup_deterministic_seed() -> None:
    """Enforce deterministic seed and single-threaded BLAS for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
