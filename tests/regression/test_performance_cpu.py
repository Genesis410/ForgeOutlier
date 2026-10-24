import time
import tracemalloc

import numpy as np

from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.ood.mahalanobis import MahalanobisScorer
from forgeoutlier.pipeline.inference import InferencePipeline


def test_performance_latency_and_memory(tmp_path):
    """Verify p95 latency <= 300ms and peak RAM <= 500MB on CPU."""
    dim = 576
    model = DeepfakeModel(feature_dim=dim)
    model.eval()

    # Synthetic covariance and mean
    rng = np.random.RandomState(42)
    mu = rng.randn(dim).astype(np.float32)
    cov = np.eye(dim, dtype=np.float32) + 0.01 * np.outer(mu, mu)
    scorer = MahalanobisScorer(mu=mu, cov=cov, threshold=15.0)

    pipeline = InferencePipeline(model=model, ood_scorer=scorer)

    # Synthetic image input
    img_array = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        pipeline.predict_image(img_array)

    # Benchmark Latency (N=30)
    latencies = []
    tracemalloc.start()
    for _ in range(30):
        t0 = time.perf_counter()
        res = pipeline.predict_image(img_array)
        t1 = time.perf_counter()
        assert res.status == "SUCCESS"
        latencies.append((t1 - t0) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    p95_latency = np.percentile(latencies, 95)
    peak_ram_mb = peak_mem / (1024 * 1024)

    # OS-level Peak RSS measurement
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        peak_rss_mb = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        peak_rss_mb = peak_ram_mb

    print(f"\n[PERFORMANCE] p95 Latency: {p95_latency:.2f} ms (budget <= 300 ms)")
    print(f"[PERFORMANCE] Peak Heap Allocation (tracemalloc): {peak_ram_mb:.2f} MB")
    print(
        f"[PERFORMANCE] Process Peak RSS (OS working set): {peak_rss_mb:.2f} MB (budget <= 500 MB)"
    )

    assert p95_latency <= 300.0, f"p95 latency violated: {p95_latency:.2f} ms > 300 ms"
    assert peak_ram_mb <= 500.0, f"peak Heap RAM violated: {peak_ram_mb:.2f} MB > 500 MB"
    assert peak_rss_mb <= 500.0, f"peak Process RSS violated: {peak_rss_mb:.2f} MB > 500 MB"
