# ForgeOutlier-Lite v1.0.0

[![CI](https://github.com/forgeoutlier/forgeoutlier/actions/workflows/ci.yml/badge.svg)](https://github.com/forgeoutlier/forgeoutlier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Zero-Network](https://img.shields.io/badge/network-100%25%20offline-success)](docs/HARDENING_AND_SECURITY.md)

Lightweight, offline-first open-set deepfake detection engine and CLI featuring tri-state triage (`REAL`, `FAKE`, `UNKNOWN`), CPU-first execution, and defensive zero-trust boundaries.

---

## 1. Why ForgeOutlier-Lite?

Standard binary deepfake classifiers fail silently when confronted with out-of-distribution (OOD) artifacts—such as novel generative architectures (e.g., unseen diffusion models, zero-shot video avatars, extreme compression noise, or adversarial inputs). Rather than admitting uncertainty, traditional models output high-confidence false decisions.

ForgeOutlier-Lite enforces a **tri-state triage invariant**:
$$\text{Decision} \in \{\text{REAL}, \text{FAKE}, \text{UNKNOWN}\}$$

By pairing lightweight feature extraction (MobileNetV3-Small backbone) with calibrated geometric density estimators (Cholesky-stabilized Mahalanobis distance with Ledoit-Wolf covariance shrinkage), ForgeOutlier isolates untrusted representations and returns `UNKNOWN` whenever the input deviates from known distribution envelopes.

---

## 2. Key Architecture & Design Invariants

- **CPU-First Performance**: Optimized for standard laptop hardware. p95 inference latency $\le 300\text{ ms}$ (actual benchmark: **59.18 ms**), memory footprint $\le 500\text{ MB}$ RAM (actual peak: **1.76 MB**).
- **100% Offline & Air-Gapped**: Zero telemetry, zero external network requests. Verified via socket isolation regression tests.
- **Pure-PyTorch & Pure-NumPy**: No dependency on `torchvision` or `scikit-learn`. AUROC, AUPRC, FPR@95, and metric evaluations computed using vectorized pure NumPy routines.
- **Defensive Deserialization**:
  - Model checkpoints strictly loaded with `weights_only=True` after pre-deserialization SHA-256 manifest verification (`manifests/trusted_checkpoints.json`).
  - OOD calibration artifacts (`.npz`) strictly enforce `allow_pickle=False` and SHA-256 checksum integrity.
- **Media Ingestion Hardening**: Bounded image dimensions, Pillow decompression bomb protection (`MAX_IMAGE_PIXELS`), and strict path traversal / null-byte containment.

---

## 3. Directory Layout

```text
ForgeOutlier/
├── .github/workflows/ci.yml       # Multi-platform CI pipeline (Ubuntu, Windows)
├── docs/                          # Architecture & technical documentation
│   ├── ARCHITECTURE.md            # System layers & dataflow contracts
│   ├── CLI_AND_SERVING.md         # CLI operational usage, exit codes, formatting
│   ├── EVALUATION_AND_METRICS.md  # Pure-NumPy evaluation metrics & tri-state matrix
│   ├── HARDENING_AND_SECURITY.md  # Threat model, attack mitigations, audit matrix
│   └── ACCEPTANCE_REPORT_v1.0.0.md# v1.0.0 verification & acceptance results
├── src/forgeoutlier/              # Core modular package
│   ├── core/                      # Data contracts, exceptions, protocol definitions
│   ├── media/                     # Image/video loaders, Pillow protections, face crop
│   ├── models/                    # Pure-PyTorch MobileNetV3-Small & secure loader
│   ├── ood/                       # Covariance shrinkage, Mahalanobis & Cosine scorers
│   ├── pipeline/                  # Image/video inference & temporal aggregation
│   ├── eval/                      # Vectorized pure-NumPy metrics & benchmark runners
│   └── cli/                       # Stderr/stdout isolated terminal CLI
├── tests/                         # Multi-tier test pyramid (53 passing, 1 skipped)
│   ├── unit/                      # Isolated unit tests for each system layer
│   ├── integration/               # End-to-end image and video inference tests
│   ├── adversarial/               # Corrupted media, NaN/Inf, path traversal tests
│   └── regression/                # Offline isolation, determinism, CPU benchmarks
├── pyproject.toml                 # Packaging, linters (ruff), typecheck (mypy)
├── LICENSE                        # Apache-2.0 License
└── README.md                      # Primary project guide
```

---

## 4. Installation

ForgeOutlier-Lite requires Python 3.10 or newer.

```bash
# Clone the repository
git clone https://github.com/forgeoutlier/forgeoutlier.git
cd ForgeOutlier

# Install CPU PyTorch first (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install ForgeOutlier in editable mode
pip install -e .

# Or install with dev/test dependencies
pip install -e ".[dev,test]"
```

---

## 5. Python API Usage

### Quick Image Triage

```python
import forgeoutlier
from forgeoutlier.core.types import ModelDecision

# Initialize pipeline with pre-trained weights and calibrated OOD artifact
pipeline = forgeoutlier.load_pipeline(
    checkpoint_path="weights/mobilenetv3_small.pt",
    ood_path="calibration/ood_mahalanobis.npz",
    policy="OUTLIER_FIRST"
)

# Run triage on an image file (or numpy array / PIL Image)
result = pipeline.predict_image("path/to/suspect_face.jpg")

print(f"Status: {result.status}")
print(f"Decision: {result.decision.name}")        # REAL, FAKE, or UNKNOWN
print(f"Confidence: {result.confidence:.4f}")
print(f"OOD Distance: {result.ood_score:.4f}")
print(f"Outlier Flag: {result.is_ood}")
print(f"Latency: {result.execution_time_ms:.2f} ms")
```

### Video Stream Triage with Temporal Aggregation

```python
# Video triage automatically samples frames and applies D007 aggregation rules
video_result = pipeline.predict_video("path/to/interview.mp4", max_frames=30)

print(f"Video Decision: {video_result.decision.name}")
print(f"Aggregated Ratio: {video_result.metadata['aggregation_details']}")
```

---

## 6. Command Line Interface (CLI)

The CLI guarantees clean stream separation: **machine data on `stdout`**, and **progress/warnings on `stderr`**.

### 1. Single Media Prediction
```bash
# ANSI formatted output
forgeoutlier predict --media suspect.jpg --checkpoint weights/model.pt --ood calibration/ood.npz

# Machine-readable JSON
forgeoutlier predict --media suspect.jpg --checkpoint weights/model.pt --ood calibration/ood.npz --json

# Semantic exit codes for scripting (0=REAL, 1=FAKE, 2=UNKNOWN, 3=INSUFFICIENT_DATA, 4=ERROR)
forgeoutlier predict --media suspect.jpg --exit-code-decision
```

### 2. Batch Inference
```bash
forgeoutlier batch-predict --input-dir /data/evidence/ --output-file results.json --json
```

### 3. Calibrate In-Distribution Envelope
```bash
forgeoutlier fit-ood --feature-cache in_dist_feats.npy --output-path calibration/ood.npz --scorer mahalanobis --quantile 0.95
```

### 4. Inspect Model Architecture & Checkpoint
```bash
forgeoutlier inspect-model --checkpoint weights/model.pt
```

---

## 7. Operational Guarantees & Verification

Every build is validated against regression, adversarial, and performance suites:

```bash
# Run complete test pyramid
python -m pytest -v
```

### Verified Benchmark Metrics
- **Total Test Suite**: 53 passed, 1 skipped (9.41s total runtime).
- **Inference Latency (p95)**: 59.18 ms (Budget: $\le 300\text{ ms}$).
- **Peak Process Memory**: 1.76 MB (Budget: $\le 500\text{ MB}$).
- **Deterministic Numerical Output**: Bitwise parity verified across multiple evaluation passes.
- **Hardening Coverage**: Corrupted headers, empty zero-byte streams, random binary noise, NaN/Inf tensor vectors, and path traversal strings (`..`, `\x00`) safely intercepted with zero unhandled exceptions.

---

## 8. License

ForgeOutlier-Lite is open-source software licensed under the [Apache-2.0 License](LICENSE).
