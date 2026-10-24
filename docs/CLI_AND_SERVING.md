# CLI & Serving Interface Specification

## Overview
ForgeOutlier-Lite provides a production-grade, zero-dependency CLI interface engineered for single-media inference, batch processing, OOD baseline fitting, and model checkpoint auditing.

The CLI strictly respects pipeline safety contracts:
1. **Piping Purity**: Standard output (`sys.stdout`) only streams requested data formats (JSON or TSV). Progress indicators, tracebacks, and warnings are directed exclusively to `sys.stderr`.
2. **Deterministic Exit Codes**: Supports both standard POSIX execution status and tri-state decision status codes for downstream CI/CD pipelines.

---

## Command Reference

### 1. `predict`
Predicts a single image or video file.

```bash
# Standard box output with ANSI styling
python -m forgeoutlier predict samples/sample.jpg \
  --model-path models/checkpoints/deepfake_v1.pt \
  --profile-path models/ood/baseline_profile.npz

# JSON output piped to jq
python -m forgeoutlier predict samples/sample.jpg --json | jq .result.decision

# Semantic decision exit codes
python -m forgeoutlier predict samples/sample.jpg --exit-code-decision
```

#### Exit Codes (`--exit-code-decision`)
- `0`: REAL (`Decision.REAL`)
- `2`: FAKE (`Decision.FAKE`)
- `3`: UNKNOWN (`Decision.UNKNOWN`)
- `1`: Execution failure / invalid argument

---

### 2. `batch-predict`
Processes directories or manifests of mixed image/video inputs with real-time `stderr` progress reporting.

```bash
python -m forgeoutlier batch-predict data/eval_subset/ \
  --model-path models/checkpoints/deepfake_v1.pt \
  --profile-path models/ood/baseline_profile.npz \
  --plain > results.tsv
```

---

### 3. `fit-ood`
Computes statistical baselines (Centroid and Ledoit-Wolf Covariance) from saved In-Distribution feature embeddings (`.npy`).

```bash
python -m forgeoutlier fit-ood \
  --embeddings-path artifacts/id_features.npy \
  --output-path models/ood/baseline_profile.npz \
  --method mahalanobis \
  --quantile 0.95
```

---

### 4. `inspect-model`
Calculates file SHA-256 and validates checkpoint integrity before deployment.

```bash
python -m forgeoutlier inspect-model models/checkpoints/deepfake_v1.pt
```
