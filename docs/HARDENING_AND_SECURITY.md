# HARDENING, ADVERSARIAL TESTING, AND SECURITY AUDIT

## 1. Overview
ForgeOutlier-Lite v1.0 enforces defensive programming and strict zero-trust boundaries across data ingestion, numerical calculation, deserialization, and pipeline execution.

## 2. Threat Models & Countermeasures

| Attack Vector / Failure Mode | Threat Profile | Defense Implementation | Verified Test Case |
| :--- | :--- | :--- | :--- |
| **Path Traversal / Escape** | Attacker passes relative path `../../file` or symlinks pointing outside allowed root to read sensitive system files. | Path canonicalization via `os.path.realpath`, prefix boundary assertion in `sanitize_input_path`. Rejects null-bytes (`\x00`). | `test_path_traversal_dots`, `test_null_byte_injection` |
| **Arbitrary Code Execution via Pickling** | Attacker supplies malicious `.pkl` or forged weights payload triggering Python RCE on load. | Universal ban of `pickle` across model checkpoints (`weights_only=True`) and OOD artifacts (`allow_pickle=False`). SHA-256 pre-deserialization validation against trusted manifest. | `test_secure_loader_and_manifest_verification`, `test_safe_serialization` |
| **Decompression Bomb (Zip/Image Bomb)** | Huge pixel dimension image (e.g. 100,000 x 100,000) intended to exhaust system RAM / crash server. | Pillow `Image.MAX_IMAGE_PIXELS` clamped to `MEDIA_LIMITS.max_image_pixels` (100,000,000 pixels). Throws `SecurityIntegrityError`. | `test_load_image_file_and_decompression_bomb` |
| **Corrupted Media Streams & Noise** | Truncated binary streams, corrupted JPEG headers, random binary garbage. | Controlled error boundary wrapping `PIL.UnidentifiedImageError` and `cv2` decode failures into `CorruptedMediaError` or `ExecutionFailureError`. Zero unhandled crashes. | `test_corrupted_image_header`, `test_random_binary_garbage_as_image`, `test_empty_zero_byte_media` |
| **Numerical Instability (NaN / Inf)** | Division by zero or non-finite float injection in feature vectors leading to poisoned metric calculation or OOD decisions. | Explicit finite checks (`np.all(np.isfinite(...))`) in metric evaluation, safe epsilon regularization in L2 normalization (`max(norm, eps)`), and guarded inverted covariance in Mahalanobis scorer. | `test_ood_scorers_reject_nan_features`, `test_metrics_reject_nan_arrays`, `test_normalize_l2_with_zeros` |
| **Network Data Exfiltration / Phoning Home** | Third-party dependencies attempting network telemetry or socket operations in sensitive air-gapped environments. | Absolute zero external network calls. Pipeline tested with socket completely disabled (`RuntimeError` on `socket.socket`). | `test_zero_network_pipeline_inference` |
| **Floating-Point Determinism Drift** | Non-deterministic weight initialization or random state variation leading to inconsistent forensic triage. | Strict seeding, deterministic matrix shrinkage (Ledoit-Wolf / OAS analytical estimators), and frozen PyTorch evaluation modes (`model.eval()`). | `test_frozen_synthetic_model_inference_determinism`, `test_frozen_covariance_and_mahalanobis_determinism` |

## 3. Verification Matrix
Total suite tests passing: **52 unit, integration, adversarial, and regression tests**.
- Core & Security: 5 tests
- Media Ingestion: 5 tests
- Model Architectures & Checkpoints: 6 tests
- OOD Scoring & Serialization: 6 tests
- Pipeline Logic: 5 tests
- Evaluation Engine: 5 tests
- CLI & Serving: 5 tests
- Adversarial Hardening: 10 tests
- Regression & Determinism: 3 tests
- Symlink Escape: 1 skipped (requires administrative symlink privilege on Windows).
