# Core Architecture & Contracts (`forgeoutlier.core`)

Modul `forgeoutlier.core` adalah Layer 1 (fondasi independen) dari arsitektur ForgeOutlier Lite. Modul ini tidak mengimpor modul downstream (`models`, `ood`, `media`, `pipeline`, `cli`) demi menjaga prinsip unidirectional dependency flow.

## 1. Domain Types (`core.types`)
- `Decision`: Enum keputusan tri-state (`REAL`, `FAKE`, `UNKNOWN`).
- `Policy`: Enum strategi triage (`OUTLIER_FIRST`, `DUAL_SCORE`).
- `MediaMeta`: Metadata teknis media masukan (dimensi, durasi, frame count).
- `ScoreResult`: Struktur nilai numerik (`prob_fake`, `prob_real`, `ood_score`, `is_outlier`, `threshold`).
- `VideoSummary`: Ringkasan agregasi frame video (rasio OOD, rasio FAKE/REAL, rincian per frame).
- `PredictionResult`: Kontrak data root yang serializable ke format JSON standar (mematuhi JSON schema `SRS_TECHNICAL_CONTRACT.md`).

## 2. Exceptions Hierarchy (`core.exceptions`)
- `ForgeOutlierException`: Base class dengan atribut kode error terstruktur (`code`).
- `SecurityIntegrityError` (`SECURITY_INTEGRITY_VIOLATION`): Pelanggaran path traversal, symlink escape, mismatch SHA-256.
- `CorruptedMediaError` (`CORRUPTED_MEDIA_ERROR`): Berkas rusak atau terpotong.
- `ResourceLimitExceededError` (`RESOURCE_LIMIT_EXCEEDED`): Pelanggaran batas resource (resolusi $>4096$, ukuran $>50$ MB / $500$ MB, frame $>1000$).
- `NumericalInstabilityError` (`NUMERICAL_INSTABILITY_ERROR`): Matriks kovarians singular, dekomposisi Cholesky gagal, nilai `NaN`/`Inf`.
- `InsufficientDataError` (`INSUFFICIENT_DATA_ERROR`): Data sampel tidak mencukupi untuk fitting kovarians / kalibrasi.

## 3. Runtime Contracts (`core.protocols`)
- `EncoderProtocol`: Kontrak ekstraktor fitur (`feature_dim`, `extract_features(x)`, `predict_probabilities(features)`).
- `OODScorerProtocol`: Kontrak detektor OOD (`threshold`, `score(embeddings)`, `is_outlier(embeddings)`).
- Ditandai `@runtime_checkable` agar dapat divalidasi via `isinstance()`.

## 4. Configuration & Boundaries (`core.config`)
- `MEDIA_LIMITS`: Dataclass beku (`max_image_dim=4096`, `max_image_bytes=52428800`, `max_video_frames=1000`).
- `NUMERICAL_CONSTANTS`: Epsilon stabilisasi $\epsilon = 10^{-12}$, default quantile threshold $= 95.0\%$, toleransi komputasi float32/float64.

## 5. Security Utilities (`core.security`)
- `sanitize_input_path(user_path, base_dir, allow_absolute)`: Mengamankan masukan path berkas dari serangan injeksi null-byte (`\0`) dan directory traversal (`../`).
