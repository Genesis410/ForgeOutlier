# Pipeline Orchestration & Decision Engine (Layer 2D)

## Ringkasan Arsitektur
Layer 2D mengorkestrasi end-to-end inferensi citra tunggal dan sekuens video berbatas (bounded streaming) dengan memadukan feature extraction dari model PyTorch, scoring Out-Of-Distribution (OOD), dan evaluasi tri-state decision triage (`REAL`, `FAKE`, `UNKNOWN`).

```
Input Media (Image / VideoStream)
           │
           ▼
 Image Ingestion & Standardization (ImageNet Normalization CHW)
           │
           ▼
 PyTorch DeepfakeModel (Feature Extraction: 576-dim / 1024-dim)
     │                                            │
     ▼                                            ▼
 Classifier Head (Sigmoid)                 L2-Normalized Embedding
     │                                            │
     ▼                                            ▼
 prob_fake, prob_real                      OOD Scorer (Mahalanobis / Cosine)
     │                                            │
     │                                            ▼
     │                                     ood_score, is_outlier
     │                                            │
     └───────────────────┬────────────────────────┘
                         │
                         ▼
        Decision Policy Engine (decisions.py)
           [OUTLIER_FIRST / DUAL_SCORE]
                         │
                         ▼
             Single-Frame Tri-State Decision
                         │
                         ▼ (Khusus Video)
        Video Temporal Frame Aggregation (aggregation.py)
         - Check Minimum Frames ($M \ge 4$)
         - Outlier Dominance ($R_{OOD} > 0.40 \to$ UNKNOWN)
         - Fake Persistence ($R_{FAKE} \ge 0.25 \to$ FAKE)
         - Homogeneous Real ($R_{REAL} > 0.75 \to$ REAL)
                         │
                         ▼
              PredictionResult Schema 1.0.0
```

---

## Modul dan Komponen

### 1. `forgeoutlier.pipeline.decisions`
Mengevaluasi keputusan tri-state frame tunggal:
- **`OUTLIER_FIRST`**:
  - Jika $\text{ood\_score} > \tau$ (atau `is_outlier == True`): Keputusan = `Decision.UNKNOWN`, reason = `OOD_THRESHOLD_EXCEEDED`.
  - Jika $\text{ood\_score} \le \tau$:
    - Jika $\text{prob\_fake} \ge \tau_{clf}$ (default 0.50): `Decision.FAKE`.
    - Selain itu: `Decision.REAL`.
    - reason = `CONFIDENT_CLASSIFICATION`.
- **`DUAL_SCORE`**:
  - Jika $\text{prob\_fake} \ge 0.95$ dan $\text{ood\_score} \le 1.25 \times \tau$: Override ke `Decision.FAKE`, reason = `CONFIDENT_CLASSIFICATION`.
  - Selain itu jika $\text{ood\_score} > \tau$: `Decision.UNKNOWN`, reason = `OOD_THRESHOLD_EXCEEDED`.
  - Selain itu jika $\text{prob\_fake} \ge \tau_{clf}$: `Decision.FAKE`, reason = `CONFIDENT_CLASSIFICATION`.
  - Default: `Decision.REAL`, reason = `CONFIDENT_CLASSIFICATION`.

### 2. `forgeoutlier.pipeline.aggregation`
Mengevaluasi agregasi statistik untuk video stream:
- **Kontrak $M \ge 4$**:
  - Jika jumlah frame teranalisis $M < 4$, melempar `InsufficientDataError("INSUFFICIENT_DATA")`.
- **Rasio Outlier Dominan**:
  - Jika $R_{OOD} = \frac{N_{OOD}}{M} > 0.40 \implies \text{Decision.UNKNOWN}$, reason = `VIDEO_FRAMES_MOSTLY_OOD`.
- **Rasio Fake Persisten**:
  - Jika $R_{FAKE} = \frac{N_{FAKE}}{M} \ge 0.25 \implies \text{Decision.FAKE}$, reason = `PERSISTENT_FAKE_FRAMES`.
- **Real Homogen**:
  - Kondisi sisa $\implies \text{Decision.REAL}$, reason = `HOMOGENEOUS_REAL_FRAMES`.

### 3. `forgeoutlier.pipeline.inference.InferencePipeline`
Kelas orkestrator pipeline:
- `predict_image(source)`: Menerima path citra atau `Image.Image`, mengekstrak fitur, menghitung probabilitas classifier dan OOD score, mengembalikan `PredictionResult`.
- `predict_video(video_path, target_samples=16)`: Membaca sekuens frame video menggunakan context manager `VideoStream`, mengekstrak fitur tiap frame, mengagregasi rasio keputusan, mengembalikan `PredictionResult` lengkap dengan `VideoSummary`.

### 4. `forgeoutlier.load_pipeline`
Fungsi facade tingkat atas:
- Memverifikasi manifest dan SHA-256 checkpoint model PyTorch via `load_checkpoint_safe`.
- Memvalidasi dan memuat artefak profil OOD `.npz` via `load_ood_artifact`.
- Menginisiasi instance `InferencePipeline` yang siap pakai.

---

## Verifikasi dan Pengujian
- Unit test keputusan & agregasi: `tests/unit/test_pipeline_units.py` (3 test lulus).
- Integration test inferensi end-to-end citra & video: `tests/integration/test_pipeline_e2e.py` (2 test lulus).
- Seluruh rangkaian test suite (29/29 tests) terverifikasi lulus 100% pada Python 3.14 CPU environment.
