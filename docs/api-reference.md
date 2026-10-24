# Referensi API Python ForgeOutlier

Dokumen ini mendokumentasikan antarmuka publik Python library ForgeOutlier.

---

## 1. Pemuatan Pipeline (`forgeoutlier.load_pipeline`)

```python
from forgeoutlier import load_pipeline

pipeline = load_pipeline(
    model_checkpoint_path="checkpoints/mobilenetv3_small.pt",
    ood_artifact_path="artifacts/id_profile.npz",
    policy="outlier_first",  # Pilihan: 'outlier_first', 'dual_score', 'baseline_naive'
    device="cpu",
)
```

### Parameter:
- `model_checkpoint_path` (*str | Path*): Jalur ke berkas checkpoint PyTorch model (`.pt`).
- `ood_artifact_path` (*str | Path*, opsional): Jalur ke artefak distribusi referensi In-Distribution (`.npz`). Wajib jika menggunakan policy non-naive.
- `policy` (*str*, default: `'outlier_first'`): Strategi triage tri-state.
- `device` (*str*, default: `'cpu'`): Perangkat komputasi PyTorch.

---

## 2. Kelas `InferencePipeline`

Menyediakan fungsi inferensi per-citra dan per-video.

### `pipeline.predict_image(image_input: str | Path | np.ndarray) -> PredictionResult`
Menjalankan inferensi pada citra tunggal.
- **Return**: Objek `PredictionResult` yang berisi:
  - `decision`: Enum `Decision.REAL`, `Decision.FAKE`, atau `Decision.UNKNOWN`.
  - `scores`: Objek `ScoreBundle` memuat `prob_real`, `prob_fake`, `ood_score`, `threshold`.
  - `reason`: Kode alasan mesin (misal `OOD_THRESHOLD_EXCEEDED`, `PROBABILITY_CLASSIFICATION`).
  - `latency_ms`: Waktu inferensi dalam milidetik.

### `pipeline.predict_video(video_path: str | Path, sample_fps: float = 1.0) -> VideoPredictionResult`
Menjalankan inferensi berbasis frame sampling pada berkas video.
- **Return**: Objek `VideoPredictionResult` memuat keputusan agregat video dan rincian frame-by-frame.

---

## 3. Ekstraksi Fitur & Evaluasi

### `forgeoutlier.eval.metrics`
- `compute_auroc(y_true, scores)`: Menghitung Area Under ROC Curve via vectorized rank statistic.
- `compute_auprc(y_true, scores)`: Menghitung Area Under Precision-Recall Curve via Riemann summation step-function (paritas bit-exact vs scikit-learn).
- `compute_risk_coverage(risk_errors, confidence_scores)`: Menghitung kurva coverage, selective risk, dan AURC.
- `compute_ece(y_true, probs, n_bins=10)`: Menghitung Expected Calibration Error (ECE).
