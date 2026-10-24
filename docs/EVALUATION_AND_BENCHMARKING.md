# Evaluation & Benchmarking Suite (Layer 3: `forgeoutlier.eval`)

## Ringkasan Arsitektur
Modul `forgeoutlier.eval` dirancang untuk peneliti dan perekayasa sistem deteksi deepfake open-set guna mengukur kemampuan separasi domain In-Distribution (ID) vs Out-of-Distribution (OOD) serta evaluasi kinerja tri-state decision triage (`REAL`, `FAKE`, `UNKNOWN`).

Implementasi dilakukan secara mandiri (pure-NumPy) tanpa ketergantungan pada pustaka eksternal seperti `scikit-learn` untuk menjaga jejak instalasi CPU yang ramping dan terprediksi.

```
Evaluasi Dataset Labeled (ID / OOD)
             │
             ▼
 InferencePipeline (Batch Image Analysis)
             │
             ├──► ood_scores, prob_fake, decisions
             │
             ▼
 Metrics Computation Engine (`eval/metrics.py`)
  ├── ROC Curve & AUROC (Trapezoid integration)
  ├── PR Curve & AUPRC (Precision-Recall integration)
  ├── FPR@95% TPR (Ambang False Positive pada 95% TPR OOD)
  └── Tri-State Confusion Matrix:
        Rows: Ground Truth [REAL, FAKE, OOD]
        Cols: Predicted Decision [REAL, FAKE, UNKNOWN]
             │
             ▼
 Benchmark Runner (`eval/benchmark.py`)
  ├── Agregasi metrik keseluruhan (BenchmarkSummary)
  ├── Perhitungan fatal error rate (OOD diprediksi REAL)
  └── Perhitungan abstained rate (OOD diprediksi UNKNOWN)
             │
             ▼
 Export & Reporting Suite (`eval/export.py`)
  ├── export_summary_json: Format JSON terstruktur
  ├── export_confusion_matrix_csv: Format tabel matriks konfusi
  └── export_predictions_csv: Detail prediksi per sampel
```

---

## Modul dan Spesifikasi

### 1. `forgeoutlier.eval.metrics`
Menghitung kurva dan metrik skalar evaluasi:
- **`compute_roc_curve(y_true, scores)`**:
  - Menghitung nilai False Positive Rate ($FPR$), True Positive Rate ($TPR$), dan ambang batas berurutan.
- **`compute_auroc(y_true, scores)`**:
  - Menghitung luas area di bawah kurva ROC menggunakan integrasi trapesium NumPy.
- **`compute_pr_curve(y_true, scores)`**:
  - Menghitung Precision dan Recall pada berbagai threshold.
- **`compute_auprc(y_true, scores)`**:
  - Menghitung luas area di bawah kurva Precision-Recall.
- **`compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)`**:
  - Mencari nilai $FPR$ terkecil pada saat $TPR \ge 0.95$, standar evaluasi deteksi outlier ilmiah.
- **`compute_tristate_confusion_matrix(y_true_labels, decisions)`**:
  - Menghitung matriks konfusi tri-state $3 \times 3$ untuk `REAL`, `FAKE`, dan `OOD`.

### 2. `forgeoutlier.eval.benchmark`
- **`BenchmarkSummary`**:
  - Dataclass yang merangkum total sampel, pembagian ID/OOD, nilai AUROC, AUPRC, FPR@95% TPR, matriks konfusi, serta rasio fatal error dan penolakan (abstention).
- **`evaluate_dataset(pipeline, image_paths, labels)`**:
  - Mengevaluasi seluruh gambar dalam dataset menggunakan `InferencePipeline` dan mengembalikan objek `BenchmarkSummary`.

### 3. `forgeoutlier.eval.export`
- **`export_summary_json(summary, output_path)`**:
  - Menyimpan ringkasan benchmark ke file JSON terformat.
- **`export_confusion_matrix_csv(confusion_matrix, output_path)`**:
  - Menyimpan tabel matriks konfusi ke CSV dengan kolom `GROUND_TRUTH, PRED_REAL, PRED_FAKE, PRED_UNKNOWN`.
- **`export_predictions_csv(predictions, output_path)`**:
  - Menyimpan rincian prediksi per sampel citra ke CSV.

---

## Pengujian dan Verifikasi
- Unit test: `tests/unit/test_eval.py`
  - Pemisahan sempurna (AUROC = 1.0, FPR@95 = 0.0).
  - Skor terbalik / performa buruk (AUROC = 0.0, FPR@95 = 1.0).
  - Validasi error input (dimensi non-1D, ketidakcocokan panjang, ketiadaan kelas positif/negatif).
  - Validasi matriks konfusi tri-state.
  - End-to-end benchmark dataset sintetis dan ekspor JSON/CSV.
- Status: 5 unit test lulus (total keseluruhan test suite: 34 tests passing).
