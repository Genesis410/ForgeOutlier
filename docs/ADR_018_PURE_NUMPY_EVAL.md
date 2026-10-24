# ADR-018: Evaluasi Metrik Berbasis Pure-NumPy Tanpa Ketergantungan scikit-learn

## Status
Diterima & Diterapkan (Accepted)

## Konteks
Keputusan arsitektur awal D017 mempertimbangkan `scikit-learn` sebagai dependensi kandidat untuk estimasi kovarians Ledoit-Wolf dan evaluasi metrik (AUROC, AUPRC, FPR@95% TPR). Namun, `scikit-learn` memperkenalkan ukuran dependensi binary yang besar (~30–50 MB wheel) dan overhead import yang signifikan, yang bertentangan dengan prinsip ForgeOutlier-Lite sebagai runtime CPU-only ultra-ringan dengan instalasi instan.

## Keputusan
1. Mengeliminasi `scikit-learn` dari dependensi wajib (`dependencies` di `pyproject.toml`).
2. Mengimplementasikan seluruh algoritma metrik secara internal menggunakan pure vectorized NumPy:
   - Estimasi kovarians shrinkage Ledoit-Wolf terintegrasi di `forgeoutlier.ood.fitting`.
   - Kurva ROC dan AUROC via rank sorting dan akumulasi `np.cumsum`.
   - Kurva Precision-Recall dan Average Precision (AUPRC) terstandardisasi.
   - Perhitungan FPR pada target TPR tertentu via linear interpolation / nearest neighbor.
3. Menjamin verifikasi numerik: deviasi toleransi terhadap `sklearn.metrics` pada float64 dibatasi $\le 10^{-5}$.

## Konsekuensi
- **Positif**:
  - Ukuran instalasi library sangat kecil (hanya bergantung pada `torch`, `numpy`, `pillow`, `opencv-python-headless`).
  - Tidak ada overhead dynamic linking ekstra di lingkungan serverless atau embedded CPU.
  - Alur kalkulasi transparan dan sepenuhnya dapat diaudit.
- **Tantangan / Mitigasi**:
  - Implementasi AUPRC harus menggunakan formula step-wise Riemann sum $\sum (R_n - R_{n-1}) P_n$ agar identik dengan standar industri `sklearn.metrics.average_precision_score`, bukan interpolasi trapesium biasa yang rentan optimistik pada kurva PR.
