# OOD Engine & Mathematical Specifications (`forgeoutlier.ood`)

Modul `forgeoutlier.ood` adalah Layer 2A yang mengimplementasikan mesin deteksi Out-of-Distribution (OOD), fitting kovarians statistik, dan serialisasi terproteksi.

## 1. Mathematical Implementations (`ood.fitting`)
- `normalize_l2(x, eps)`:
  $$x_{\text{norm}} = \frac{x}{\|x\|_2 + \epsilon}$$
  Mengubah vektor embedding menjadi unit norm pada dimensi terakhir, validasi nilai finite (menolak NaN/Inf).
- `fit_centroid_and_covariance(embeddings)`:
  - Sentroid $\mu = \frac{1}{N}\sum z_i$.
  - Kovarians well-conditioned via Ledoit-Wolf shrinkage ($\Sigma_{\text{LW}} = (1-\delta)\Sigma + \delta \mu_p I$).
  - Menolak sampel $N < 2$ (`InsufficientDataError`).
- `calibrate_quantile_threshold(scores, percentile)`:
  Menghitung ambang batas empiris (default persentil ke-95) dari ID validation set.

## 2. Scoring Engines
### Mahalanobis Distance (`ood.mahalanobis.MahalanobisScorer`)
- Berbasis dekomposisi Cholesky lower-triangular $\Sigma = L L^T$.
- Jarak dihitung dengan menyelesaikan sistem triangular $L y = (z - \mu)$ menggunakan `scipy.linalg.solve_triangular` ($O(D^2)$ per sampel).
- Menghindari inversi eksplisit $\Sigma^{-1}$ ($O(D^3)$) yang rentan ketidakstabilan numerik.
- Rumus jarak: $d = \|y\|_2 = \sqrt{y^T y}$.

### Cosine Centroid Distance (`ood.cosine.CosineCentroidScorer`)
- Jarak kosinus terpusat:
  $$d_{\cos} = 1.0 - (\hat{z} \cdot \hat{\mu})$$
- Hasil berada pada rentang $[0.0, 2.0]$.

## 3. Safe Serialization (`ood.serialization`)
- `save_ood_artifact`: Menyimpan array $\mu$, threshold, dan opsi $\Sigma$ ke format `.npz` terkompresi tanpa pickle (`allow_pickle=False`).
- `load_ood_artifact`: Memuat artefak `.npz` dengan verifikasi hash SHA-256 dan penolakan format yang tidak aman.
- `calculate_sha256`: Menghitung digest SHA-256 per chunk 64KB.
