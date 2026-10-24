# Metodologi Ilmiah & Arsitektur Tri-State ForgeOutlier

Dokumen ini menjelaskan fondasi matematika deteksi deepfake open-set, estimasi kovarians Ledoit-Wolf, dekomposisi Cholesky, serta logika selective triage tri-state (`REAL`, `FAKE`, `UNKNOWN`).

---

## 1. Masalah Deteksi Deepfake Open-Set

Pengklasifikasi deepfake biner konvensional mengasumsikan distribusi tertutup (*closed-world assumption*):
$$f: \mathcal{X} \to \{\text{Real}, \text{Fake}\}$$

Dalam penerapan dunia nyata, model dihadapkan pada masukan yang berada di luar distribusi pelatihan (*Out-of-Distribution* / OOD), seperti:
1. Citra non-wajah (anime, diagram, pemandangan, mikroskopi).
2. Model generatif unseen (*novel generative architectures*).
3. Degradasi visual ekstrem (noise sensor berat, kompresi destruktif).

Pada arsitektur biner standar, masukan OOD ini dipaksa menghasilkan prediksi biner dengan tingkat keyakinan softmax yang terdistorsi (*overconfident silent failure*).

---

## 2. Ekstraksi Fitur & Normalisasi L2

Model menggunakan MobileNetV3-Small sebagai ekstraktor fitur backbone. Vektor representasi $\mathbf{f} \in \mathbb{R}^{576}$ diekstraksi dari lapisan *penultimate* (setelah Global Average Pooling).

Untuk menstabilkan metrik jarak terhadap variasi magnitudo, diterapkan normalisasi L2:
$$\mathbf{z} = \frac{\mathbf{f}}{\max(\|\mathbf{f}\|_2, 10^{-12})}$$

Vektor ternormalisasi $\mathbf{z}$ berada pada hypersphere satuan $\mathbb{S}^{575}$.

---

## 3. Estimasi Kovarians Ledoit-Wolf Shrinkage

Jika jumlah sampel referensi In-Distribution $N$ lebih kecil dari dimensi fitur $D = 576$, matriks kovarians sampel empiris $\mathbf{S}$ menjadi *singular* atau *ill-conditioned*.

ForgeOutlier mengimplementasikan shrinkage Ledoit-Wolf analitis:
$$\mathbf{\Sigma}_{LW} = (1 - \delta)\mathbf{S} + \delta \nu \mathbf{I}$$

di mana:
- $\mathbf{S} = \frac{1}{N} \sum_{i=1}^N (\mathbf{z}_i - \boldsymbol{\mu})(\mathbf{z}_i - \boldsymbol{\mu})^T$
- $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{z}_i$
- $\nu = \frac{1}{D} \text{Tr}(\mathbf{S})$ (skalar dispersi rata-rata)
- $\delta \in [0, 1]$ adalah koefisien shrinkage optimal yang meminimalkan Frobenius loss $\mathbb{E}[\|\mathbf{\Sigma}_{LW} - \mathbf{\Sigma}\|_F^2]$.

Hasilnya adalah matriks kovarians $\mathbf{\Sigma}_{LW}$ yang *symmetric positive-definite* (SPD) dan selalu dapat diinvers.

---

## 4. Jarak Mahalanobis via Dekomposisi Cholesky

Jarak Mahalanobis antara fitur uji $\mathbf{z}$ dan centroid referensi $\boldsymbol{\mu}$ didefinisikan sebagai:
$$d_M(\mathbf{z}, \boldsymbol{\mu}) = \sqrt{(\mathbf{z} - \boldsymbol{\mu})^T \mathbf{\Sigma}_{LW}^{-1} (\mathbf{z} - \boldsymbol{\mu})}$$

Untuk menghindari inversi matriks langsung $\mathcal{O}(D^3)$ yang rentan terhadap instabilitas numerik, digunakan faktorisasi Cholesky:
$$\mathbf{\Sigma}_{LW} = \mathbf{L} \mathbf{L}^T$$

dengan $\mathbf{L}$ matriks segitiga bawah (*lower triangular*). Sistem persamaan diselesaikan via *forward substitution*:
$$\mathbf{L} \mathbf{y} = \mathbf{z} - \boldsymbol{\mu}$$

Jarak Mahalanobis dihitung langsung dari norm vektor terbobot $\mathbf{y}$:
$$d_M(\mathbf{z}, \boldsymbol{\mu}) = \|\mathbf{y}\|_2$$

---

## 5. Logika Triage Tri-State

Hasil inferensi dikategorikan ke dalam tiga status diskret:

$$\text{Decision}(\mathbf{z}) = \begin{cases}
\text{UNKNOWN}, & \text{jika } d_M(\mathbf{z}, \boldsymbol{\mu}) > \tau_{OOD} \\
\text{FAKE}, & \text{jika } d_M(\mathbf{z}, \boldsymbol{\mu}) \le \tau_{OOD} \text{ dan } P(\text{Fake} \mid \mathbf{z}) \ge \theta_{Fake} \\
\text{REAL}, & \text{jika } d_M(\mathbf{z}, \boldsymbol{\mu}) \le \tau_{OOD} \text{ dan } P(\text{Fake} \mid \mathbf{z}) < \theta_{Fake}
\end{cases}$$

Ambang batas $\tau_{OOD}$ dikalibrasi secara empiris dari kuantil $\alpha$ data In-Distribution (misal persentil ke-95), sehingga menjamin false alarm rate terkontrol.
