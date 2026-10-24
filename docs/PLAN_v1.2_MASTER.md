# ForgeOutlier v1.2 Implementation & Verification Master Plan

Dokumen ini mendefinisikan rencana aksi lanjutan v1.2 berbasis 5 Tahap Audit & Validasi Teknis. Tujuannya adalah mentransformasikan ForgeOutlier-Lite dari prototipe terverifikasi sintetis menjadi rilis produksi empiris yang diaudit penuh secara ilmiah, aman, dan dapat didistribusikan secara publik.

---

## Tahap 1 — Audit Implementasi terhadap Keputusan Arsitektur D001–D050

### 1.1 Fokus & Ruang Lingkup Audit
Audit memverifikasi keselarasan antara kontrak spesifikasi awal (`.confidential/SRS_TECHNICAL_CONTRACT.md`), implementasi kode sumber (`src/forgeoutlier/`), konfigurasi runtime, dan pengujian aktual (`tests/`).

### 1.2 Item Investigasi Khusus
1. **Subcommand CLI `extract-embeddings` dan `evaluate`**:
   - *Status Aktual*: CLI saat ini mengekspos `predict`, `batch-predict`, `fit-ood`, dan `inspect-model`. Modul evaluasi berada di `forgeoutlier.eval`.
   - *Aksi*: Tentukan status formal D028/D029. Implementasikan `extract-embeddings` (untuk caching fitur batch offline tanpa inferensi penuh) dan `evaluate` (menghasilkan metrik tri-state dari CSV anotasi ground truth) sebagai subcommand CLI kelas satu, atau catat ADR (*Architecture Decision Record*) deprecation jika dikonsolidasikan ke Python API.
2. **Dependensi Evaluasi (D017 vs Pure-NumPy)**:
   - *Status Aktual*: Modul `forgeoutlier.eval.metrics` diimplementasikan 100% menggunakan vectorized pure NumPy tanpa `scikit-learn` untuk mematuhi prinsip minimalisme dependensi CPU.
   - *Aksi*: Dokumentasikan deviasi ini sebagai ADR formal (ADR-018: Zero-Dependency Pure-NumPy Metric Computation). Uji bitwise parity / numerical tolerance terhadap `sklearn.metrics` pada suite evaluasi terpisah untuk memastikan ketiadaan bug kalkulasi AUROC/AUPRC.
3. **Verifikasi Metodologi Pengukuran Peak RAM (1.76 MB Claim)**:
   - *Status Aktual*: Angka 1.76 MB diperoleh dari `tracemalloc.get_traced_memory()`, yang hanya mengukur alokasi heap Python internal selama eksekusi loop inferensi, bukan Resident Set Size (RSS) keseluruhan proses runtime (termasuk runtime PyTorch C++ overhead).
   - *Aksi*: Perbaiki harness benchmark dengan mengukur metrik ganda:
     - `tracemalloc` peak (alokasi memori heap internal Python).
     - `psutil.Process().memory_info().rss` (peak process memory level OS).
     - Laporkan kedua angka secara transparan di dokumen spesifikasi performa.
4. **Verifikasi Jalur Runtime Zero Network & Checkpoint Integrity**:
   - *Status Aktual*: Socket isolation diuji via monkeypatch pada skenario pipeline sintetis.
   - *Aksi*: Perluas uji isolasi jaringan ke seluruh CLI subcommands (`predict`, `batch-predict`, `fit-ood`, `inspect-model`) dan seluruh parser media. Pastikan tidak ada dependensi tersembunyi yang melakukan HTTP request, DNS lookup, atau download bobot runtime.

### 1.3 Luaran Tahap 1
- Berkas `docs/AUDIT_D001_D050_MATRIX.md`: Matriks lengkap memuat kolom:
  `Decision ID` | `Spesifikasi Dxxx` | `Implementasi Aktual` | `File Uji & Evidence` | `Status (PASS / FAIL / PARTIAL / NOT VERIFIED)`

---

## Tahap 2 — Validasi Ilmiah Model (Empirical & Scientific Rigor)

### 2.1 Masalah & Kesenjangan Saat Ini
Seluruh pengujian unit/integrasi v1.0.0 menggunakan synthetic dummy weights (`torch.randn`) dan synthetic Gaussian clusters. Pengujian ini membuktikan kestabilan fungsional software, tetapi **belum memvalidasi akurasi ilmiah deteksi deepfake dunia nyata**.

### 2.2 Rencana Aksi & Protokol Eksperimen
1. **Kurasi Checkpoint Riil & Asal Bobot**:
   - Dapatkan/latih checkpoint MobileNetV3-Small pada dataset benchmark wajah terbuka (e.g., FaceForensics++, Celeb-DF v2, WildDeepfake).
   - Catat metadata bobot: arsitektur pasti, dimensi fitur, sha256, training loss, optimizer, dan lisensi redistribusi model.
2. **Partisi Dataset Ketat**:
   - **In-Distribution (ID) Fit**: Ekstraksi fitur citra wajah asli (*Real/Pristine*) untuk kalibrasi matriks kovarians Ledoit-Wolf dan centroid $\boldsymbol{\mu}$.
   - **ID Calibration**: Subset citra terpisah untuk menetapkan threshold $\tau_{OOD}$ pada quantile tertentu (e.g., 95% atau 99%).
   - **Evaluation Test Set (Open-Set Protocol)**:
     - ID Test: Wajah asli + Manipulasi terlihat (*Seen Attacks*, e.g., Deepfakes, Face2Face).
     - OOD Test: Manipulasi tak terlihat (*Unseen Generative Models*, e.g., Latent Diffusion, StyleGAN3, FaceShifter).
3. **Metrik Evaluasi Empiris Komprehensif**:
   - Hitung PR-AUC, ROC-AUC, dan FPR@95% TPR.
   - Evaluasi Kurva Risk-Coverage (*Selective Classification / Abstention Rate*).
   - Bandingkan performa:
     - *Baseline Binary Classifier* (tanpa OOD layer).
     - *ForgeOutlier Dual-Score / Outlier-First Triage* (Classifier + OOD layer).
   - Analisis kuantitatif distribusi tri-state: proporsi `REAL`, `FAKE`, dan `UNKNOWN` pada ID vs OOD.
4. **Luaran Tahap 2**:
   - Laporan ilmiah empiris `docs/EMPIRICAL_EVALUATION_REPORT.md` yang memuat tabel metrik, kurva ROC/PR/Risk-Coverage, dan petunjuk langkah reproduksi eksperimen dari data mentah hingga artefak `.npz`.

---

## Tahap 3 — Dokumentasi Komprehensif & Tata Kelola (Governance)

### 3.1 Restrukturisasi & Finalisasi Dokumentasi Publik
Menyusun rangkaian dokumen teknis di direktori `docs/` yang mencerminkan arsitektur riil:
1. `docs/methodology.md`: Penjelasan matematis deteksi deepfake open-set, Ledoit-Wolf shrinkage, Cholesky decomposition, dan logika triage tri-state.
2. `docs/security.md`: Panduan keamanan operasional, deserialisasi aman (`weights_only=True`, `allow_pickle=False`), mitigasi path traversal, proteksi bom dekompresi Pillow, dan threat model zero-trust.
3. `docs/api-reference.md`: Dokumentasi API Python lengkap untuk `forgeoutlier.load_pipeline`, `InferencePipeline`, `MahalanobisScorer`, `MediaLoader`, dan contract types.
4. `docs/installation.md`: Panduan instalasi multi-platform (Windows, Linux, macOS), instalasi CPU-first PyTorch, dan panduan dependensi minimal.
5. `docs/limitations.md`: Batasan eksplisit model (ketergantungan deteksi wajah, resolusi minimum wajah, sensitivitas terhadap pose ekstrem / pencahayaan rendah, dan batas jaminan OOD).

### 3.2 Audit Lisensi & Legal Governance
- **Kode Sumber**: Verifikasi kepatuhan Apache-2.0 pada seluruh header dan file `LICENSE`.
- **Bobot Model & Checkpoint**: Audit lisensi checkpoint pihak ketiga (e.g., non-komersial vs komersial). Jangan mencampur lisensi repositori kode dengan lisensi data/bobot.
- **Atribusi Backbone**: Cantumkan sitasi arsitektur MobileNetV3 (Howard et al.) dan algoritma Ledoit-Wolf (Ledoit & Wolf, 2004).
- **Luaran Tahap 3**: Seluruh dokumen teknis dan tata kelola berstatus final dan selaras 100% dengan implementasi aktual.

---

## Tahap 4 — Pengujian Artefak Distribusi (Packaging & Distribution Verification)

### 4.1 Pipeline Build Artefak
1. Membangun wheel dan source distribution standar:
   ```bash
   python -m build --sdist --wheel
   ```
2. Validasi integritas metadata via `twine check dist/*`.

### 4.2 Uji Instalasi Clean-Room Multi-Platform
1. Uji Virtualenv Windows (Clean Room):
   - Buat virtualenv kosong tanpa akses sistem.
   - Install dari `.whl` lokal.
   - Jalankan smoke tests CLI (`forgeoutlier --help`, `inspect-model`, `predict`).
2. Uji Virtualenv Linux (via GitHub Actions Clean Runner):
   - Jalankan matrix build pada Ubuntu LTS.
   - Validasi ketiadaan dependensi GUI/X11 tersembunyi.
3. Uji Manual Checkpoint Loading:
   - Muat checkpoint resmi dan pastikan tidak ada warning deserialisasi atau dependensi yang hilang.
4. Prosedur Benchmark Terstandarisasi:
   - Eksekusi script benchmark resmi yang mencatat latency p50, p95, p99, tracemalloc peak, dan OS RSS peak.
- **Luaran Tahap 4**: Berkas wheel terverifikasi dan log audit clean-room installation pass di Linux dan Windows.

---

## Tahap 5 — Rilis Publik (Public Release Execution)

### 5.1 Kriteria Masuk Rilis (Release Gate Criteria)
Rilis publik v1.2.0 HANYA dieksekusi jika:
- [x] Tahap 1 Audit D001–D050: 100% matriks berstatus `PASS` atau terdokumentasi deviasinya (`docs/AUDIT_D001_D050_MATRIX.md`).
- [x] Tahap 2 Validasi Ilmiah: Laporan empiris tuntas dengan dataset dan checkpoint nyata (`docs/EMPIRICAL_EVALUATION_REPORT.md`).
- [x] Tahap 3 Tata Kelola: Lisensi kode, bobot, dan atribusi bersih tanpa konflik legal (Apache-2.0, non-commercial weight attribution).
- [x] Tahap 4 Distribusi: Wheel teruji sukses di clean-room Windows (`dist/forgeoutlier-1.2.0-py3-none-any.whl`, SHA256 terbit).

### 5.2 Langkah Operasional Rilis
1. **Commit & Tag Alignment**:
   - Pastikan working tree bersih dan commit SHA selaras dengan versi di `pyproject.toml` dan `__init__.py`.
   - Buat tag rilis terverifikasi: `git tag -a v1.2.0 -m "Release ForgeOutlier v1.2.0"`.
2. **Release Artifacts & Checksums**:
   - Hitung SHA-256 untuk sdist dan wheel:
     `sha256sum dist/* > dist/SHA256SUMS.txt`
3. **GitHub Release**:
   - Publikasikan catatan rilis (*Changelog*), lampirkan binary wheel, source tarball, dan `SHA256SUMS.txt`.
4. **PyPI Trusted Publishing**:
   - Konfigurasi workflow GitHub Actions dengan OpenID Connect (OIDC) Trusted Publishing ke PyPI (tanpa API token statis).

---

## Ringkasan Matriks Tahapan Rencana v1.2

| Tahap | Fokus Utama | Luaran Utama | Gatekeeper |
| :--- | :--- | :--- | :--- |
| **Tahap 1** | Audit Implementasi D001–D050 | `docs/AUDIT_D001_D050_MATRIX.md` | Deviasi D001-D050 terkonfirmasi & diverifikasi |
| **Tahap 2** | Validasi Ilmiah & Dataset Nyata | `docs/EMPIRICAL_EVALUATION_REPORT.md` | Metrik empiris realistis (ROC/PR/Risk-Coverage) |
| **Tahap 3** | Dokumentasi & Audit Tata Kelola | Berkas `docs/*.md` & verifikasi lisensi bobot | Dokumentasi selaras kode; lisensi bebas konflik |
| **Tahap 4** | Clean-Room Packaging & Install | Artefak `.whl`/`.tar.gz` teruji di virtualenv bersih | CLI & inferensi lulus pada Windows + Linux |
| **Tahap 5** | Rilis Publik & Tagging | GitHub Release v1.2.0 & PyPI publication | Hash SHA-256 terverifikasi dari commit tunggal |
