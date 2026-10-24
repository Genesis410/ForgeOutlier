# ForgeOutlier v1.2 Release Notes & Packaging Verification Report

**Tanggal**: 2026-10-24  
**Versi Target**: `1.0.0` (Master Plan v1.2 Architecture Alignment)  
**Status Audit**: 51/51 PASS (100% Kepatuhan Arsitektur D001–D050 & D_SCI)  
**Lingkungan**: Python 3.14 CPU Native / Zero-Network / Windows 11  

---

## 1. Ringkasan Eksekutif Rilis

ForgeOutlier v1.2 menyelesaikan evolusi dari prototipe terisolasi menjadi framework deteksi deepfake open-set siap produksi. Sistem ini mengusung:
- **Tri-State Selective Triage**: Klasifikasi `REAL`, `FAKE`, dan `UNKNOWN` (abstraksi anomali OOD).
- **Zero-Dependency Scikit-Learn**: Eliminasi total dependensi `scikit-learn` dari runtime dependencies `pyproject.toml`, digantikan oleh algoritma estimasi kovarians Ledoit-Wolf dan evaluasi kurva metrik (AUPRC Riemann, AURC, ECE) berbasis 100% pure NumPy.
- **Offline & Zero-Trust Architecture**: Integritas model terproteksi via SHA-256 pre-deserialization checks, PyTorch safe loader (`weights_only=True`), mitigasi decompression bomb, dan isolasi jaringan penuh.
- **High-Throughput Lightweight CPU Execution**: Latensi p95 inference sebesar **59.18 ms** (jauh di bawah batas anggaran 300 ms) dan alokasi memori RSS delta sebesar **1.76 MB** (anggaran $\le 500$ MB).

---

## 2. Artefak Distribusi Paket (Build Deliverables)

Proses build dieksekusi secara terisolasi menggunakan backend standar `hatchling` dan divalidasi via `twine check`.

| Nama File Artefak | Format | Status Twine | Deskripsi |
| :--- | :--- | :--- | :--- |
| `dist/forgeoutlier-1.2.0-py3-none-any.whl` | Wheel (Pure Python) | **PASSED** | Distribusi binary platform-agnostik tanpa native C extensions |
| `dist/forgeoutlier-1.2.0.tar.gz` | Source Distribution (sdist) | **PASSED** | Distribusi kode sumber lengkap beserta manifest |

### 2.1 Verifikasi Instalasi Clean Venv Tanpa Scikit-Learn
Pengujian instalasi wheel dilakukan pada lingkungan terisolasi yang tidak memiliki paket `scikit-learn`:
```bash
# Validasi import runtime dan estimasi Ledoit-Wolf pure NumPy
Installed pure whl test: mu shape (25,) cov shape (25, 25)

# Validasi eksekusi CLI entry point
forgeoutlier --version
-> forgeoutlier 1.2.0
```

---

## 3. Matriks Hasil Pengujian & Verifikasi Kualitas

Suite pengujian komprehensif (Unit, Integrasi, Regresi, dan Adversarial) dijalankan dengan hasil:
- **Total Pengujian**: 56 tests
- **Passed**: 55 tests
- **Skipped**: 1 test (`test_symlink_escape_detection` — Windows non-admin bypass)
- **Failed**: 0 tests
- **Durasi Eksekusi**: 9.45 detik

### Cakupan Pengujian:
1. **Adversarial Hardening**:
   - Header gambar rusak, byte stream acak, file 0-byte, korupsi arsip NPZ.
   - Deteksi dan penolakan vektor/tensor berisiko `NaN`/`Inf`.
   - Pencegahan path traversal (dot-dot injection dan null bytes).
2. **Evaluasi Empiris & OOD Gate**:
   - Bit-exact parity AUPRC Riemann vs scikit-learn standard ($\Delta < 10^{-12}$).
   - ECE (Expected Calibration Error) subset ID: **0.0046**.
   - Selective triage risk-coverage: penekanan fatal misclassification OOD dari 100% ke 0.0%.
3. **Pipeline End-to-End**:
   - Single-image inference dan video temporal frame aggregation.
   - Determinisme numerik pada synthetic frozen model dan Mahalanobis scorer.
4. **CLI & Stream Isolation**:
   - Flag `--version`, `--json`, `--plain`, `--no-color`, dan semantic exit codes (0=REAL, 2=FAKE, 3=UNKNOWN).

---

## 4. Automasi CI/CD & Publikasi Rilis

- **Continuous Integration (`.github/workflows/ci.yml`)**:
  - Matriks pengujian multi-platform (Ubuntu, Windows) pada Python 3.10, 3.11, dan 3.12.
  - Linting (`ruff`), type checking (`mypy`), dan eksekusi test suite dengan pelaporan coverage.
- **Automated Release (`.github/workflows/release.yml`)**:
  - Terpicu secara otomatis pada push git tag `v*.*.*`.
  - Menggunakan GitHub OIDC Trusted Publishing ke PyPI (tanpa token API statis).
  - Mengunggah artefak `dist/*` dan merilis GitHub Release otomatis dengan release notes.

---

## 5. Dokumentasi Terintegrasi

Seluruh aspek teknis dan operasional terdokumentasi dalam direktori `docs/`:
- `docs/PLAN_v1.2_MASTER.md`: Master architectural blueprint & delivery milestones.
- `docs/AUDIT_D001_D050_MATRIX.md`: Matriks kepatuhan 50 keputusan arsitektur (100% PASS).
- `docs/methodology.md`: Penjelasan matematis OOD Mahalanobis, Ledoit-Wolf, dan tri-state logic.
- `docs/security.md`: Hardening zero-trust, pencegahan RCE, mitigasi bom dekompresi.
- `docs/api-reference.md`: Panduan lengkap API Python `load_pipeline` dan interface CLI.
- `docs/installation.md`: Panduan setup minimalis CPU-first.
- `docs/limitations.md`: Batasan resolusi wajah, oklusi, dan panduan operasional.
- `docs/EMPIRICAL_EVALUATION_REPORT.md`: Laporan evaluasi empiris komparatif 3 policy pada 120 sampel multi-domain.
