# Panduan Instalasi ForgeOutlier

Dokumen ini menjelaskan instalasi ForgeOutlier pada berbagai lingkungan platform dengan prinsip minimalisme dependensi CPU.

---

## 1. Persyaratan Sistem

- **Sistem Operasi**: Windows 10/11 (x64), Linux (Ubuntu 20.04+, Debian, RHEL), macOS (Apple Silicon / Intel).
- **Python**: Python 3.10 hingga 3.14.
- **Hardware**: CPU standard dengan dukungan instruksi SSE4/AVX (GPU tidak diwajibkan).
- **RAM**: Minimal 1 GB (Peak runtime $\le 500\text{ MB}$).

---

## 2. Instalasi PyTorch CPU

Untuk menjaga ukuran image/lingkungan tetap ramping tanpa bloat CUDA binaries (yang berukuran >2 GB), instal PyTorch versi CPU:

```bash
# Linux / Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu

# macOS (sudah termasuk akselerasi CPU/MPS)
pip install torch
```

---

## 3. Instalasi ForgeOutlier

### 3.1 Instalasi dari Source Code (Mode Pengembang)
```bash
git clone https://github.com/forgeoutlier/forgeoutlier.git
cd forgeoutlier
pip install -e .
```

### 3.2 Instalasi dari Berkas Wheel (`.whl`)
```bash
pip install dist/forgeoutlier-1.0.0-py3-none-any.whl
```

---

## 4. Verifikasi Instalasi

Jalankan perintah verifikasi CLI untuk memastikan instalasi berhasil:
```bash
forgeoutlier --help
```
Output konfirmasi akan menampilkan subcommand yang tersedia:
- `predict`
- `batch-predict`
- `fit-ood`
- `inspect-model`
- `extract-embeddings`
- `evaluate`
