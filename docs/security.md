# Keamanan Operasional & Hardening Runtime ForgeOutlier

Dokumen ini mendokumentasikan postur keamanan zero-trust, mitigasi ancaman injeksi artefak, dan kontrol integritas runtime pada ForgeOutlier.

---

## 1. Threat Model & Vektor Serangan

ForgeOutlier didesain untuk dijalankan pada lingkungan server dan edge yang memproses berkas media pihak ketiga (*untrusted input*). Vektor ancaman yang dimitigasi meliputi:

1. **Remote Code Execution (RCE) via Serialization**: Penyerang menyisipkan bytecode berbahaya dalam bobot PyTorch atau profil artefak OOD (`.pt`, `.npz`).
2. **Decompression Bomb Denial-of-Service**: Penyerang mengunggah citra mikro berdimensi raksasa (misal $100000 \times 100000$ piksel) yang menghabiskan memori heap sistem saat diekspansi.
3. **Path Traversal / Arbitrary File Read**: Penyerang memanipulasi parameter jalur berkas input untuk membaca berkas sensitif OS.
4. **Data Exfiltration / Network Snooping**: Komponen runtime diam-diam menghubungi telemetry server pihak ketiga.

---

## 2. Kontrol Keamanan Terverifikasi

### 2.1 Deserialisasi Bobot Aman (`weights_only=True`)
Pemuatan checkpoint PyTorch (`src/forgeoutlier/models/loader.py`) menggunakan:
```python
torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```
Ini membatasi unpickler Python hanya pada struktur tensor primitif dan memblokir instansiasi objek arbitrer yang dapat mengeksekusi shellcode.

### 2.2 Deserialisasi Artefak OOD Aman (`allow_pickle=False`)
Profil OOD disimpan dan dimuat menggunakan format NumPy NPZ terkompresi:
```python
np.load(file_path, allow_pickle=False)
```
Integritas profil diverifikasi dengan hash SHA-256 sebelum pemuatan ke memori runtime.

### 2.3 Proteksi Dekompresi Bom Citra
Modul pemrosesan citra mengaktifkan batasan maksimum piksel Pillow:
```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = 178956970  # Proteksi DecompressionBombError
```
Jika citra melebihi batas atau rusak (*corrupted payload*), exception `CorruptMediaError` ditangkap dan ditangani secara graceful tanpa *crash*.

### 2.4 Isolasi Jaringan Penuh (Zero-Network Invariant)
Seluruh pipeline inferensi, ekstraksi fitur, dan evaluasi beroperasi 100% lokal tanpa dependensi cloud. Tidak ada panggilan HTTP, DNS resolution, atau pengunduhan bobot otomatis di runtime.
