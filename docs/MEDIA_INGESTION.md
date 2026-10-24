# Media Ingestion & Resource Bounding (`forgeoutlier.media`)

Modul `forgeoutlier.media` adalah Layer 2B yang bertanggung jawab atas decoding media citra dan streaming video secara aman, berbatas sumber daya (*resource-bounded*), anti-decompression-bomb, dan tanpa penulisan frame ke disk (*zero-disk retention*).

## 1. Komponen Utama

### A. Image Loader (`media.image.load_image`)
- **Proteksi Decompression Bomb**: Mengatur `Image.MAX_IMAGE_PIXELS = MEDIA_LIMITS.max_image_pixels` (default 16.777.216 piksel) untuk mencegah eksploitasi Denial-of-Service via gambar kompresi ekstrem.
- **Validasi Batas**: Memverifikasi resolusi maksimal $4096 \times 4096$ dan ukuran berkas maksimal 50 MB (`ResourceLimitExceededError`).
- **Sanitasi Path & EXIF Transpose**: Menggunakan `sanitize_input_path` dan auto-orientasi EXIF via `ImageOps.exif_transpose`.
- **Output Seragam**: Mengembalikan array NumPy float32 bernilai $[0.0, 1.0]$ dengan format shape `(3, H, W)` dan `MediaMeta`.

### B. Streaming Video (`media.video.VideoStream` & `extract_video_frames`)
- **Zero Disk Buffer**: Membaca frame video langsung dari kontainer (mis. MP4) ke memori in-flight melalui OpenCV `cv2.VideoCapture`.
- **Uniform Sampling**: Mengambil $N$ frame berjarak linier ekuidistan (`np.linspace`) melintasi durasi video (default 16 frame, batas maksimal 1000 frame).
- **Context Manager**: `VideoStream` mengimplementasikan context manager `__enter__` dan `__exit__` untuk pelepasan sumber daya (*resource cleanup*) instan tanpa memori bocor.
- **Generator**: Method `.frames()` menghasilkan pasangan `(index, frame_chw)` satu per satu demi menjaga konsumsi RAM $\le 1.5$ GB pada CPU.

### C. Strict Face Cropping (`media.face_crop.crop_face`)
- **Anti-Silent Fallback**: Jika deteksi wajah gagal atau berkas tidak memuat wajah yang valid, langsung melempar `ExecutionFailureError("NO_FACE_DETECTED")`. Tidak melakukan fallback diam-diam ke full image.
- **Detektor Plug-and-Play**: Mendukung injeksi custom detector callback (`detector_fn`) serta default Haar cascade extractor berbatas margin proporsional (default 20%).
- **Resizing & Normalisasi**: Menghasilkan crop berdimensi target `(3, 224, 224)` float32 beserta koordinat piksel bounding box asli `(x, y, w, h)`.

## 2. Kontrak Kesalahan
- `SecurityIntegrityError`: Terjadi jika path berkas mengandung null-byte atau traversal directory berbahaya.
- `CorruptedMediaError`: Terjadi jika berkas tidak ada, bukan citra/video yang valid, atau tidak dapat didecode oleh backend.
- `ResourceLimitExceededError`: Terjadi jika batas ukuran berkas, dimensi piksel, atau batas frame terlewati.
- `ExecutionFailureError`: Terjadi saat operasi deteksi wajah spesifik gagal menemukan wajah target.
