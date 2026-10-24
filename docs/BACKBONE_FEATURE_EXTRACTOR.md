# Layer 2C: Backbone Feature Extractor & Secure Checkpoint Loader

## 1. Overview
Layer 2C merealisasikan arsitektur backbone deepfake feature extractor dan modul pemuatan bobot model berbasis standar keamanan zero-trust.
Seluruh komponen dibangun secara native berbasis PyTorch CPU-first murni tanpa ketergantungan paket eksternal `torchvision` guna menjamin kompatibilitas deterministik di lingkungan offline.

## 2. Implementasi Komponen
1. **Backbone (`forgeoutlier.models.backbone.MobileNetV3SmallBackbone`)**:
   - Struktur InvertedResidual dengan Squeeze-and-Excitation (SE) dan Hardswish activation.
   - Pengekstraksian representasi pre-logits pooling (default $D=576$, opsional $D=1024$).
   - Eksekusi deterministik pada mode evaluasi (`eval()`, `torch.no_grad()`).

2. **Classification Head (`forgeoutlier.models.classifier.BinaryClassifierHead`)**:
   - Linear projection dari feature embedding pre-logits ke 2 kelas output (`REAL`: 0, `FAKE`: 1).

3. **Integrated Model (`forgeoutlier.models.model.DeepfakeModel`)**:
   - Menggabungkan backbone dan classifier head.
   - Mengimplementasikan `EncoderProtocol` (`extract_features(x)` dan properti `feature_dim`).
   - Menyediakan `predict_probabilities(x)` untuk klasifikasi softmax biner.
   - Sanitasi array numerik terhadap nilai `NaN` dan `Inf`.

4. **Secure Checkpoint Loader (`forgeoutlier.models.loader.load_checkpoint`)**:
   - Sanitasi path input file via `sanitize_input_path` (mencegah traversal dan null-byte injection).
   - Validasi kriptografis SHA-256 terhadap manifest tepercaya (`trusted_checkpoints.json`).
   - Deserialisasi strict via `torch.load(..., weights_only=True, map_location="cpu")` (mencegah arbitrary code execution via pickle vulnerability D014/D025/D039).
   - Lempar `SecurityIntegrityError` saat terjadi mismatch hash atau korupsi file.

## 3. Status Verifikasi
- Pengujian Unit: `tests/unit/test_models.py` (6 skenario uji).
- Hasil Test Suite Total: 24/24 passed (100% lulus dalam 13.55s).
