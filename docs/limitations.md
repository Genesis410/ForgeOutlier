# Batasan Sistem & Jaminan Operasional (Limitations)

Dokumen ini mendefinisikan batasan operasional dan kondisi batas (*edge cases*) sistem deteksi ForgeOutlier.

---

## 1. Batasan Deteksi Wajah & Kualitas Media

1. **Resolusi Wajah Minimum**:
   - Model dilatih dan dioptimalkan pada area crop wajah dengan resolusi minimal $112 \times 112$ piksel.
   - Citra wajah dengan resolusi di bawah batas ini dapat menghasilkan representasi embedding dengan noise tinggi.
2. **Oklusi Berat & Pose Ekstrem**:
   - Wajah dengan sudut rotasi yaw/pitch $> 60^\circ$ atau oklusi wajah $> 50\%$ (masker tebal, kacamata hitam gelap, tangan menutupi wajah) dapat memicu peningkatan jarak Mahalanobis dan dialihkan ke `UNKNOWN`.
3. **Pencahayaan Ekstrem**:
   - Citra dengan underexposure parah (hampir hitam total) atau overexposure (washed out) menghasilkan fitur terdistorsi yang dideteksi sebagai Out-of-Distribution.

---

## 2. Batas Jaminan Deteksi OOD

1. **Serangan Adversarial Tertarget (*White-Box Perturbation*)**:
   - Meskipun OOD layer menggunakan jarak Mahalanobis pada feature space, serangan gradien white-box terarah yang secara eksplisit meminimalkan jarak Mahalanobis ke $\boldsymbol{\mu}_{ID}$ dapat mengelabui gate.
2. **Ketergantungan Representasi Backbone**:
   - Lapisan OOD hanya dapat mendeteksi keanehan pada domain yang berbeda dalam representasi MobileNetV3-Small. Jika manipulasi baru memiliki representasi fitur yang identik dengan citra wajah asli, sistem bergantung sepenuhnya pada classifier biner.

---

## 3. Ketentuan Penggunaan Bertanggung Jawab

ForgeOutlier didesain sebagai alat triage bantu (*human-in-the-loop assistance*). Keputusan otomatis sistem berstatus `UNKNOWN` menandakan data tidak dapat dipercaya untuk klasifikasi biner dan harus dialihkan ke peninjauan manual manusia.
