# FloraVokal: ASR & TTS Flora Dashboard

FloraVokal adalah aplikasi web interaktif untuk klasifikasi suara ucapan nama bunga Indonesia (Automatic Speech Recognition - ASR) secara real-time dan sintesis suara alami (Text-to-Speech - TTS)

## Fitur Utama

- **Real-Time ASR**: Merekam ucapan nama bunga melalui mikrofon browser dan mengklasifikasikannya ke dalam 10 kategori bunga menggunakan model klasifikasi (SVM).
- **MFCC Visualizer**: Menampilkan sidik jari akustik suara berupa matriks 2D Mel-Frequency Cepstral Coefficients (MFCC) dalam bentuk heatmap.
- **Neural TTS**: Mengubah teks Bahasa Indonesia menjadi ucapan suara alami (pria/wanita) dengan kontrol tempo kecepatan menggunakan Microsoft Edge TTS.
- **Integrasi ASR ➔ TTS**: Menghubungkan hasil prediksi ASR secara langsung untuk disintesis kembali oleh modul TTS secara otomatis.

## Kategori Bunga yang Didukung

Aplikasi ini mendukung klasifikasi untuk 10 jenis bunga Indonesia berikut:
`alamanda`, `anggrek`, `dahlia`, `krisan`, `lily`, `matahari`, `mawar`, `melati`, `teratai`, `tulip`.

## Persyaratan Sistem

- Python 3.8+
- FFmpeg (untuk konversi format audio)

## Instalasi

1. **Clone Repositori**:
   ```bash
   git clone <url-repositori>
   cd Tubes-PTU
   ```

2. **Instal Dependensi**:
   Instal pustaka Python yang diperlukan:
   ```bash
   pip install fastapi uvicorn librosa soundfile numpy scikit-learn joblib matplotlib edge-tts static-ffmpeg
   ```

3. **Inisialisasi FFmpeg**:
   Aplikasi menggunakan `static-ffmpeg` untuk mengelola pustaka FFmpeg secara otomatis. Pastikan dependensi terpasang dengan benar.

## Cara Menjalankan Aplikasi

1. **Jalankan Server**:
   ```bash
   python app.py
   ```

2. **Akses Dashboard**:
   Buka browser Anda dan akses halaman:
   ```text
   http://127.0.0.1:8000
   ```

## Struktur Proyek

```text
├── app.py                # Server backend FastAPI & routing API
├── model_utils.py        # Pengolahan audio, ekstraksi fitur (MFCC), & klasifikasi ML
├── tts_utils.py          # Modul Text-to-Speech (Edge-TTS)
├── test_model.py         # Skrip pengujian mandiri model klasifikasi
├── model.pkl             # Model klasifikasi yang terlatih
├── scaler.pkl            # StandardScaler untuk normalisasi fitur
├── dataset/              # Folder dataset audio latih per kategori bunga
├── static/               # File aset frontend (HTML, CSS, JS)
└── temp/                 # Folder sementara untuk konversi audio
```
