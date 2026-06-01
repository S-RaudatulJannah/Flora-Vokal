import os
import io
import base64
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 10 official Indonesian flower classes in the dataset
CLASSES = [
    "alamanda",
    "anggrek",
    "dahlia",
    "krisan",
    "lily",
    "matahari",
    "mawar",
    "melati",
    "teratai",
    "tulip"
]

# Audio feature constants
SR = 16000           # Target sample rate (16 kHz mono)
DURATION = 1.5       # Target duration in seconds
N_MFCC = 13          # Number of MFCC coefficients to extract
MAX_FRAMES = 94      # Number of time frames for MFCC (1.5s / 256 hop_length = 94 frames)
N_FEATURES = 3 * N_MFCC * MAX_FRAMES  # 39 * 94 = 3666 total flattened features (MFCC + Delta + Delta-Delta)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

def ensure_directories():
    """
    Ensure the dataset directory and its 10 class subfolders exist.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    for class_name in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)

def preprocess_audio(file_path_or_y, is_signal=False):
    """
    Load an audio file or handle a raw signal, apply robust silence trimming (top_db=20),
    normalize peak amplitude to 1.0 (vital for different microphones),
    pad or truncate to exactly 1.5 seconds, and extract 39 dynamic features (MFCC + Delta + Delta-Delta).
    
    Returns:
    - mfcc_combined: 2D numpy array of shape (39, 94)
    - y_padded: 1D numpy array of the resampled/padded audio signal
    """
    try:
        if is_signal:
            y = file_path_or_y
        else:
            # Load audio file (librosa automatically resamples to 16kHz and downmixes to mono)
            y, sr = librosa.load(file_path_or_y, sr=SR)
        
        # Cek apakah sinyal suara kosong atau terlalu sunyi (indikasi mikrofon mati/silent)
        max_amp = np.max(np.abs(y)) if len(y) > 0 else 0
        if max_amp < 0.015:  # Di bawah 1.5% volume penuh
            print(f"DEBUG ASR: Audio terlalu sunyi / mikrofon tidak menangkap suara! (Amplitudo Maks: {max_amp:.6f})")
            return None, None
            
        # 1. Trim silence (top_db=20 is standard and handles room hum/fan noise robustly)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Jika setelah dipotong hening sinyal suara habis / terlalu pendek, tolak berkas
        if len(y_trimmed) < 2400:  # Kurang dari 0.15 detik suara aktif
            print(f"DEBUG ASR: Sinyal suara habis setelah trimming silence! (Panjang: {len(y_trimmed)} sampel)")
            return None, None
            
        # 2. Peak Amplitude Normalization (vital to standardise volume differences)
        y_trimmed = librosa.util.normalize(y_trimmed)
            
        # 3. Standardise duration to exactly 1.5 seconds (24000 samples at 16kHz)
        target_len = int(DURATION * SR)
        if len(y_trimmed) < target_len:
            # Pad with silent zeros
            y_padded = np.pad(y_trimmed, (0, target_len - len(y_trimmed)), mode='constant')
        else:
            # Truncate to target length
            y_padded = y_trimmed[:target_len]
            
        # 4. Extract MFCC features (hop_length=256 ensures high temporal resolution)
        mfcc = librosa.feature.mfcc(y=y_padded, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
        
        # 5. Lock dimensions to exactly MAX_FRAMES (94)
        if mfcc.shape[1] < MAX_FRAMES:
            mfcc_fixed = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
        else:
            mfcc_fixed = mfcc[:, :MAX_FRAMES]
            
        # 6. Extract Delta (speed) and Delta-Delta (acceleration) spectral dynamics
        mfcc_delta = librosa.feature.delta(mfcc_fixed)
        mfcc_delta2 = librosa.feature.delta(mfcc_fixed, order=2)
        
        # Stack vertically to get shape (39, 94)
        mfcc_combined = np.vstack([mfcc_fixed, mfcc_delta, mfcc_delta2])
            
        return mfcc_combined, y_padded
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

def generate_mfcc_plot(mfcc):
    """
    Plot the base MFCC spectrogram matrix (first 13 coefficients) as a Base64 PNG.
    """
    try:
        # Only plot the first 13 base MFCC coefficients for visual clarity
        mfcc_base = mfcc[:13, :] if mfcc.shape[0] == 39 else mfcc
        
        plt.figure(figsize=(5, 3))
        # Display MFCC heatmap using magma color space (beautiful pastel look)
        librosa.display.specshow(mfcc_base, sr=SR, hop_length=256, x_axis='time', cmap='magma')
        plt.axis('off')  # Clean layout with no coordinates/axes
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
        plt.close()
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error generating MFCC plot: {e}")
        return ""

def augment_audio(y, sr):
    """
    Melakukan augmentasi data audio untuk mencegah overfitting pada dataset kecil.
    Menghasilkan 6 variasi audio yang merepresentasikan kondisi dunia nyata:
    1. Sinyal asli yang bersih.
    2. Subtle White Noise: Simulasi derau ruangan (noise latar belakang).
    3. Slow Stretch: Simulasi pengucapan kata yang lambat.
    4. Fast Stretch: Simulasi pengucapan kata yang cepat.
    5. Pitch Shift Up: Simulasi suara bernada lebih tinggi (misal: perempuan).
    6. Pitch Shift Down: Simulasi suara bernada lebih rendah (misal: laki-laki).
    """
    augmented_signals = [y]
    
    # 1. Menambahkan subtle white noise
    try:
        noise = np.random.randn(len(y))
        y_noise = y + 0.004 * noise
        augmented_signals.append(y_noise)
    except Exception as e:
        print(f"Gagal augmentasi noise: {e}")
        
    # 2. Time-stretching (lambat: rate=0.88)
    try:
        y_slow = librosa.effects.time_stretch(y, rate=0.88)
        augmented_signals.append(y_slow)
    except Exception as e:
        print(f"Gagal augmentasi time-stretch lambat: {e}")
        
    # 3. Time-stretching (cepat: rate=1.12)
    try:
        y_fast = librosa.effects.time_stretch(y, rate=1.12)
        augmented_signals.append(y_fast)
    except Exception as e:
        print(f"Gagal augmentasi time-stretch cepat: {e}")
        
    # 4. Pitch-shifting (+1.5 semitones - nada tinggi)
    try:
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)
        augmented_signals.append(y_pitch_up)
    except Exception as e:
        print(f"Gagal augmentasi pitch-shift up: {e}")

    # 5. Pitch-shifting (-1.5 semitones - nada rendah)
    try:
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5)
        augmented_signals.append(y_pitch_down)
    except Exception as e:
        print(f"Gagal augmentasi pitch-shift down: {e}")
        
    return augmented_signals

def load_dataset(augment=True):
    """
    Memindai folder dataset bunga, mengekstrak fitur MFCC untuk setiap kelas bunga,
    dan opsional menerapkan augmentasi data untuk memperbanyak jumlah sampel latih.
    
    Returns:
    - X: Array fitur berukuran (N_samples, 1222)
    - y: Array label integer (0 hingga 9)
    - stats: Statistik jumlah sampel per kategori
    """
    ensure_directories()
    
    X = []
    y = []
    stats = {c: 0 for c in CLASSES}
    stats["total"] = 0
    stats["failed"] = 0
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if os.path.exists(class_dir):
            for file_name in os.listdir(class_dir):
                # Membaca file audio dengan ekstensi umum
                if file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                    file_path = os.path.join(class_dir, file_name)
                    
                    try:
                        # Load audio asli menggunakan librosa
                        y_raw, sr = librosa.load(file_path, sr=SR)
                        
                        # Terapkan augmentasi data jika diset True
                        if augment:
                            signals = augment_audio(y_raw, sr)
                        else:
                            signals = [y_raw]
                            
                        # Ekstrak fitur MFCC untuk setiap sinyal audio
                        for sig in signals:
                            mfcc, _ = preprocess_audio(sig, is_signal=True)
                            if mfcc is not None:
                                X.append(mfcc.flatten())
                                y.append(label_idx)
                                stats[class_name] += 1
                                stats["total"] += 1
                            else:
                                stats["failed"] += 1
                                
                    except Exception as e:
                        print(f"Gagal memuat file {file_name}: {e}")
                        stats["failed"] += 1
                        
    return np.array(X), np.array(y), stats

def train_model(model_type="mlp"):
    """
    Melatih model klasifikasi (MLP Neural Network atau SVM) menggunakan fitur audio teraugmentasi.
    Menyimpan model terlatih dan StandardScaler ke dalam file .pkl.
    """
    # 1. Load dataset dengan data augmentation
    X, y, stats = load_dataset(augment=True)
    
    if len(X) < 10:
        return {
            "success": False,
            "message": f"Dataset terlalu sedikit ({len(X)} sampel). Minimal butuh 10 sampel audio untuk melatih model.",
            "stats": stats
        }
        
    # 2. Standardisasi Fitur menggunakan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Bagi data menjadi 85% Latih dan 15% Validasi (stratified split untuk keadilan kelas)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.15, 
        random_state=42, 
        stratify=y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1 else None
    )
    
    # 4. Inisialisasi Model Klasifikasi
    if model_type.lower() == "svm":
        # SVM dengan Kernel RBF yang robust untuk dataset berdimensi menengah
        model = SVC(kernel='rbf', C=1.5, probability=True, random_state=42)
    else:
        # MLP Neural Network yang diatur alpha-nya (L2 regularization) demi mencegah overfitting
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.15,       # Penalti regularisasi yang kuat untuk mencegah menghafal data
            max_iter=450,     # Jumlah epoch latihan yang memadai untuk konvergensi Adam
            random_state=42
        )
        
    # 5. Pelatihan Model
    model.fit(X_train, y_train)
    
    # 6. Hitung Akurasi
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test) if len(X_test) > 0 else train_acc
    
    # 7. Simpan Model & Scaler ke file .pkl
    joblib.dump({"model": model, "model_type": model_type}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    success_msg = f"Berhasil melatih model {model_type.upper()} dengan total {stats['total']} sampel audio teraugmentasi (6x lipat dari dataset asli)!"
    
    return {
        "success": True,
        "message": success_msg,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "stats": stats
    }

def predict_audio(file_path):
    """
    Melakukan klasifikasi/prediksi real-time untuk sebuah file rekaman suara mic.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return {
            "success": False,
            "message": "Model belum dilatih. Silakan jalankan pelatihan model terlebih dahulu."
        }
        
    try:
        # 1. Pra-pemrosesan audio mic
        mfcc, _ = preprocess_audio(file_path)
        if mfcc is None:
            return {
                "success": False,
                "message": "Mikrofon Anda tidak mendeteksi suara dengan jelas (terlalu sunyi / kosong). Silakan periksa pengaturan mic Anda, posisikan mic lebih dekat, dan berikan jeda 0.5 detik sebelum Anda mulai berbicara!"
            }
            
        # 2. Gambar spektrogram 2D MFCC
        mfcc_image = generate_mfcc_plot(mfcc)
        
        # 3. Load Model dan Scaler
        saved_data = joblib.load(MODEL_PATH)
        model = saved_data["model"]
        scaler = joblib.load(SCALER_PATH)
        
        # 4. Standardisasi fitur input
        features = mfcc.flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # 5. Prediksi probabilitas kelas bunga
        probs = model.predict_proba(features_scaled)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        predicted_class = CLASSES[pred_idx]
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "mfcc_image": mfcc_image,
            "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        }
    except Exception as e:
        print(f"Gagal saat inferensi model: {e}")
        return {
            "success": False,
            "message": f"Terjadi kesalahan saat klasifikasi: {str(e)}"
        }

# Pastikan folder terbuat saat modul pertama kali diimpor
ensure_directories()
