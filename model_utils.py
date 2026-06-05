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
    Memastikan direktori dataset dan kelas bunga tersedia.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    for class_name in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)

def preprocess_audio(file_path_or_y, is_signal=False):
    """
    Pra-pemrosesan audio: load, trim silence, normalisasi, dan ekstraksi fitur MFCC + Delta.
    """
    try:
        if is_signal:
            y = file_path_or_y
        else:
            y, sr = librosa.load(file_path_or_y, sr=SR)
        
        # Validasi amplitudo minimum
        max_amp = np.max(np.abs(y)) if len(y) > 0 else 0
        if max_amp < 0.015:
            return None, None
            
        # Potong bagian hening
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Validasi panjang audio minimum setelah trim
        if len(y_trimmed) < 2400:
            return None, None
            
        # Normalisasi amplitudo
        y_trimmed = librosa.util.normalize(y_trimmed)
            
        # Standarkan durasi ke 1.5 detik
        target_len = int(DURATION * SR)
        if len(y_trimmed) < target_len:
            y_padded = np.pad(y_trimmed, (0, target_len - len(y_trimmed)), mode='constant')
        else:
            y_padded = y_trimmed[:target_len]
            
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y_padded, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
        
        # Penyesuaian frame length
        if mfcc.shape[1] < MAX_FRAMES:
            mfcc_fixed = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
        else:
            mfcc_fixed = mfcc[:, :MAX_FRAMES]
            
        # Ekstraksi Delta & Delta-Delta
        mfcc_delta = librosa.feature.delta(mfcc_fixed)
        mfcc_delta2 = librosa.feature.delta(mfcc_fixed, order=2)
        
        # Gabungkan fitur secara vertikal
        mfcc_combined = np.vstack([mfcc_fixed, mfcc_delta, mfcc_delta2])
            
        return mfcc_combined, y_padded
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

def generate_mfcc_plot(mfcc):
    """
    Membuat visualisasi spectrogram MFCC dalam format Base64 PNG.
    """
    try:
        mfcc_base = mfcc[:13, :] if mfcc.shape[0] == 39 else mfcc
        
        plt.figure(figsize=(5, 3))
        librosa.display.specshow(mfcc_base, sr=SR, hop_length=256, x_axis='time', cmap='magma')
        plt.axis('off')
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
    Melakukan augmentasi data audio (noise, time-stretch, pitch-shift).
    """
    augmented_signals = [y]
    
    # 1. Tambah White Noise
    try:
        noise = np.random.randn(len(y))
        y_noise = y + 0.004 * noise
        augmented_signals.append(y_noise)
    except Exception as e:
        print(f"Gagal augmentasi noise: {e}")
        
    # 2. Time-stretch (Lambat)
    try:
        y_slow = librosa.effects.time_stretch(y, rate=0.88)
        augmented_signals.append(y_slow)
    except Exception as e:
        print(f"Gagal augmentasi time-stretch lambat: {e}")
        
    # 3. Time-stretch (Cepat)
    try:
        y_fast = librosa.effects.time_stretch(y, rate=1.12)
        augmented_signals.append(y_fast)
    except Exception as e:
        print(f"Gagal augmentasi time-stretch cepat: {e}")
        
    # 4. Pitch-shifting (Tinggi)
    try:
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)
        augmented_signals.append(y_pitch_up)
    except Exception as e:
        print(f"Gagal augmentasi pitch-shift up: {e}")

    # 5. Pitch-shifting (Rendah)
    try:
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5)
        augmented_signals.append(y_pitch_down)
    except Exception as e:
        print(f"Gagal augmentasi pitch-shift down: {e}")
        
    return augmented_signals

def load_dataset(augment=True):
    """
    Memindai folder dataset dan memuat seluruh fitur audio serta label kelas.
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
                if file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                    file_path = os.path.join(class_dir, file_name)
                    
                    try:
                        y_raw, sr = librosa.load(file_path, sr=SR)
                        signals = augment_audio(y_raw, sr) if augment else [y_raw]
                            
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
    Melatih model SVM atau MLP dan menyimpannya ke dalam file .pkl.
    """
    X, y, stats = load_dataset(augment=True)
    
    if len(X) < 10:
        return {
            "success": False,
            "message": f"Dataset terlalu sedikit ({len(X)} sampel). Minimal butuh 10 sampel audio.",
            "stats": stats
        }
        
    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset (85% Latih, 15% Validasi)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.15, 
        random_state=42, 
        stratify=y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1 else None
    )
    
    # Inisialisasi model
    if model_type.lower() == "svm":
        model = SVC(kernel='rbf', C=1.5, probability=True, random_state=42)
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.15,
            max_iter=450,
            random_state=42
        )
        
    # Fitting model
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test) if len(X_test) > 0 else train_acc
    
    # Simpan hasil latih
    joblib.dump({"model": model, "model_type": model_type}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    success_msg = f"Berhasil melatih model {model_type.upper()} dengan total {stats['total']} sampel."
    
    return {
        "success": True,
        "message": success_msg,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "stats": stats
    }

def predict_audio(file_path):
    """
    Melakukan klasifikasi kelas bunga dari file input audio.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return {
            "success": False,
            "message": "Model belum dilatih."
        }
        
    try:
        mfcc, _ = preprocess_audio(file_path)
        if mfcc is None:
            return {
                "success": False,
                "message": "Audio input terlalu sunyi atau kosong. Harap coba merekam kembali."
            }
            
        mfcc_image = generate_mfcc_plot(mfcc)
        
        saved_data = joblib.load(MODEL_PATH)
        model = saved_data["model"]
        scaler = joblib.load(SCALER_PATH)
        
        features = mfcc.flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)
        
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
        print(f"Error saat prediksi: {e}")
        return {
            "success": False,
            "message": f"Terjadi kesalahan saat klasifikasi: {str(e)}"
        }

ensure_directories()

