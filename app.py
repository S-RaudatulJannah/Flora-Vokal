import os
import shutil
import time
import subprocess
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import model_utils
import tts_utils

app = FastAPI(title="Taman Suara ASR Web Sederhana", version="1.0.0")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Memastikan folder statis, outputs, dan temp tersedia
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount folder static agar file HTML, JS, dan CSS dapat diakses publik oleh browser
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
def startup_event():
    """
    Melatih model klasifikasi secara otomatis pada saat startup server
    agar model, scaler, dan fitur selalu 100% sinkron secara sempurna.
    """
    print("=" * 60)
    print("Memulai pelatihan model ASR otomatis pada startup...")
    try:
        res = model_utils.train_model("svm")
        print("Status Pelatihan:", res["message"])
    except Exception as e:
        print("Gagal melatih model otomatis saat startup:", e)
    print("=" * 60)

@app.get("/", response_class=HTMLResponse)
def get_index():
    """
    Menyajikan halaman web utama ASR sederhana.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>Server Berjalan!</h2><p>Halaman index.html belum terbuat di folder static/.</p>"
    )

@app.post("/api/asr/predict")
async def predict_audio_file(file: UploadFile = File(...)):
    """
    Menerima file rekaman audio WebM/OGG/WAV dari browser microphone,
    menyimpannya ke folder temp, mengonversinya secara eksplisit ke berkas .wav 16kHz Mono
    menggunakan FFmpeg, lalu mengklasifikasikan ucapan nama bunga secara real-time.
    """
    # 1. Dapatkan ekstensi berkas mentah dari browser
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".webm"
    
    # Path untuk menyimpan berkas audio mentah (.webm atau .ogg)
    raw_filename = f"raw_{int(time.time())}{ext}"
    raw_path = os.path.join(TEMP_DIR, raw_filename)
    
    # Path untuk menyimpan berkas audio WAV hasil konversi bersih
    wav_filename = f"converted_{int(time.time())}.wav"
    wav_path = os.path.join(TEMP_DIR, wav_filename)
    
    try:
        # Simpan berkas audio mentah dari client
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Jalankan konversi eksplisit ke format WAV (16kHz, mono, PCM 16-bit) menggunakan FFmpeg
        import static_ffmpeg
        static_ffmpeg.add_paths() # Pastikan biner FFmpeg dari static-ffmpeg terdaftar di PATH
        
        # Susun argumen command line FFmpeg
        cmd = [
            "ffmpeg", "-y",             # Overwrite file jika sudah ada
            "-i", raw_path,             # File input mentah
            "-ar", "16000",             # Resample ke 16000 Hz
            "-ac", "1",                 # Downmix ke mono (1 channel)
            "-c:a", "pcm_s16le",        # Codec PCM 16-bit WAV standar
            wav_path                    # Path file output hasil konversi
        ]
        
        # Jalankan FFmpeg secara sinkron di background
        result_conv = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Pastikan file WAV hasil konversi sukses terbuat dan memiliki ukuran byte
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise Exception("Konversi audio ke WAV gagal dilakukan oleh FFmpeg.")
            
        # 3. Panggil fungsi klasifikasi suara murni dari model_utils pada berkas WAV bersih!
        result = model_utils.predict_audio(wav_path)
        
        # 4. Hapus berkas sampah sementara (mentah & hasil konversi) demi efisiensi storage
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return result
    except Exception as e:
        # Bersihkan berkas temp jika terjadi crash/eror
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise HTTPException(status_code=500, detail=f"Gagal memproses dan mengonversi audio: {str(e)}")

@app.get("/api/tts")
async def text_to_speech(
    text: str = Query(..., description="Teks yang ingin disintesis menjadi suara"),
    gender: str = Query("female", description="Gender suara: male atau female"),
    speed: str = Query("normal", description="Kecepatan bicara: slow, normal, atau fast")
):
    """
    Sintesis teks Bahasa Indonesia menjadi berkas audio suara alami secara real-time.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")
        
    # Buat nama berkas MP3 yang unik berdasarkan timestamp dan parameter
    filename = f"tts_{int(time.time())}_{gender}_{speed}.mp3"
    output_path = os.path.join(OUTPUTS_DIR, filename)
    
    try:
        # Jalankan sintesis suara neural secara asinkron lewat tts_utils
        await tts_utils.generate_speech(text, gender, speed, output_path)
        
        # Kembalikan URL publik berkas suara
        return {
            "success": True,
            "audio_url": f"/static/outputs/{filename}",
            "text": text,
            "gender": gender,
            "speed": speed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan Text-to-Speech: {str(e)}")

if __name__ == "__main__":
    # Jalankan server lokal di alamat 127.0.0.1 port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
