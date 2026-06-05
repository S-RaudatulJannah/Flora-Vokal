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

app = FastAPI(title="FloraVokal ASR & TTS Dashboard", version="1.0.0")

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
def startup_event():
    """
    Melatih/menyesuaikan model saat startup server.
    """
    print("Startup: Menjalankan training model otomatis...")
    try:
        res = model_utils.train_model("svm")
        print("Training Status:", res["message"])
    except Exception as e:
        print("Gagal melatih model saat startup:", e)

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>Server Running</h2><p>Halaman static/index.html tidak ditemukan.</p>"
    )

@app.post("/api/asr/predict")
async def predict_audio_file(file: UploadFile = File(...)):
    """
    Menerima input audio mentah, konversi ke WAV 16kHz Mono dengan FFmpeg,
    lalu lakukan prediksi kelas bunga.
    """
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".webm"
    
    raw_filename = f"raw_{int(time.time())}{ext}"
    raw_path = os.path.join(TEMP_DIR, raw_filename)
    
    wav_filename = f"converted_{int(time.time())}.wav"
    wav_path = os.path.join(TEMP_DIR, wav_filename)
    
    try:
        # Simpan file input mentah
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Konversi ke WAV (16kHz, mono, 16-bit PCM) menggunakan FFmpeg
        import static_ffmpeg
        static_ffmpeg.add_paths()
        
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise Exception("Konversi audio ke WAV gagal.")
            
        # Prediksi audio
        result = model_utils.predict_audio(wav_path)
        
        # Bersihkan temporary files
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return result
    except Exception as e:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts")
async def text_to_speech(
    text: str = Query(..., description="Teks Bahasa Indonesia"),
    gender: str = Query("female", description="Gender: male atau female"),
    speed: str = Query("normal", description="Kecepatan: slow, normal, atau fast")
):
    """
    Sintesis teks menjadi file audio MP3 secara real-time.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")
        
    filename = f"tts_{int(time.time())}_{gender}_{speed}.mp3"
    output_path = os.path.join(OUTPUTS_DIR, filename)
    
    try:
        await tts_utils.generate_speech(text, gender, speed, output_path)
        return {
            "success": True,
            "audio_url": f"/static/outputs/{filename}",
            "text": text,
            "gender": gender,
            "speed": speed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

