import asyncio
import edge_tts

async def generate_speech(text, gender="female", speed="normal", output_path="output.mp3"):
    """
    Mengubah teks Bahasa Indonesia menjadi berkas audio suara alami (Neural) 
    menggunakan layanan Microsoft Edge TTS secara asinkron.
    
    Parameters:
    - text (str): Teks kalimat Bahasa Indonesia yang ingin disuarakan.
    - gender (str): Pilihan gender pengisi suara ('female' untuk suara Gadis, 'male' untuk suara Ardi).
    - speed (str): Kecepatan bicara ('slow' untuk lambat, 'normal' untuk sedang, 'fast' untuk cepat).
    - output_path (str): Path lokasi tempat berkas audio .mp3 hasil sintesis akan disimpan.
    """
    # 1. Tentukan nama pengisi suara (Voice Model) berdasarkan gender yang dipilih
    if gender.lower() == "male":
        voice_name = "id-ID-ArdiNeural"       # Suara Pria (Ardi)
    else:
        voice_name = "id-ID-GadisNeural"     # Suara Wanita (Gadis)
        
    # 2. Tentukan persentase kecepatan bicara (Speech Rate)
    if speed.lower() == "slow":
        rate_str = "-35%"   # Diperlambat 35% agar perbedaan tempo terdengar sangat jelas
    elif speed.lower() == "fast":
        rate_str = "+40%"   # Dipercepat 40% agar tempo cepat terdengar tegas dan alami
    else:
        rate_str = "+0%"    # Kecepatan normal standar
        
    # 3. Buat objek Communicate dari edge-tts untuk merangkai teks & suara
    communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate_str)
    
    # 4. Jalankan asinkron untuk menyintesis teks dan menyimpan ke file output (.mp3)
    await communicate.save(output_path)
