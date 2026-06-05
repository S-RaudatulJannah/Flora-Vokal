import asyncio
import edge_tts

async def generate_speech(text, gender="female", speed="normal", output_path="output.mp3"):
    """
    Mengubah teks menjadi suara (TTS) menggunakan Microsoft Edge TTS API.
    """
    # Tentukan voice model berdasarkan gender
    if gender.lower() == "male":
        voice_name = "id-ID-ArdiNeural"
    else:
        voice_name = "id-ID-GadisNeural"
        
    # Atur tempo kecepatan bicara
    if speed.lower() == "slow":
        rate_str = "-35%"
    elif speed.lower() == "fast":
        rate_str = "+40%"
    else:
        rate_str = "+0%"
        
    # Buat objek communicate dan simpan ke file output
    communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate_str)
    await communicate.save(output_path)

