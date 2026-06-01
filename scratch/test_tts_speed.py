import asyncio
import edge_tts
import os

async def test_speed():
    text = "Halo! Ini adalah uji coba kecepatan suara dari sistem Bloom Socials."
    
    # Uji Coba 3 kecepatan berbeda
    speeds = {
        "slow": "-40%",    # Diperlambat lebih signifikan
        "normal": "+0%",
        "fast": "+45%"     # Dipercepat lebih signifikan
    }
    
    for name, rate in speeds.items():
        output = f"scratch_test_{name}.mp3"
        print(f"Menyintesis suara untuk kecepatan '{name}' dengan rate '{rate}'...")
        communicate = edge_tts.Communicate(text=text, voice="id-ID-GadisNeural", rate=rate)
        await communicate.save(output)
        print(f"Berkas audio '{output}' berhasil disimpan. Ukuran: {os.path.getsize(output)} byte")

if __name__ == "__main__":
    asyncio.run(test_speed())
