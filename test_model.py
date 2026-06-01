import model_utils
import random
import os

print("=" * 60)
print("     UJI COBA MANDIRI: TAMAN SUARA BUNGA NUSANTARA")
print("=" * 60)

# 1. Melatih Model
print("\n[Langkah 1] Menjalankan Pelatihan Model ASR...")
hasil_latih = model_utils.train_model("mlp")
print(f"-> Status: {hasil_latih['message']}")
print(f"-> Akurasi Latih   : {round(hasil_latih['train_accuracy'] * 100, 2)}%")
print(f"-> Akurasi Validasi: {round(hasil_latih['test_accuracy'] * 100, 2)}%")

# 2. Menguji Prediksi Secara Acak
print("\n[Langkah 2] Menguji Tebakan Model Pada 3 Sampel Acak...")
dataset_dir = "dataset"
kategori_bunga = model_utils.CLASSES

# Pilih 3 kategori bunga secara acak untuk diuji
kategori_uji = random.sample(kategori_bunga, 3)

for idx, bunga in enumerate(kategori_uji, 1):
    folder_bunga = os.path.join(dataset_dir, bunga)
    if os.path.exists(folder_bunga):
        files = [f for f in os.listdir(folder_bunga) if f.lower().endswith('.wav')]
        if files:
            file_terpilih = random.choice(files)
            path_uji = os.path.join(folder_bunga, file_terpilih)
            
            # Jalankan prediksi
            res = model_utils.predict_audio(path_uji)
            
            print(f"\n--- Tes #{idx} ---")
            print(f"File Audio Asli  : {bunga}/{file_terpilih}")
            print(f"Tebakan Model    : {res['predicted_class'].upper()}")
            print(f"Tingkat Keyakinan: {round(res['confidence'] * 100, 2)}%")
            
            if res['predicted_class'] == bunga:
                print("Hasil            : ✅ BENAR!")
            else:
                print("Hasil            : ❌ SALAH!")

print("\n" + "=" * 60)
print("             UJI COBA SELESAI DENGAN SUKSES!")
print("=" * 60)
