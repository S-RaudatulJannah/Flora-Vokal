import model_utils
import random
import os

print("=" * 50)
print("Testing Model ASR: Klasifikasi Bunga")
print("=" * 50)

# 1. Melatih Model
print("\n[1] Melatih Model ASR...")
hasil_latih = model_utils.train_model("mlp")
print(f"Status: {hasil_latih['message']}")
print(f"Akurasi Train: {round(hasil_latih['train_accuracy'] * 100, 2)}%")
print(f"Akurasi Test : {round(hasil_latih['test_accuracy'] * 100, 2)}%")

# 2. Menguji Prediksi Secara Acak
print("\n[2] Uji Prediksi Sampel Acak...")
dataset_dir = "dataset"
kategori_bunga = model_utils.CLASSES

# Ambil 3 sampel kategori secara acak
kategori_uji = random.sample(kategori_bunga, 3)

for idx, bunga in enumerate(kategori_uji, 1):
    folder_bunga = os.path.join(dataset_dir, bunga)
    if os.path.exists(folder_bunga):
        files = [f for f in os.listdir(folder_bunga) if f.lower().endswith('.wav')]
        if files:
            file_terpilih = random.choice(files)
            path_uji = os.path.join(folder_bunga, file_terpilih)
            
            res = model_utils.predict_audio(path_uji)
            
            print(f"\n--- Pengujian #{idx} ---")
            print(f"File Target: {bunga}/{file_terpilih}")
            print(f"Prediksi   : {res['predicted_class'].upper()}")
            print(f"Confidence : {round(res['confidence'] * 100, 2)}%")
            
            if res['predicted_class'] == bunga:
                print("Hasil      : OK")
            else:
                print("Hasil      : FAILED")

print("\n" + "=" * 50)
print("Testing Selesai.")
print("=" * 50)

