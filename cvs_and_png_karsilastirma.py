import os
import pandas as pd

# Ayarlar
csv_dosyasi = "dicom_metadata.csv"
png_sutun_adi = "Image_ID"  # CSV'deki PNG dosya adlarini içeren sütun adi
ana_klasor = "D:\\cropped_images"

# CSV'yi oku
df = pd.read_csv(csv_dosyasi, sep=";")

# CSV'deki PNG dosya adlarini al
csv_png_dosyalar = df[png_sutun_adi].astype(str).str.strip().tolist()

# Klasördeki tüm PNG dosyalarini (alt klasörler dahil) al
klasordeki_png_dosyalar = []
for root, dirs, files in os.walk(ana_klasor):
    for file in files:
        if file.lower().endswith(".png"):
            klasordeki_png_dosyalar.append(file.strip())

# Eksik dosyalar: CSV'de var, klasörde yok
eksik_dosyalar = [f for f in csv_png_dosyalar if f not in klasordeki_png_dosyalar]

# Fazla dosyalar: Klasörde var, CSV'de yok
fazla_dosyalar = [f for f in klasordeki_png_dosyalar if f not in csv_png_dosyalar]

# Raporla
print("FAZLA PNG DOSYALARI (Klasörde olup CSV'de olmayanlar):")
for dosya in fazla_dosyalar:
    print(f"- {dosya}")
print(f"Toplam {len(fazla_dosyalar)} fazla PNG dosyasi.")

print("EKSİK PNG DOSYALARI (CSV'de olup klasörde olmayanlar):")
for dosya in eksik_dosyalar:
    print(f"- {dosya}")
print(f"Toplam {len(eksik_dosyalar)} eksik PNG dosyasi.\n")
# Eksik PNG'leri içermeyen yeni DataFrame oluştur
df_filtered = df[~df[png_sutun_adi].astype(str).str.strip().isin(eksik_dosyalar)]

# Yeni CSV'yi kaydet
df_filtered.to_csv("dicom_metadata_filtered.csv", sep=";", index=False)
print("Eksik PNG'ler silinerek 'dicom_metadata_filtered.csv' dosyasina kaydedildi.")
