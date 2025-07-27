import os
import pandas as pd

# Ayarlar
csv_dosyasi = "dicom_metadata.csv"
study_id_sutun_adi = "Study_ID"  # CSV'de klasörle eşleşen sütun
ana_klasor = "D:\\cropped_images"

# CSV'yi oku
df = pd.read_csv(csv_dosyasi, sep= ";")

# CSV'deki klasör yollarını oluştur (StudyInstanceUID -> klasör ismi)
csv_klasor_yollari = df[study_id_sutun_adi].astype(str).str.strip().apply(
    lambda x: os.path.normpath(os.path.join(ana_klasor, x))
)

# Gerçek dosya sistemindeki klasörleri bul
gercek_klasor_yollari = []
for dirpath, dirnames, filenames in os.walk(ana_klasor):
    gercek_klasor_yollari.append(os.path.normpath(dirpath))

# Eksik klasörleri bul: CSV'de var, ama dosya sisteminde yok
eksik_klasorler = [klasor for klasor in csv_klasor_yollari if klasor not in gercek_klasor_yollari]

# Sonuçları yazdır
print("\nCSV'de olup dosya sisteminde olmayan klasörler:")
for klasor in eksik_klasorler:
    print(klasor)

print(f"\nToplam {len(eksik_klasorler)} eksik klasör bulundu.")

# Eksik klasörlerin sadece ID'lerini al
eksik_study_idler = [os.path.basename(klasor) for klasor in eksik_klasorler]

# CSV'de bu ID'leri içermeyen satırları filtrele (yani sadece gerçekten var olanları tut)
df_temiz = df[~df[study_id_sutun_adi].astype(str).isin(eksik_study_idler)]

# Temizlenmiş CSV'yi kaydet
df_temiz.to_csv("dicom_metadata_temizlenmis.csv", sep=";", index=False)

print(f"\nYeni temizlenmiş CSV kaydedildi: dicom_metadata_temizlenmis.csv")
print(f"Yeni satır sayısı: {len(df_temiz)}")

