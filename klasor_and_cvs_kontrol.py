import os
import pandas as pd

# Ayarlar
csv_dosyasi = "dicom_metadata.csv"
study_id_sutun_adi = "Study_ID"
ana_klasor = "D:\\cropped_images"

# CSV'yi oku
df = pd.read_csv(csv_dosyasi, sep=";")

# CSV'deki Study_ID'leri al
csv_study_idler = df[study_id_sutun_adi].astype(str).str.strip().tolist()

# Klasördeki alt klasör isimlerini al (birinci seviye klasörler)
klasordeki_study_idler = [
    isim for isim in os.listdir(ana_klasor)
    if os.path.isdir(os.path.join(ana_klasor, isim))
]

# Eksik klasörler: CSV'de var, klasörde yok
eksik_klasorler = [id for id in csv_study_idler if id not in klasordeki_study_idler]

# Fazla klasörler: Klasörde var, CSV'de yok
fazla_klasorler = [id for id in klasordeki_study_idler if id not in csv_study_idler]

# Raporla
print("EKSİK KLASÖRLER (CSV'de olup klasörde olmayanlar):")
for klasor in eksik_klasorler:
    print(f"- {klasor}")
print(f"Toplam {len(eksik_klasorler)} eksik klasör.\n")

print("FAZLA KLASÖRLER (Klasörde olup CSV'de olmayanlar):")
for klasor in fazla_klasorler:
    print(f"- {klasor}")
print(f"Toplam {len(fazla_klasorler)} fazla klasör.")
