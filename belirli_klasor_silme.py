import os
import shutil
import pandas as pd

def klasorleri_bul_kacinci_png_1_olanlar(root_dir):
    klasorler = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        png_sayisi = sum(1 for file in filenames if file.lower().endswith('.png'))
        if png_sayisi == 7:
            klasorler.append(dirpath)

    return klasorler

# Kullanıcıdan gelen ayarlar
ana_klasor = "D:\\cropped_images"  # Ana klasör yolu

# 1 adet PNG içeren klasörleri bul
silinecek_klasorler = klasorleri_bul_kacinci_png_1_olanlar(ana_klasor)

# Klasörleri sil
for klasor in silinecek_klasorler:
    try:
        shutil.rmtree(klasor)
        print(f"Silindi: {klasor}")
    except Exception as e:
        print(f"Silinemedi: {klasor} - {e}")

