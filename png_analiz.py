import os

def klasorleri_bul_kacinci_png_1_olanlar(root_dir):
    klasorler = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        png_sayisi = sum(1 for file in filenames if file.lower().endswith('.png'))
        if png_sayisi == 2:
            klasorler.append(dirpath)

    return klasorler

# Ana dizini buraya yaz
ana_klasor = "D:\\cropped_images"  # Örneğin

sonuc = klasorleri_bul_kacinci_png_1_olanlar(ana_klasor)

# Sonuçları yazdır
print("2 adet .png içeren klasörler:")
for klasor in sonuc:
    print(klasor)
    
print(f"\n2 tane goruntu içeren toplam {len(sonuc)} klasör bulundu.")

