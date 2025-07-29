🩻 Medical DICOM Processing Pipeline

Bu proje, dijital mamografi (DICOM) görüntülerini işleyerek:
- Meme bölgesini otomatik olarak tespit eder,
- Görüntüyü kırpar ve normalize eder,
- PNG formatına dönüştürür,
- Görüntüyle ilişkili hasta ve çekim bilgilerini `CSV` dosyasına kaydeder.

🚀 Özellikler

- Gelişmiş segmentasyon (Gaussian Blur + Otsu Threshold + morfolojik işlemler)
- Başarısız segmentasyon durumunda fallback yöntem (Adaptif Thresholding)
- Bozuk veya sıkıştırılmış DICOM dosyaları için `gdcmconv` desteği
- DICOM → PNG dönüşümü
- Otomatik metadata çıkarımı (`Age`, `Study_ID`, `ViewPosition`, vs.)
- Başarılı ve başarısız dosya listeleri

🔧 Gereksinimler

- Python 3.7+
- `pydicom`, `opencv-python`, `scikit-image`, `pandas`, `Pillow`
- Harici: `gdcmconv` (sıkıştırılmış DICOM'ları açmak için)

Tüm Python paketlerini yüklemek için:
```bash
pip install -r requirements.txt
