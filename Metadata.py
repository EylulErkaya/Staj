import os
import pydicom
import pandas as pd

def find_dicom_files(root_dir):
    """
    root_dir içindeki tüm alt klasörleri gezerek .dcm dosyalarını bulur.
    """
    dicom_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

def extract_all_metadata(dicom_path):
    """
    Bir DICOM dosyasındaki tüm metadata alanlarını sözlük olarak çıkarır.
    """
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        metadata = {"file_path": dicom_path}
        for elem in dcm.iterall():
            if elem.VR != 'SQ':  # Nested sequence'ler hariç tutuldu
                key = f"{elem.tag}_{elem.keyword}"
                metadata[key] = str(elem.value)
        return metadata
    except Exception as e:
        print(f"Hata: {dicom_path} - {e}")
        return None

def main(root_folder, output_csv="dicom_metadata.csv"):
    all_metadata = []
    dicom_files = find_dicom_files(root_folder)

    print(f"{len(dicom_files)} DICOM dosyası bulundu. Metadata çıkartılıyor...")

    for file in dicom_files:
        metadata = extract_all_metadata(file)
        if metadata:
            all_metadata.append(metadata)

    # Tüm unique alanları kapsayan dataframe
    df = pd.DataFrame(all_metadata)
    df.to_csv(output_csv, index=False)
    print(f"Metadata CSV'ye yazıldı: {output_csv}")

if __name__ == "__main__":
    root_dicom_folder = "PacsMG" 
    main(root_dicom_folder)
