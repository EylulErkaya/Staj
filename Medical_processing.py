"""
Medical DICOM Processing Pipeline
----------------------------------

Bu modül, dijital mamografi (DICOM formatindaki) medikal görüntüleri işlemek için tasarlanmiş bütünleşik bir sistem sunar. Amaç, ham DICOM dosyalarini otomatik olarak analiz etmek, meme bölgesini tespit ederek kirpmak, PNG formatinda kaydetmek ve hasta bilgilerini organize bir CSV dosyasina aktarmaktir.

İşlem Aşamalari:
1. DICOM dosyalarini belirtilen klasörde bulur.
2. Her dosyadan hasta yaşi, görüntü pozisyonu gibi temel metadata bilgilerini çikarir.
3. Görüntü verisi okunamadiğinda `gdcmconv` ile dekompresyon denemesi yapilir.
4. Görüntü işleme aşamasinda meme bölgesi tespit edilir:
   - Önce gelişmiş segmentasyon (Gaussian blur + Otsu threshold + morfolojik işlemler) denenir.
   - Başarisiz olursa adaptif thresholding ile basit segmentasyon uygulanir.
5. Tespit edilen bölgeye göre görüntü kirpilir, normalize edilip uint8'e çevrilir ve PNG olarak kaydedilir.
6. Başariyla işlenen dosyalarin bilgileri `dicom_metadata.csv` dosyasina kaydedilir.
7. İşlenemeyen dosyalar `failed_files.csv` olarak loglanir.

Yapi:
- `DICOMProcessor`: Ana işleyici sinif. Tüm işlem adimlarini içerir.
- `main()`: Parametreleri belirleyip işleyiciyi çaliştiran fonksiyon.

Gereksinimler:
- `pydicom`, `opencv-python`, `scikit-image`, `Pillow`, `pandas`, `gdcmconv` (harici araç)
- Python 3.7+ önerilir
"""

import os
import pandas as pd
import pydicom
from PIL import Image
import numpy as np
import cv2
import subprocess
from pathlib import Path
from skimage import morphology, measure
import logging

# Logging ayarlari
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DICOMProcessor:
    """Main class for processing DICOM files and extracting breast regions"""
    
    def __init__(self, root_folder, output_folder, csv_filename):
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.csv_filename = csv_filename
        self.processed_files = []
        self.failed_files = []
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
    
    def run(self):
        """Main processing pipeline"""
        print("Starting DICOM processing...")
        
        # Find all DICOM files
        dicom_files = self._find_dicom_files()
        print(f"Found {len(dicom_files)} DICOM files")
        
        if not dicom_files:
            print("No DICOM files found!")
            return
        
        # Process each file
        for i, file_path in enumerate(dicom_files, 1):
            print(f"\nProcessing {i}/{len(dicom_files)}: {os.path.basename(file_path)}")
            self._process_single_file(file_path)
        
        # Save results to CSV
        self._save_metadata()
        self._save_failed_files()
        print(f"\nProcessing complete! Results saved to {self.csv_filename}")
        print(f"Successfully processed: {len(self.processed_files)} files")
        print(f"Failed: {len(self.failed_files)} files")
    
    def _find_dicom_files(self):
        """Find all DICOM files in the root folder"""
        print("Finding all DICOM files in the root folder...")
        dicom_files = []
        
        for root, dirs, files in os.walk(self.root_folder):
            for file in files:
                if file.lower().endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    dicom_files.append(file_path)
        
        return dicom_files
    
    def _is_dicom_file(self, file_path):
        """Check if file is a valid DICOM file"""
        try:
            with pydicom.dcmread(file_path, stop_before_pixels=True, force=True) as ds:
                return True
        except Exception:
            return False
    
    def _process_single_file(self, file_path):
        """Process a single DICOM file with better error handling"""
        try:
            # Extract metadata with force=True to handle corrupted files
            metadata, dicom_data = self._extract_metadata(file_path)
            if not metadata or not dicom_data:
                self.failed_files.append({
                    'file_path': file_path,
                    'error': 'Failed to extract metadata'
                })
                return
            
            # Process and save image
            image_info = self._process_image(dicom_data, file_path, metadata)
            if image_info:
                metadata.update(image_info)
                self.processed_files.append(metadata)
            else:
                self.failed_files.append({
                    'file_path': file_path,
                    'error': 'Failed to process image'
                })
                
        except Exception as e:
            error_msg = f"Failed to process {file_path}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.failed_files.append({
                'file_path': file_path,
                'error': str(e)
            })
    
    @staticmethod
    def _clean_patient_age(age_str):
        """Clean DICOM PatientAge string and return numeric age"""
        if age_str == 'N/A' or not age_str:
            return age_str
        
        age_str = str(age_str).strip()
        
        # Normal format: '032Y', '105Y', etc.
        if len(age_str) < 2:
            return age_str
            
        unit = age_str[-1]
        number_part = age_str[:-1]
        
        # Convert to number
        try:
            number = int(number_part)
        except ValueError:
            return age_str
        
        if unit != 'Y':
            return age_str
        
        return str(number)

    def _extract_metadata(self, file_path):
        """Extract metadata from DICOM file with GDCM fallback if needed"""
        try:
            ds = pydicom.dcmread(file_path, force=True)

            # DICOM metadata
            raw_view = getattr(ds, 'ViewPosition', 'N/A')
            normalized_view = self._normalize_view_position(raw_view)

            raw_age = getattr(ds, 'PatientAge', 'N/A')
            clean_age = self._clean_patient_age(raw_age)

            metadata = {
                'Directory_Path': file_path,
                'View_Position': normalized_view,
                'Image_Laterality': getattr(ds, 'ImageLaterality', 'N/A'),
                'Patient_Age': clean_age,
                'Study_ID': getattr(ds, 'SeriesInstanceUID', 'N/A'),
                'Image_ID': getattr(ds, 'SOPInstanceUID', 'N/A')
            }

            return metadata, ds

        except Exception as e:
            print(f"[ERROR] Metadata extraction failed for {file_path}: {str(e)}")
            return None, None

    def _decompress_with_gdcmconv(self, input_path, output_path):
        try:
            result = subprocess.run(["gdcmconv", "--raw", input_path, output_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] gdcmconv failed: {e}")
            return False

    
    def _normalize_view_position(self, view_position):
        """Normalize view position values"""
        if view_position == 'N/A':
            return 'N/A'
        
        view_position = str(view_position).upper().strip()
        
        if 'XCC' in view_position:
            return 'CC'
        elif 'SIO' in view_position:
            return 'MLO'
        elif 'CC' in view_position:
            return 'CC'
        elif 'MLO' in view_position:
            return 'MLO'
        else:
            return view_position
    
    def _process_image(self, dicom_data, file_path, metadata):
        """Process and save DICOM image with GDCM fallback if pixel_array fails"""
        try:
            # Check if pixel data exists
            if not hasattr(dicom_data, 'pixel_array'):
                raise Exception("No pixel array found in DICOM")

            try:
                pixel_array = dicom_data.pixel_array

            except Exception as e:
                print(f"[WARNING] Cannot decompress pixel data: {str(e)}")
                decompressed_path = file_path.replace(".dcm", "_decompressed.dcm")
                
                if self._decompress_with_gdcmconv(file_path, decompressed_path):
                    try:
                        dicom_data = pydicom.dcmread(decompressed_path, force=True)
                        pixel_array = dicom_data.pixel_array
                    except Exception as e2:
                        print(f"[ERROR] Still cannot read pixel array after decompress: {str(e2)}")
                        return None
                else:
                    return None

            # Detect breast region
            breast_region = self._detect_breast_region(pixel_array)

            # Crop image
            if breast_region:
                cropped_image = self._crop_image(pixel_array, breast_region)
            else:
                print(f"[WARNING] Using full image for: {file_path}")
                cropped_image = pixel_array

            # Convert to proper format
            processed_image = self._convert_to_uint8(cropped_image, dicom_data)

            # Save image
            self._save_image(processed_image, file_path, metadata)

            width, height = processed_image.shape[1], processed_image.shape[0]
            return {
                'Width': width,
                'Height': height,
            }

        except Exception as e:
            print(f"[ERROR] Image processing failed for {file_path}: {str(e)}")
            return None

    
    def _detect_breast_region(self, pixel_array):
        """Detect breast region using advanced method, fallback to simple method"""
        # Try advanced detection first
        region = self._detect_breast_advanced(pixel_array)
        if region:
            return region
        
        # Fallback to simple detection
        print("[INFO] Using fallback detection method")
        return self._detect_breast_simple(pixel_array)
    
    def _detect_breast_advanced(self, pixel_array):
        """Advanced breast region detection using edge and intensity analysis"""
        try:
            # Normalize image
            normalized = self._normalize_image(pixel_array)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
            
            # Apply Otsu thresholding
            _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            combined_mask = otsu_thresh > 0
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small objects and holes
            combined_mask = morphology.remove_small_objects(combined_mask.astype(bool), min_size=1000)
            combined_mask = morphology.remove_small_holes(combined_mask, area_threshold=500)
            
            # Find largest region
            labeled_mask = measure.label(combined_mask)
            if labeled_mask.max() == 0:
                return None
            
            regions = measure.regionprops(labeled_mask)
            if not regions:
                return None
            
            largest_region = max(regions, key=lambda x: x.area)
            min_row, min_col, max_row, max_col = largest_region.bbox
            
            # Add padding
            padding = 20
            min_row = max(0, min_row - padding)
            min_col = max(0, min_col - padding)
            max_row = min(pixel_array.shape[0], max_row + padding)
            max_col = min(pixel_array.shape[1], max_col + padding)
            
            return (min_row, min_col, max_row, max_col)
            
        except Exception as e:
            print(f"[ERROR] Advanced detection failed: {str(e)}")
            return None
    
    def _detect_breast_simple(self, pixel_array):
        """Simple breast region detection using adaptive thresholding"""
        try:
            # Normalize image
            normalized = self._normalize_image(pixel_array)
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(pixel_array.shape[1] - x, w + 2 * padding)
            h = min(pixel_array.shape[0] - y, h + 2 * padding)
            
            return (y, x, y + h, x + w)
            
        except Exception as e:
            print(f"[ERROR] Simple detection failed: {str(e)}")
            return None
    
    def _normalize_image(self, pixel_array):
        """Normalize pixel array to 0-255 range"""
        try:
            if pixel_array.max() > 255:
                normalized = ((pixel_array - pixel_array.min()) / 
                             (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                normalized = pixel_array.astype(np.uint8)
            return normalized
        except Exception:
            return pixel_array.astype(np.uint8)
    
    def _crop_image(self, pixel_array, region):
        """Crop image using detected region"""
        min_row, min_col, max_row, max_col = region
        return pixel_array[min_row:max_row, min_col:max_col]
    
    def _convert_to_uint8(self, image, dicom_data):
        """Convert image to uint8 format"""
        if image.dtype == np.uint8:
            return image
        
        if image.max() <= 255:
            return image.astype(np.uint8)
        
        # Use window center/width if available
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            center = dicom_data.WindowCenter
            width = dicom_data.WindowWidth
            
            # Handle multiple values
            if isinstance(center, (list, tuple, pydicom.multival.MultiValue)):
                center = center[0]
            if isinstance(width, (list, tuple, pydicom.multival.MultiValue)):
                width = width[0]
            
            img_min = center - width // 2
            img_max = center + width // 2
            image = np.clip(image, img_min, img_max)
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            # Use percentile-based normalization
            p_low = np.percentile(image, 1)
            p_high = np.percentile(image, 99)
            image = np.clip(image, p_low, p_high)
            image = ((image - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        
        return image
    
    def _save_image(self, image, original_path, metadata):
        """Save processed image as PNG"""
        try:
            # Create study folder
            study_folder = self._create_study_folder(metadata['Study_ID'])
            
            # Generate filename
            file_name = Path(original_path).stem
            png_filename = f"{file_name}.png"
            output_path = os.path.join(study_folder, png_filename)
            
            # Save image
            Image.fromarray(image).save(output_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to save image: {str(e)}")
            return None
    
    def _create_study_folder(self, study_id):
        """Create folder for study"""
        try:
            safe_study_id = str(study_id).replace('/', '').replace('\\', '').replace(':', '_') if study_id != 'N/A' else 'Unknown_StudyID'
            study_folder = os.path.join(self.output_folder, safe_study_id)
            os.makedirs(study_folder, exist_ok=True)
            return study_folder
        except Exception as e:
            print(f"[ERROR] Could not create folder: {str(e)}")
            fallback = os.path.join(self.output_folder, "Unknown_Study")
            os.makedirs(fallback, exist_ok=True)
            return fallback
    
    def _save_metadata(self):
        """Save metadata to CSV file"""
        if not self.processed_files:
            print("No files were processed successfully.")
            return
        
        df = pd.DataFrame(self.processed_files)
        
        # Format IDs
        df['Study_ID'] = df['Study_ID'].astype(str)
        df['Image_ID'] = df['Image_ID'].astype(str) + '.png'

        # Remove Output_Path column if exists
        if 'Output_Path' in df.columns:
            df = df.drop(columns=['Output_Path'])
        
        # Save to CSV
        df.to_csv(self.csv_filename, index=False, encoding='utf-8-sig')
        print(f"Metadata saved to {self.csv_filename}")
        print(f"Total processed files: {len(self.processed_files)}")
    
    def _save_failed_files(self):
        """Save failed files information"""
        if self.failed_files:
            failed_df = pd.DataFrame(self.failed_files)
            failed_csv = "failed_files.csv"
            failed_df.to_csv(failed_csv, index=False, encoding='utf-8-sig')
            print(f"Failed files list saved to {failed_csv}")


def main():
    """Main function to run the DICOM processor"""
    # Configuration
    ROOT_FOLDER = "D:\\2020"
    OUTPUT_FOLDER = "cropped_images"
    CSV_FILENAME = "dicom_metadata.csv"
    
    # Create processor and run
    processor = DICOMProcessor(ROOT_FOLDER, OUTPUT_FOLDER, CSV_FILENAME)
    processor.run()


if __name__ == "__main__":
    main()