import os
import cv2
import glob

def convert_to_mnist_format(input_base_dir, output_base_dir):
    """
    Kırpılmış rakam fotoğraflarını MNIST veri seti standartlarına
    (siyah arka plan, beyaz rakam, 28x28 boyut) dönüştürür.
    """
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    
    subfolders = [f.path for f in os.scandir(input_base_dir) if f.is_dir()]
    
    if not subfolders:
        print("Alt klasörler bulunamadı. Lütfen girdi adresini kontrol edin.")
        return
        
    total_processed = 0
    
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        
        out_folder = os.path.join(output_base_dir, folder_name)
        os.makedirs(out_folder, exist_ok=True)
        
        image_paths = glob.glob(os.path.join(folder, "*.jpg"))
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            out_path = os.path.join(out_folder, img_name)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Bu işlem arka planı TAM SİYAH (0), kalem izini TAM BEYAZ (255) yapar.
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # orijinal MNIST verisindeki gibi kenarlara doğal gri tonlama (anti-aliasing) katar.
            resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(out_path, resized)
            total_processed += 1
            
        print(f"'{folder_name}' klasöründeki resimler MNIST formatına dönüştürüldü.")


input_dir = r"C:\Projeler\MNIST_odev\kendi_test_verilerim"

output_dir = r"C:\Projeler\MNIST_odev\kendi_test_verilerim_mnist_format"

# Fonksiyonu çalıştır
convert_to_mnist_format(input_dir, output_dir)