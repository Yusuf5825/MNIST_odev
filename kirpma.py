import cv2
import os
import uuid
import glob

def auto_crop_all_digits(input_dir, base_output_dir, padding=30):

    os.makedirs(base_output_dir, exist_ok=True)
    
    image_extensions = ('*.jpeg', '*.jpg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not image_paths:
        print("Belirtilen klasörde işlenecek resim bulunamadı.")
        return

    total_cropped = 0
    
    for img_path in image_paths:

        filename = os.path.basename(img_path)
        
        # Dosya isminden sınıf adını çıkar (alt çizgi öncesini alır -> sifir)
        class_name = filename.split('_')[0]
        
        class_dir = os.path.join(base_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Görüntüyü oku ve siyah-beyaza çevir
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Konturları bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count_for_this_img = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            
            # Gürültü filtreleme
            if 10 < w < 400 and 30 < h < 400:
                size = max(w, h)
                center_x = x + w // 2
                center_y = y + h // 2
                
                half_size = (size // 2) + padding
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(img.shape[1], center_x + half_size)
                y2 = min(img.shape[0], center_y + half_size)
                
                cropped = img[y1:y2, x1:x2]
                
                if cropped.shape[0] == cropped.shape[1] and cropped.shape[0] > 0:
                    # Çakışmayı önlemek için 6 haneli rastgele bir ID oluştur
                    unique_id = uuid.uuid4().hex[:6]
                    
                    
                    save_path = os.path.join(class_dir, f"{class_name}_{unique_id}.jpg")
                    
                    cv2.imwrite(save_path, cropped)
                    count_for_this_img += 1
                    total_cropped += 1
                    
        print(f"'{filename}' işlendi: {count_for_this_img} adet '{class_name}' kırpıldı ve klasörlendi.")



input_folder = r"C:\Projeler\MNIST_odev\ham_uretilen_test_data\gozden_kacan"
output_folder = r"C:\Projeler\MNIST_odev\kendi_test_verilerim"

auto_crop_all_digits(input_folder, output_folder)