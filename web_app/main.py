import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# ==========================================
# 1. MODEL MİMARİSİ
# ==========================================
class MNISTClassifier(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model_path = os.path.join(PROJECT_ROOT, "results", "models", "mnist_Scheduler_Nöron_sayısı_artırıldı_v7.pth")
if not os.path.exists(model_path):
    model_path = os.path.join(PROJECT_ROOT, "results", "models", "mnist_Scheduler_Noron_sayisi_artirildi_v7.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MNISTClassifier(input_shape=784, hidden_units=64, output_shape=10).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

# ==========================================
# 2. WEB API VE YATAY EKSEN (SATIR) ALGORİTMASI
# ==========================================
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/predict")
async def predict_handwriting(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Görüntüyü Threshold ile Siyah Zemin / Beyaz Rakam yap (Klavye tuşlarını izole eder)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = thresh.shape
    margin_x, margin_y = int(img_w * 0.03), int(img_h * 0.03)
    
    potential_boxes = []
    
    # 2. Hatalı Nesneleri Eleme
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Kenardaki spiralleri yoksay
        if x < margin_x or x+w > img_w - margin_x or y < margin_y or y+h > img_h - margin_y:
            continue
            
        # Çok küçük (toz) veya devasa (hatalı) nesneleri yoksay
        if h < img_h * 0.03 or h > img_h * 0.7 or w < 3:
            continue
        if w > h * 2.5: # Aşırı yatay çizgileri yoksay
            continue
            
        # Klavye Tuşlarını Eleme: 1 rakamı için toleransı %85'e çıkardık
        roi = thresh[y:y+h, x:x+w]
        fill_ratio = cv2.countNonZero(roi) / (w * h)
        if fill_ratio > 0.85 or fill_ratio < 0.05: 
            continue
            
        potential_boxes.append((x, y, w, h))
        
    if not potential_boxes:
        return {"okunan_numara": "Rakam Bulunamadı (Filtre Hatası)"}
        
    # 3. YATAY SATIR GRUPLAMASI (Aralarındaki boşluk ne olursa olsun aynı hizadakileri bulur)
    lines = []
    for box in potential_boxes:
        b_cy = box[1] + box[3] / 2
        found_line = False
        
        for line in lines:
            l_cy = line['cy']
            # Eğer Y merkezleri yakınsa, aynı satırdadır
            if abs(b_cy - l_cy) < line['med_h'] * 0.6:
                line['boxes'].append(box)
                line['cy'] = np.mean([b[1] + b[3]/2 for b in line['boxes']])
                line['med_h'] = np.median([b[3] for b in line['boxes']])
                found_line = True
                break
                
        if not found_line:
            lines.append({'boxes': [box], 'cy': b_cy, 'med_h': box[3]})
            
    # En çok rakama sahip olan satırı seç
    best_line = max(lines, key=lambda l: len(l['boxes']))
    valid_boxes = best_line['boxes']
    
    # Kendi satır ortalamasına göre çok küçük/büyük kalan lekeleri temizle
    med_h = best_line['med_h']
    final_boxes = [b for b in valid_boxes if med_h * 0.5 < b[3] < med_h * 1.5]
    
    # Rakamları soldan sağa doğru diz
    final_boxes.sort(key=lambda b: b[0])
    
    if not final_boxes:
        return {"okunan_numara": "Rakam Bulunamadı (Hizalama Hatası)"}
        
    predictions = []
    
    # 4. KÜTLE MERKEZİ İLE TAHMİN (Rakam formunu bozmaz)
    for x, y, w, h in final_boxes:
        digit_crop = thresh[y:y+h, x:x+w]
        
        size = max(w, h)
        padding = int(size * 0.25)
        square_size = size + (padding * 2)
        
        canvas = np.zeros((square_size, square_size), dtype=np.uint8)
        start_x = padding + (size - w) // 2
        start_y = padding + (size - h) // 2
        canvas[start_y:start_y+h, start_x:start_x+w] = digit_crop
        
        resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        
        M = cv2.moments(resized)
        if M["m00"] != 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = 14, 14
            
        shift_x, shift_y = 14.0 - cx, 14.0 - cy
        M_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        final_img = cv2.warpAffine(resized, M_matrix, (28, 28))
        
        tensor_img = test_transform(final_img).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            pred = model(tensor_img)
            digit = torch.argmax(F.softmax(pred, dim=1), dim=1).item()
            predictions.append(str(digit))
            
    return {"okunan_numara": "".join(predictions)}

if __name__ == "__main__":
    print("🚀 Web sunucusu başlatılıyor... Lütfen tarayıcıdan http://127.0.0.1:8000 adresine gidin.")
    uvicorn.run(app, host="127.0.0.1", port=8000)