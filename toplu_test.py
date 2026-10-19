import os
import glob
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from test_data_loader import KendiElYazimDataset
import torch.nn as nn

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
    def forward(self,x):
        return self.layer_stack(x)

def calculate_accuracy(y_pred, y_true):
    correct = torch.eq(torch.argmax(y_pred, dim=1), y_true).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def test_all_models():
    # 1. Klasör yollarını belirle
    models_dir = r"C:\Projeler\MNIST_odev\results\models"
    test_data_dir = r"C:\Projeler\MNIST_odev\kendi_test_verilerim_mnist_format"
    results_dir = r"C:\Projeler\MNIST_odev\results/test_sonuclari"
    os.makedirs(results_dir, exist_ok=True)

    # 2. Test verilerini hazırlama (Zorunlu Transform)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    print("Kendi el yazım test verileri yükleniyor...")
    custom_test_dataset = KendiElYazimDataset(root_dir=test_data_dir, transform=test_transform)
    test_dataloader = DataLoader(custom_test_dataset, batch_size=32, shuffle=False)
    
    all_results = {}
    
    loss_fn = torch.nn.CrossEntropyLoss()
    model_paths = glob.glob(os.path.join(models_dir, "*.pth"))
    
    if not model_paths:
        print("models klasöründe hiçbir .pth dosyası bulunamadı!")
        return

    print(f"Toplam {len(model_paths)} adet model bulundu. Testler başlıyor...\n")

    # 4. Modelleri sırayla test etme döngüsü
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"[{model_name}] test ediliyor...")
        
        if model_name == "mnist_Scheduler_Nöron_sayısı_artırıldı_v7.pth":
           hidden_units = 64
        else:
            hidden_units = 16
        model = MNISTClassifier(input_shape=784, hidden_units=hidden_units, output_shape=10) 
        model.load_state_dict(torch.load(model_path))
        
        
        model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for X, y_test in test_dataloader:
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y_test).item()
                test_acc += calculate_accuracy(test_pred, y_test)
                
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
            
        print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%\n")
        
        all_results[model_name] = {
            "Test_Loss": round(test_loss, 5),
            "Test_Accuracy": round(test_acc, 2)
        }

    json_path = os.path.join(results_dir, "tum_modellerin_sonuclari.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
        
    txt_path = os.path.join(results_dir, "rapor.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== KENDİ EL YAZISI TEST VERİSİ SONUÇLARI ===\n\n")
        for m_name, res in all_results.items():
            f.write(f"Model Adı: {m_name}\n")
            f.write(f"Test Loss: {res['Test_Loss']}\n")
            f.write(f"Accuracy: {res['Test_Accuracy']:.2f}%\n")
            f.write("-" * 40 + "\n")

    print(f">>> Tüm testler bitti! Sonuçlar '{results_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    test_all_models()