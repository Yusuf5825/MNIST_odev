import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class KendiElYazimDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Klasör isimlerini doğrudan matematiksel etiketlere eşliyoruz. 
        # (Elle etiketleme yapmanıza gerek kalmıyor!)
        self.label_map = {
            "sifir": 0, "bir": 1, "iki": 2, "uc": 3, "dort": 4,
            "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9
        }
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            
            if os.path.isdir(folder_path) and folder_name in self.label_map:
                for img_path in glob.glob(os.path.join(folder_path, "*.jpg")):
                    self.image_paths.append(img_path)
                    # Klasör ismine göre etiketi atıyoruz
                    self.labels.append(self.label_map[folder_name])
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # MNIST formatı için zorunlu olan gri tonlamalı (1 kanal - "L") okuma
        image = Image.open(img_path).convert("L")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 1. Test verilerimiz için ortak ve sadece ZORUNLU işlemleri barındıran transform
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

# 2. Veri setimizi oluşturuyoruz (Klasör yolunu kendinize göre güncelleyin)
benim_test_verim_dir = r"C:\Projeler\MNIST_odev\kendi_test_verilerim_mnist_format"
custom_test_dataset = KendiElYazimDataset(root_dir=benim_test_verim_dir, transform=test_transform)

# 3. Modelin içine vermek için DataLoader'a sarıyoruz
custom_test_dataloader = DataLoader(custom_test_dataset, batch_size=32, shuffle=False)

print(f"Toplam {len(custom_test_dataset)} adet kendi el yazınız başarıyla yüklendi ve etiketlendi!")
