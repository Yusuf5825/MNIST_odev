import matplotlib.pyplot as plt

def ciz_mlp_mimarisi():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Katmanlardaki çizilecek "sembolik" nöron sayıları (Ekrana sığması için)
    layer_sizes = [14, 8, 8, 10]
    # Gerçek nöron sayıları
    actual_sizes = [784, 16, 16, 10]
    
    # Katman isimleri
    labels = ['Girdi Katmanı\n(Flatten: 784)', 
              'Gizli Katman 1\n(16 Nöron + ReLU)', 
              'Gizli Katman 2\n(16 Nöron + ReLU)', 
              'Çıktı Katmanı\n(10 Sınıf: 0-9)']

    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    node_coords = []

    # Nöronları (Daireleri) Çizme
    for n, (layer_size, actual_size) in enumerate(zip(layer_sizes, actual_sizes)):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        
        # Üst Başlık Kutusunu Ekleme
        ax.text(n * h_spacing + left, top + 0.05, labels[n], ha='center', va='bottom', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
        
        layer_nodes = []
        for m in range(layer_size):
            x = n * h_spacing + left
            y = layer_top - m * v_spacing
            
            # Orta kısımlara denk gelen nöronları gizleyip "..." yazalım (Çok nöron olduğunu belirtmek için)
            if actual_size > 10 and m in [layer_size//2 - 1, layer_size//2, layer_size//2 + 1]:
                if m == layer_size//2:
                    ax.text(x, y, "...", ha='center', va='center', fontsize=24, fontweight='bold')
                continue # Daire çizme
            
            # Renk Seçimi
            if n == 0: color = '#87CEFA'          # Mavi (Girdi)
            elif n == len(layer_sizes)-1: color = '#98FB98' # Yeşil (Çıktı)
            else: color = '#FFB6C1'               # Pembe (Gizli)
                
            circle = plt.Circle((x, y), v_spacing / 3., color=color, ec='black', zorder=4)
            ax.add_artist(circle)
            layer_nodes.append((x, y))
            
        node_coords.append(layer_nodes)

    # Bağlantıları (Sinaps Ağırlıklarını) Çizme
    for n in range(len(node_coords) - 1):
        for coord_a in node_coords[n]:
            for coord_b in node_coords[n + 1]:
                line = plt.Line2D([coord_a[0], coord_b[0]], [coord_a[1], coord_b[1]], c='gray', alpha=0.15, zorder=1)
                ax.add_artist(line)

    plt.title("MNIST MLP Mimarisi (Bottleneck - 16 Nöronlu Model)", fontsize=16, fontweight='bold', y=1.08)
    
    # Raporun plots klasörüne otomatik kaydet
    import os
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/mlp_mimarisi.png", dpi=300, bbox_inches='tight')
    print("Mimari çizimi başarıyla 'results/plots/mlp_mimarisi.png' olarak kaydedildi!")
    plt.show()

# Kodu çalıştır
ciz_mlp_mimarisi()