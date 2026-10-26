document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById('drop-zone');
    const imageUpload = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const dropContent = document.getElementById('drop-content');
    const predictBtn = document.getElementById('predictBtn');
    const spinner = document.getElementById('spinner');
    const btnText = document.querySelector('.btn-text');
    const resultBox = document.getElementById('resultBox');
    const resultNumber = document.getElementById('resultNumber');
    
    let selectedFile = null;

    // Fotoğrafı Ekranda Gösterme Fonksiyonu
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = function(event) {
                preview.src = event.target.result;
                preview.style.display = 'block';
                dropContent.style.display = 'none';
                predictBtn.style.display = 'flex';
                resultBox.style.display = 'none';
            }
            reader.readAsDataURL(file);
        } else {
            alert("Lütfen geçerli bir resim dosyası yükleyin.");
        }
    }

    // Tıklayarak Dosya Seçme
    imageUpload.addEventListener('change', function(e) {
        handleFile(e.target.files[0]);
    });

    // Sürükle ve Bırak Olayları
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Tıklayarak Kutuya Yükleme Desteği
    dropZone.addEventListener('click', (e) => {
        if(e.target !== imageUpload && e.target.tagName !== 'LABEL') {
            imageUpload.click();
        }
    });

    // Backend (FastAPI) İletişimi
    predictBtn.addEventListener('click', async function() {
        if (!selectedFile) return;

        // Loading Animasyonu Başlat
        btnText.innerText = "Yapay Zeka Analiz Ediyor...";
        spinner.style.display = 'block';
        predictBtn.style.pointerEvents = 'none'; // Çift tıklamayı önle
        resultBox.style.display = 'none';

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) throw new Error("Sunucu hatası!");

            const data = await response.json();
            
            // Sonucu Göster
            resultNumber.innerText = data.okunan_numara || "Rakam Bulunamadı";
            resultBox.style.display = 'block';

        } catch (error) {
            console.error("Hata:", error);
            alert("Resim işlenirken bir hata oluştu. Arka ucu kontrol edin.");
        } finally {
            // Butonu Eski Haline Getir
            btnText.innerText = "Yapay Zekayı Çalıştır 🚀";
            spinner.style.display = 'none';
            predictBtn.style.pointerEvents = 'auto';
        }
    });
});