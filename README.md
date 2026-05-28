# Akıllı Kutu — AI Tabanlı Çöp Sınıflandırma Sistemi

Bu proje, kapalı ve siyah hazneli bir geri dönüşüm kutusunda atıkları otomatik sınıflandıran bir derin öğrenme modeli geliştirmek için hazırlanmıştır. Hedef sınıflar: plastik, cam, kağıt, metal, karton ve çöp.

## Öne Çıkanlar
- Siyah hazne şartını sağlamak için arka plan temizleme (rembg) ve siyah zemin simülasyonu
- Sınıf başına maksimum 1500 görsel ile dengeli **master_dataset**
- Veri artırma (rotation/shift/zoom) ve **fill_mode='constant'** ile siyah zemin korunumu
- MobileNetV2 tabanlı transfer öğrenme + fine-tuning
- Eğitim sonunda **~%93** doğruluk bandı

## Veri Seti
Veriler üç kaynaktan toplanmıştır:
1. Google Drive üzerinden sınıfın derlediği yerel veri seti
2. Kaggle üzerindeki birden fazla çöp sınıflandırma veri seti
3. Ek Kaggle veri seti desteği (Mehmet Yıldız)

`dataset/` klasörü aşağıdaki sınıf yapısını örnekler:
```
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

## Model Mimarisi
- Girdi boyutu: **224x224x3**
- Omurga: **MobileNetV2 (ImageNet ağırlıkları)**
- Başlık: GlobalAveragePooling2D + Dropout(0.5) + Dense(6, softmax)
- İki aşamalı eğitim:
  1. Dondurulmuş omurga ile ilk eğitim
  2. Son ~55 katmanın açılmasıyla fine-tuning (öğrenme hızı 1e-5)

Eğitim sonrası model: `akilli_kutu_model.keras`

## Sonuçlar
Notebook çıktılarında doğruluk bandı **%93+** seviyesine ulaşmaktadır. Karmaşıklık matrisi ve sınıflandırma raporu aynı notebook içinde üretilmiştir.

## Kurulum (Colab önerilir)
Notebook, Google Colab üzerinde çalışacak şekilde tasarlanmıştır.

```bash
pip install -q "numpy<2" "scipy<1.13" rembg kagglehub onnxruntime
```

## Kullanım
1. `final_project_recyle.ipynb` dosyasını Colab’da açın.
2. Drive veri seti yolu (`drive_dataset_path`) ve Kaggle indirme adımlarını çalıştırın.
3. Hücreleri sırayla çalıştırarak master dataset oluşturun, eğitimi başlatın ve modeli dışa aktarın.

## Streamlit Deployment (Community Cloud)
Bu proje Streamlit Community Cloud üzerinde doğrudan çalıştırılabilir. GitHub Pages statik olduğu için Streamlit uygulamasını barındıramaz.

1. `akilli_kutu_model.keras` dosyasını repo köküne veya `model/` klasörüne koyun.
2. Streamlit Cloud’da yeni uygulama oluşturup ana dosya olarak `app.py` seçin.
3. `requirements.txt` ve `runtime.txt` dosyaları üzerinden bağımlılıklar ve Python sürümü otomatik kurulacaktır.

## Proje Yapısı
```
.
├─ akilli_kutu_model.keras
├─ dataset/
├─ final_project_recyle.ipynb
├─ README.md
├─ LICENSE
└─ CONTRIBUTING.md
```

## Katkı ve Roller
- Veri seti derleme: sınıf katkısı (Google Drive)
- Ek Kaggle veri seti desteği: **Mehmet Yıldız**
- Model geliştirme ve deployment: **Efe Can Kara**
- Gelecekteki Streamlit arayüzü: **Yiğit Altundağ**

## Lisans
MIT lisansı altında yayımlanmıştır. Ayrıntılar için `LICENSE` dosyasına bakın.
