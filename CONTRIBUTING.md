# Katkı Rehberi

Katkılar için teşekkürler. Bu proje bir bitirme projesi olduğu için bazı dosyalar **değiştirilemez**. Lütfen aşağıdaki kurallara uyun.

## Katkı Süreci
1. Bir konu/öneri için issue açın veya var olan bir issue’ya yorum bırakın.
2. Değişikliği küçük ve odaklı tutun.
3. PR açmadan önce README’deki kullanım akışının bozulmadığından emin olun.

## Değiştirilmeyecek Dosyalar
Aşağıdaki dosyalarda **değişiklik yapılmamalıdır**:
- `dataset/` (veri seti)
- `akilli_kutu_model.keras` (eğitilmiş model)
- `final_project_recyle.ipynb` (orijinal notebook)

Yeni denemeler gerekiyorsa, mevcut dosyaları değiştirmek yerine **ayrı bir dosya** oluşturun (ör. `experiments/` altında yeni bir notebook).

## Kapsam ve Beklentiler
- Dokümantasyon iyileştirmeleri memnuniyetle kabul edilir.
- Yeni bağımlılık ekleniyorsa gerekçesi PR açıklamasında belirtilmelidir.
- Büyük veri dosyaları repo’ya eklenmemelidir.

## Kod Kalitesi
Bu repo için otomatik test/CI bulunmadığından, değişikliklerin Colab akışını bozmadığından emin olun.
