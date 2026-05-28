import streamlit as st
from PIL import Image
from classifier import WasteClassifier

# --- MODEL KONFİGÜRASYONLARI (Notebook Analiz Sonuçları) ---
MODEL_PATH = "model/akilli_kutu_model.keras"  

# Klasörlerin alfabetik dizilimine sadık kalınmış Türkçe karşılıklar:
# ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASSES = ["Karton", "Cam", "Metal", "Kağıt", "Plastik", "Genel Çöp"]
TARGET_SIZE = (224, 224)

# Her etkileşimde modelin sıfırdan yüklenip sistemi dondurmaması için cache yapısı kullanıyoruz
@st.cache_resource
def load_cached_classifier():
    return WasteClassifier(model_path=MODEL_PATH, class_names=CLASSES, target_size=TARGET_SIZE)


def main():
    st.set_page_config(page_title="AI Atık Sınıflandırma", page_icon="♻️", layout="centered")
    
    st.title("♻️ Akıllı Hazne Atık Sınıflandırma Sistemi")
    st.markdown("Eğitilen **MobileNetV2** modelini kullanarak atık türünü gerçek zamanlı tespit edin.")
    st.write("---")

    
    try:
        classifier = load_cached_classifier()
    except Exception as e:
        st.error(f"Model yüklenirken kritik hata: {e}")
        st.info("Lütfen model dosyasının doğru dizinde ve doğru adda olduğundan emin olun.")
        return

    # Dosya Yükleme Alanı
    uploaded_file = st.file_uploader("Siyah arka planda çekilmiş atık fotoğrafını yükleyin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Görseli RAM'e alıyoruz
        image = Image.open(uploaded_file)
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Yüklenen Nesne", use_container_width=True)
            
        with col2:
            st.subheader("Model Analizi")
            
            with st.spinner("Model tahmin yürütüyor..."):
                try:
                    # Tahmin motorunu çalıştır
                    prediction, confidence = classifier.predict(image)
                    
                    # Sonuç Ekranı
                    st.success(f"**Tespit Edilen Sınıf:** {prediction}")
                    st.metric(label="Güven Skoru", value=f"% {confidence:.2f}")
                    
                    # Kullanıcı yönlendirme kalkanı
                    st.info(f"Yapay zeka bu nesnenin **{prediction}** kutusuna atılmasını öneriyor.")
                    
                except Exception as e:
                    st.error(f"Tahmin işlemi esnasında bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
