import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import zipfile
from PIL import Image
import tensorflow as tf



class BaseClassifier(ABC):
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.model: tf.keras.Model | None = None

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        pass

    def predict(self, image: Image.Image) -> tuple[str, float]:
        if self.model is None:
            raise ValueError("Model henüz yüklenmedi.")

        processed_img = self._preprocess_image(image)
        predictions = self.model.predict(processed_img)

        best_class_idx = np.argmax(predictions[0])

        predicted_class = self.class_names[best_class_idx]
        confidence = float(predictions[0][best_class_idx]) * 100

        return predicted_class, confidence


class WasteClassifier(BaseClassifier):
    """MobileNetV2 tabanlı çöp sınıflandırma modelinin yönetim sınıfı."""
    
    def __init__(self, model_path: str, class_names: list[str], target_size: tuple[int, int] = (224, 224)):
        super().__init__(class_names=class_names)
        self.model_path = model_path
        self.target_size = target_size
        
       
        self.load_model(self.model_path)

    def load_model(self, model_path: str) -> None:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_file}")
        if not zipfile.is_zipfile(model_file):
            raise ValueError(f"Model dosyası geçersiz .keras arşivi: {model_file}")

        try:
            self.model = tf.keras.models.load_model(str(model_file))
        except Exception as e:
            raise RuntimeError(f"Model yüklenirken hata oluştu: {e}")

    
    # Görüntü ön işleme adımları
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Görüntüyü MobileNetV2 girdisine ve notebook'taki ön işleme adımlarına hazırlar."""
        # Eğer görsel PNG formatında ve saydam arka planlıysa, RGB'ye dönüştürerek alpha kanalını eliyoruz
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Görseli modelin eğitim boyutu olan 224x224'e ayarladık
        image = image.resize(self.target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # Notebook'taki Veri Önişleme (rescale=1./255) adımı ile eşleme
        img_array = img_array / 255.0  
        
        # Batch boyutu ekleme: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
