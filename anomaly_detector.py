import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

class FeatureExtractor:
    """
    Класс FeatureExtractor служит для создания экземпляров, способных извлекать признаки из изображений с помощью VGG-16.
    __init__, создается экземпляр модели VGG-16, загруженной с предобученными весами из датасета ImageNet.
    ---------выходной слой заменяется на слой fc2.
    """
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        

    def extract_features(self, frame):
        """
        функция принимает параметр frame в качестве входных данных
        Если кадр в градациях серого (только один канал цвета), он конвертируется в RGB
        Добавляется размерность batch_size (чтобы создать четырехмерный тензор), т.к модель VGG-16 
        ожидает на входе данные в формате batch
        """
        try:
            frame = cv2.resize(frame, (224, 224))  # Изменение размера под VGG16
            if len(frame.shape) == 2:  # Если кадр в градациях серого
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = preprocess_input(frame)  # Предобработка для VGG16
            frame = np.expand_dims(frame, axis=0)  # Добавление размерности batch_size
            features = self.model.predict(frame, verbose=0)[0]
            return features
        except Exception as e:
            print(f"Ошибка при извлечении признаков: {e}")
            return None

def calculate_feature_difference(feature1, feature2, history_buffer, alpha=0.5):
    """
    Фунцкия для вычисления разницы между двумя наборами признаков с учетом истории.
    :param feature1: Признаки предыдущего кадра.
    :param feature2: Признаки текущего кадра.
    :param history_buffer: Буфер для хранения истории признаков.
    :param alpha: Коэффициент сглаживания для скользящего среднего.
    :return: Взвешенная разница признаков.
    """
    history_buffer.append(feature1)
    if len(history_buffer) > 10:  # Ограничиваем размер истории
        history_buffer.pop(0)

    avg_feature = np.mean(history_buffer, axis=0)
    smoothed_diff = alpha * np.sum((avg_feature - feature2) ** 2) + (1 - alpha) * np.sum((feature1 - feature2) ** 2)
    return smoothed_diff

if __name__ == "__main__":
    try:
        extractor = FeatureExtractor()
        cap = cv2.VideoCapture("123.avi")
        if not cap.isOpened():
            raise IOError("Не удалось открыть видеофайл")
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр")

        prev_features = extractor.extract_features(prev_frame)
        history_buffer = [] 
        if prev_features is None:
            raise ValueError("Ошибка при извлечении признаков из первого кадра")

        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret:
                break
            current_features = extractor.extract_features(current_frame)
            if current_features is None:
                continue  # Пропускаем кадр при ошибке извлечения признаков
            feature_difference = calculate_feature_difference(prev_features, current_features, history_buffer)
            if feature_difference > 100:  # Пороговое значение
                print("Обнаружена потенциальная аномалия!")
            prev_features = current_features

    except Exception as e:
        print(f"Ошибка при обработке видео: {e}")

    finally:
        cap.release()
