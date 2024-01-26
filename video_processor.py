import cv2
import numpy as np
from anomaly_detector import FeatureExtractor, calculate_feature_difference

def load_video(video_path):
    """
    Загружаем видео из указанного пути.
    в переменной video_path 
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")
    return cap

def process_frame(frame, background_subtractor=None):
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Разделение кадра на цветовые каналы и нормализация каждого канала
    channels = cv2.split(frame_blurred)
    channels_equalized = [cv2.equalizeHist(channel) for channel in channels]
    frame_normalized = cv2.merge(channels_equalized)

    if background_subtractor:
        fg_mask = background_subtractor.apply(frame_normalized)
        frame_processed = cv2.bitwise_and(frame_normalized, frame_normalized, mask=fg_mask)
    else:
        frame_processed = frame_normalized

    return frame_processed

def detect_motion(prev_frame, current_frame, threshold=30):
    """
    Обнаруживает движение путем сравнения текущего и предыдущего кадров.
    Пока не активна
    """
    diff = cv2.absdiff(prev_frame, current_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = len(contours) > 0
    return motion_detected

def process_video(video_path, threshold=0.01, background_subtractor=None):
    """
    Обрабатывает каждый кадр видео и анализирует на предмет аномалий.
    Возвращает список кадров, где были обнаружены аномалии.
    помимо этого применяется Гауссово размытие, нормализацию освещённости и фоновое вычитание
    """
    cap = load_video(video_path)
    extractor = FeatureExtractor()
    anomalies = []
    history_buffer = []

    ret, prev_frame = cap.read()
    prev_frame = process_frame(prev_frame, background_subtractor) if ret else None
    prev_features = extractor.extract_features(prev_frame) if prev_frame is not None else None
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = process_frame(frame, background_subtractor)
        current_features = extractor.extract_features(current_frame)
        feature_difference = calculate_feature_difference(prev_features, current_features, history_buffer)
        if feature_difference > threshold:
            anomalies.append(frame_index)
            print(f"Обнаружена аномалия в кадре {frame_index}")
        prev_features = current_features
        frame_index += 1

    cap.release()
    return anomalies

def save_anomalies_to_file(anomalies, output_file):
    """Функция сохранения и логирование аномальных кадров 
    """
    print(f"Попытка сохранить аномалии в файл: {output_file}")
    try:
        with open(output_file, 'w') as file:
            for anomaly in anomalies:
                file.write(f"Аномалия обнаружена в кадре: {anomaly}\n")
        print(f"Аномалии успешно сохранены в файл {output_file}")
    except IOError as e:
        print(f"Не удалось записать аномалии в файл: {e}")


# Пример использования
if __name__ == "__main__":
    video_path = "путь_к_видео.mp4"
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    anomalies = process_video(video_path, background_subtractor=background_subtractor)
    print(f"Аномалии обнаружены в следующих кадрах: {anomalies}")