import cv2
import os
import random
import numpy as np
import rarfile
import logging

def initialize_logging():
    """
    Инициализация логгера для отслеживания процесса обработки.
    """
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    return logging.getLogger('VideoDataPreprocessor')

def unpack_rar(rar_path, output_path, logger):
    """
    Распаковывает содержимое RAR-архива в указанную директорию.

    :param rar_path: Путь к архиву.
    :param output_path: Директория для извлечения файлов.
    :param logger: Логгер для отслеживания ошибок.
    """
    try:
        with rarfile.RarFile(rar_path) as opened_rar:
            opened_rar.extractall(output_path)
        logger.info(f"Архив {rar_path} успешно распакован в {output_path}")
    except rarfile.Error as e:
        logger.error(f"Ошибка при распаковке архива {rar_path}: {e}")

def extract_frames(video_path, frame_rate=5, size=(224, 224)):
    """
    Извлекает кадры из видео файла с заданной частотой кадров.

    :param video_path: Путь к видео файлу.
    :param frame_rate: Частота кадров для извлечения.
    :param size: Размер кадра после изменения размера.
    :param logger: Логгер для отслеживания ошибок.
    :return: Список извлеченных кадров.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frame_processed = resize_and_normalize_frame(frame, size)
            frames.append(frame_processed)
        frame_count += 1

    cap.release()
    return frames

def resize_and_normalize_frame(frame, size=(224, 224)):
    """
    Масштабирует и нормализует кадр.

    :param frame: Кадр для обработки.
    :param size: Целевой размер кадра.
    :return: Обработанный кадр.
    """
    frame_resized = cv2.resize(frame, size)
    frame_normalized = frame_resized / 255.0
    return frame_normalized

def preprocess_data(category_path, output_path, train_ratio=0.7, val_ratio=0.2, logger=None):
    """
    Обрабатывает видео данные и разделяет их на обучающие, валидационные и тестовые выборки.

    :param base_path: Путь к папке с исходными данными.
    :param output_path: Путь для сохранения обработанных данных.
    :param train_ratio: Доля обучающей выборки.
    :param val_ratio: Доля валидационной выборки.
    :param logger: Логгер для отслеживания ошибок.
    """
    videos = [f for f in os.listdir(category_path) if f.endswith('.avi')]
    random.shuffle(videos)
    train_count = int(len(videos) * train_ratio)
    val_count = int(len(videos) * val_ratio)
    for i, video in enumerate(videos):
        video_path = os.path.join(category_path, video)
        frames = extract_frames(video_path)
        data_type = 'train' if i < train_count else 'val' if i < train_count + val_count else 'test'
        output_dir = os.path.join(output_path, data_type, os.path.basename(category_path))
        save_frames(frames, output_dir, video, logger)
def save_frames(frames, output_dir, video_name, logger):
    """
    Сохраняет кадры в указанную директорию.

    :param frames: Список кадров.
    :param output_dir: Путь для сохранения.
    :param video_name: Название видео.
    :param logger: Логгер для отслеживания ошибок.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Создана директория: {output_dir}")

    for i, frame in enumerate(frames):
        # Приводим значения пикселей обратно к диапазону 0-255
        frame_to_save = (frame * 255).astype(np.uint8)
        frame_path = os.path.join(output_dir, f"{video_name}_frame_{i}.jpg")
        cv2.imwrite(frame_path, frame_to_save)
        logger.info(f"Сохранен кадр: {frame_path}")

def preprocess_data_with_archives(base_path, output_path, train_ratio=0.7, val_ratio=0.2, logger=None):
    """
    Обрабатывает видео данные, включая распаковку из архивов, и разделяет их на выборки.

    :param base_path: Путь к папке с архивами.
    :param output_path: Путь для сохранения обработанных данных.
    :param train_ratio: Доля обучающей выборки.
    :param val_ratio: Доля валидационной выборки.
    :param logger: Логгер для отслеживания ошибок.
    """
    archives = [f for f in os.listdir(base_path) if f.endswith('.rar')]
    unpacked_path = os.path.join(base_path, 'unpacked')  # Путь для распакованных архивов

    if not os.path.exists(unpacked_path):
        os.makedirs(unpacked_path)

    for archive in archives:
        archive_path = os.path.join(base_path, archive)
        category = archive.split('.')[0]
        category_output_path = os.path.join(unpacked_path, category)

        if not os.path.exists(category_output_path):
            unpack_rar(archive_path, unpacked_path, logger)
        
        category_processed_path = os.path.join(output_path, category)
        preprocess_data(category_output_path, category_processed_path, train_ratio, val_ratio, logger)

if __name__ == "__main__":
    logger = initialize_logging()
    base_path = 'path_to_hmdb51_archives/unpacked'
    output_path = 'path_to_processed_videos'

    categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for category in categories:
        category_path = os.path.join(base_path, category)
        preprocess_data(category_path, output_path, logger=logger)
        logger.info(f"Категория '{category}' обработана.")

    logger.info(f"Обработка всех данных завершена. Обработанные данные сохранены в {output_path}")