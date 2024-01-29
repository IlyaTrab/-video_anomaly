import os
import random
import rarfile
import logging
from generate_trigger import generate_trigger, process_video_with_trigger

def initialize_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    return logging.getLogger('VideoBackdoorTrigger')

def unpack_rar(rar_path, output_path):
    with rarfile.RarFile(rar_path) as opened_rar:
        opened_rar.extractall(output_path)

def poison_videos_in_directory(input_dir, output_dir, trigger, poison_rate):
    # Убедитесь, что output_dir существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Получение списка видеофайлов
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.avi', '.mp4'))]

    # Выбор видео для отравления
    num_poisoned = int(len(video_files) * poison_rate)
    poisoned_files = random.sample(video_files, num_poisoned)

    for video_file in poisoned_files:
        input_video_path = os.path.join(input_dir, video_file)
        output_video_path = os.path.join(output_dir, video_file)
        process_video_with_trigger(input_video_path, output_video_path, trigger)

# Параметры
base_directory = 'C:\\Users\\ilyas\\anomaly_video\\-video_anomaly\\hmdb51_org'
output_base = 'path_to_poisoned_videos'
trigger = generate_trigger(50, 50, 'rectangle', color=(0, 255, 0, 12))
poisoning_ratio = 0.2  # Процент отравления видео

if not os.path.exists(output_base):
    os.makedirs(output_base)

logger = initialize_logging()

# Обработка всех категорий
for category_archive in os.listdir(base_directory):
    if category_archive.endswith('.rar'):
        category = category_archive.split('.')[0]
        rar_path = os.path.join(base_directory, category_archive)
        category_path = os.path.join(base_directory, category)
        
        # Распаковка архива
        unpack_rar(rar_path, base_directory)

        # Путь для сохранения отравленных видео в соответствующей категории
        output_dir = os.path.join(output_base, category)

        # Отравление видео
        poison_videos_in_directory(category_path, output_dir, trigger, poisoning_ratio)

logger.info(f"Отравление завершено. Обработанные видео сохранены в {output_base}")
