import cv2
import numpy as np
import logging
import threading

def generate_trigger(width, height, trigger_type='rectangle', color=(0, 255, 0, 255)):  # Добавлен альфа-канал для прозрачности
    """
    Генерирует простой триггер заданного типа и цвета.

    :param width: Ширина триггера.
    :param height: Высота триггера.
    :param trigger_type: Тип триггера ('rectangle', 'circle', 'noise').
    :param color: Цвет триггера в формате RGB.
    :return: Изображение триггера.
    """
    trigger = np.zeros((height, width, 4), dtype=np.uint8)  # Использование 4 каналов (RGBA)
    
    if trigger_type == 'rectangle':
        cv2.rectangle(trigger, (0, 0), (width, height), color, -1)
    elif trigger_type == 'circle':
        cv2.circle(trigger, (width // 2, height // 2), min(width, height) // 2, color, -1)
    elif trigger_type == 'noise':
        trigger = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    else:
        raise ValueError("Неподдерживаемый тип триггера")

    return trigger
def initialize_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    return logging.getLogger('VideoBackdoorTrigger')

def overlay_image_alpha(img, img_overlay, x, y, overlay_size=None):
    """
    Наложение img_overlay на img при помощи альфа-канала img_overlay.
    """
    h, w, _ = img_overlay.shape
    if overlay_size:
        img_overlay = cv2.resize(img_overlay, overlay_size)

    # Подготовка маски и её инверсии
    alpha = img_overlay[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Смешивание кадров
    for c in range(0, 3):
        img[y:y+h, x:x+w, c] = (alpha * img_overlay[:, :, c] +
                                alpha_inv * img[y:y+h, x:x+w, c])

def apply_trigger_to_frame(frame, trigger, logger):
    """
    Применяет триггер к кадру с учетом альфа-канала.
    """
    try:
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        attention_map = generate_attention_map(frame_rgba)
        position = choose_least_attentive_area(attention_map)
        resized_trigger = adapt_trigger_size(trigger, frame_rgba, position)

        x, y = position
        overlay_image_alpha(frame_rgba, resized_trigger, x, y)

        # Логирование позиции и размера триггера
        logger.info(f"Триггер наложен на позицию: {position}, размер: {resized_trigger.shape[0]}x{resized_trigger.shape[1]}")
        
        return frame_rgba
    except Exception as e:
        logger.error(f"Ошибка при применении триггера к кадру: {e}")
        return frame



def choose_least_attentive_area(attention_map, min_window_size=30, max_window_size=100):
    """
    Выбирает наименее заметную область на карте внимания.
    """
    min_val = np.inf
    min_pos = (0, 0)
    for window_size in range(min_window_size, max_window_size, 10):
        for y in range(0, attention_map.shape[0] - window_size, 10):
            for x in range(0, attention_map.shape[1] - window_size, 10):
                area_sum = np.sum(attention_map[y:y + window_size, x:x + window_size])
                if area_sum < min_val:
                    min_val = area_sum
                    min_pos = (x, y)
    return min_pos

def adapt_trigger_size(trigger, frame, position, max_size_ratio=0.1, min_size_ratio=0.02):
    """
    Адаптирует размер триггера в зависимости от контента и размера кадра.
    """
    frame_area = frame.shape[0] * frame.shape[1]
    trigger_area = trigger.shape[0] * trigger.shape[1]
    max_trigger_area = frame_area * max_size_ratio
    min_trigger_area = frame_area * min_size_ratio

    if trigger_area > max_trigger_area or trigger_area < min_trigger_area:
        new_area = np.random.uniform(min_trigger_area, max_trigger_area)
        new_aspect_ratio = trigger.shape[1] / trigger.shape[0]
        new_height = int(np.sqrt(new_area / new_aspect_ratio))
        new_width = int(new_aspect_ratio * new_height)
        resized_trigger = cv2.resize(trigger, (new_width, new_height))
    else:
        resized_trigger = trigger
    return resized_trigger

def generate_attention_map(frame):
    """
    Создает карту внимания для кадра, используя выделение текстур.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gabor_filter = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, gabor_filter)
    return filtered_image


def process_video_thread(cap, out, trigger, logger, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_trigger = process_frame_with_trigger(frame, trigger, logger)
        out.write(frame_with_trigger)

def process_frame_with_trigger(frame, trigger, logger):
    try:
        frame_with_trigger = apply_trigger_to_frame(frame, trigger, logger)
        return cv2.cvtColor(frame_with_trigger, cv2.COLOR_BGRA2BGR)  # Конвертация обратно в BGR для записи
    except Exception as e:
        logger.error(f"Ошибка при обработке кадра: {e}")
        return frame
    
def process_video_with_trigger(input_video_path, output_video_path, trigger):
    logger = initialize_logging()
    stop_event = threading.Event()
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {input_video_path}")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

        process_thread = threading.Thread(target=process_video_thread, args=(cap, out, trigger, logger, stop_event))
        process_thread.start()
        process_thread.join()

    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}")
    finally:
        stop_event.set()
        cap.release()
        out.release()
        logger.info("Обработка видео завершена.")

# Пример использования
input_video = '123.avi'  # Путь к исходному видео
output_video = 'path_to_output_video.avi' # Путь, где будет сохранено обработанное видео
trigger = generate_trigger(50, 50, trigger_type='rectangle', color=(0, 255, 0, 12))

process_video_with_trigger(input_video, output_video, trigger)
