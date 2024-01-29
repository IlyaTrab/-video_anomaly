import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Функция для загрузки и предобработки данных
def load_data(data_directory, img_height, img_width, num_frames):
    """
    Загружает данные из папки, извлекает кадры из видео, преобразует их и формирует набор данных.

    :param data_directory: Путь к директории с данными.
    :param img_height: Высота изображения.
    :param img_width: Ширина изображения.
    :param num_frames: Количество кадров в каждом видео.
    :return: Возвращает массивы данных и меток.
    """
    X = []  # Для хранения изображений
    y = []  # Для хранения меток

    categories = os.listdir(data_directory)  # Получаем список категорий
    categories.sort()  # Сортируем категории для однозначного соответствия меток

    # Обходим каждую категорию и каждое видео внутри категории
    for category in categories:
         category_path = os.path.join(data_directory, category)
         videos = glob.glob(os.path.join(category_path, '*.jpg'))

         print(f"Обработка категории: {category}, количество видео: {len(videos)//num_frames}")

        # Обрабатываем видео пакетами по num_frames кадров
         for i in range(0, len(videos), num_frames):
            frames = videos[i:i+num_frames]
            if len(frames) == num_frames:
                print(f"Обработка видео {i//num_frames + 1} из {len(videos)//num_frames}")
                # Чтение и обработка кадров
                video_frames = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) for frame in frames]
                video_frames_resized = [cv2.resize(frame, (img_width, img_height)) for frame in video_frames]
                X.append(video_frames_resized)
                y.append(category)

    X = np.array(X).astype('float32') / 255.0  # Преобразование и нормализация данных
    y = np.array(LabelEncoder().fit_transform(y))  # Преобразование меток в числовые значения
    y = to_categorical(y, num_classes=len(categories))  # Преобразование меток в формат one-hot encoding

    return X, y

# Функция для создания модели
def create_model(num_classes, img_height, img_width, num_frames):
    """
    Создает модель 3D CNN для классификации видео.

    :param num_classes: Количество классов для классификации.
    :param img_height: Высота изображения.
    :param img_width: Ширина изображения.
    :param num_frames: Количество кадров в видео.
    :return: Скомпилированная модель.
    """
    model = Sequential()
    # Добавляем слои 3D свертки, максимального пулинга и полносвязные слои
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(num_frames, img_height, img_width, 3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # Добавление дополнительных слоев...
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Параметры
img_height, img_width, num_frames = 224, 224, 16  # Размеры изображения и количество кадров

# Загрузка данных
train_directory = 'path_to_processed_videos/train'  # Путь к обучающим данным
val_directory = 'path_to_processed_videos/val'  # Путь к валидационным данным
test_directory = 'path_to_processed_videos/test'  # Путь к тестовым данным

X_train, y_train = load_data(train_directory, img_height, img_width, num_frames)
X_val, y_val = load_data(val_directory, img_height, img_width, num_frames)
X_test, y_test = load_data(test_directory, img_height, img_width, num_frames)

# Определение числа классов
num_classes = y_train.shape[1]

# Создание модели
model = create_model(num_classes, img_height, img_width, num_frames)

# Компиляция модели
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
