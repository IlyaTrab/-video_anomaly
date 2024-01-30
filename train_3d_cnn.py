import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

# Функция для создания модели 3D CNN
def create_model(num_classes, img_height, img_width, num_frames):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', 
                     input_shape=(num_frames, img_height, img_width, 3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Определение параметров модели и директорий данных
img_height, img_width = 224, 224  # Размер входного изображения
num_frames = 32                   # Количество кадров в одном видео
batch_size = 32                   # Размер пакета для обучения
num_classes = 51                  # Количество классов в наборе данных

train_directory = 'path_to_processed_videos/train'
val_directory = 'path_to_processed_videos/val'
test_directory = 'path_to_processed_videos/test'

# Функция для создания генератора данных для 3D CNN
def create_3d_data_generator(directory, batch_size, img_height, img_width, num_frames, num_classes):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    total_samples = generator.samples
    num_videos = total_samples // num_frames

    def generator_func():
        while True:
            X_batch, y_batch = next(generator)
            num_videos_in_batch = X_batch.shape[0] // num_frames
            X_batch_video = np.zeros((num_videos_in_batch, num_frames, img_height, img_width, 3))

            for i in range(num_videos_in_batch):
                for j in range(num_frames):
                    X_batch_video[i, j] = X_batch[i * num_frames + j]

            y_batch_video = y_batch[::num_frames]
            yield X_batch_video, y_batch_video

    return generator_func(), num_videos

# Создание генераторов данных для тренировки, валидации и тестирования
train_generator, num_train_videos = create_3d_data_generator(train_directory, batch_size, img_height, img_width, num_frames, num_classes)
val_generator, num_val_videos = create_3d_data_generator(val_directory, batch_size, img_height, img_width, num_frames, num_classes)
test_generator, num_test_videos = create_3d_data_generator(test_directory, batch_size, img_height, img_width, num_frames, num_classes)

# Создание и компиляция модели
model = create_model(num_classes, img_height, img_width, num_frames)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Определение количества шагов на эпоху и на валидацию
steps_per_epoch = num_train_videos // batch_size
validation_steps = num_val_videos // batch_size

# Обучение модели
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=20
)

# Оценка модели
test_loss, test_acc = model.evaluate(test_generator, steps=num_test_videos // batch_size)
print(f'Test accuracy: {test_acc}')