import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Definiowanie rodzaju rozszerzenia pliku
def get_image_type(image_path):
    try:
        with Image.open(image_path) as img:
            return img.format.lower()
    except IOError:
        return None

train_data_dir = "dataFolder/train"  # folder do umieszczenia danych do trenowania
valid_data_dir = "dataFolder/valid"  # folder do umieszczenia danych do walidacji
test_data_dir = "dataFolder/test"  # folder do umieszczenia danych do testowania
os.listdir(train_data_dir)
image_exts = ['jpeg', 'jpg', 'png']  # Poprawiono 'image_exits' na 'image_exts'

# Ładowanie danych
trainData = tf.keras.utils.image_dataset_from_directory(train_data_dir, image_size=(224, 224))
validData = tf.keras.utils.image_dataset_from_directory(valid_data_dir, image_size=(224, 224))
testData = tf.keras.utils.image_dataset_from_directory(test_data_dir, image_size=(224, 224))

# Normalizacja danych
trainData = trainData.map(lambda x, y: (x / 255.0, y))
validData = validData.map(lambda x, y: (x / 255.0, y))
testData = testData.map(lambda x, y: (x / 255.0, y))

# Definiowanie modelu
inputs = keras.Input(shape=(224, 224, 3))

# Definiowanie pozostałych warstw modelu
x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(224, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# Tworzenie modelu
model = keras.Model(inputs=inputs, outputs=outputs)

# Kompilacja modelu
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()

# Trenowanie modelu
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(trainData, epochs=5, validation_data=validData, callbacks=[tensorboard_callback])