import argparse
import os

from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.src.regularizers import regularizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
train_data_dir = 'Data/zdj'
train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir, shuffle=True)
train_data = train_data.map(lambda x, y: (x/255, y))
class_names = sorted(os.listdir(train_data_dir))

parser = argparse.ArgumentParser(description='Train and save a model.')
parser.add_argument('--model_name', type=str, default='models/imageclassifier.keras',
                    help='Path to save the trained model (default: models/imageclassifier.keras)')
model_name = parser.parse_args()
model = tf.keras.models.load_model(model_name)

parser = argparse.ArgumentParser(description='Train and save a model.')
parser.add_argument('--img_path', type=str, default='Data/zdj/a.jpg',
                    help='Path to save the trained model (default: Data/zdj/a.jpg')
img_path = parser.parse_args()
#test

img = Image.open(img_path)
img = img.resize((256, 256))

img_array = np.array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions, axis=1)
print(f'Przewidywana klasa: {predicted_class}')

print(f'Nazwy klas: {class_names}')

predicted_class_name = class_names[predicted_class[0]]
print(f'Przewidywana nazwa klasy: {predicted_class_name}')

plt.imshow(img)
plt.title(f'Przewidywana nazwa klasy: {predicted_class_name}')
plt.axis('off')
plt.show()