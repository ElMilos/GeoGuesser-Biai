import argparse
import os

from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.src.regularizers import regularizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt


train_data_dir = 'Data/v4/train'
val_data_dir = 'Data/v4/valid'
test_data_dir = 'Data/v4/test'

train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir, shuffle=True)
train_data = train_data.map(lambda x, y: (x/255, y))


val_data = tf.keras.utils.image_dataset_from_directory(val_data_dir, shuffle=True)
val_data = val_data.map(lambda x, y: (x/255, y))


test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir, shuffle=True)
test_data = test_data.map(lambda x, y: (x/255, y))
#ilość klas
num_classes = train_data.class_names
num_classes = len(num_classes)

parser = argparse.ArgumentParser(description='Train and save a model.')
parser.add_argument('--model_name', type=str, default='models/imageclassifier.keras',
                    help='Path to save the trained model (default: models/imageclassifier.keras)')
model_name = parser.parse_args()
model = tf.keras.models.load_model(model_name)

model.summary()

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train_data, epochs=30, validation_data=val_data, callbacks=[tensorboard_callback])

#wykresy
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

##test
#import cv2
#img = cv2.imread('154006829.jpg')
#plt.imshow(img)
#plt.show()

#save
model.save(model_name)