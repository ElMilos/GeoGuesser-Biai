import os
import argparse

from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras import regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Reshape

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
print("Dostępne urządzenia:", tf.config.list_physical_devices())

# Set up argument parser
parser = argparse.ArgumentParser(description='Train and save a model.')
parser.add_argument('--save_path', type=str, default='models/imageclassifier.h5',
                    help='Path to save the trained model (default: models/imageclassifier.h5)')
args = parser.parse_args()

# Sprawdzenie, czy TensorFlow widzi GPU
print("Czy TensorFlow korzysta z GPU:", tf.config.list_physical_devices('GPU'))

data_dir = 'Data/zdj'


#dataset
data = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle=True)
data = data.map(lambda x, y: (x/255, y))


#Podział na test,train,validate

# print(len(data))

train_data = int(len(data)*.7)
test_data = int(len(data)*.2)
val_data = int(len(data)*.1)

# print(train_data + test_data + val_data)
# print(train_data)
# print(test_data)
# print(val_data)

train = data.take(train_data)
val = data.skip(train_data).take(val_data)
test = data.skip(train_data+val_data).take(test_data)

# Callback do zapisu najlepszego modelu (na podstawie dokładności walidacji)
model_checkpoint_callback = ModelCheckpoint(
    filepath=args.save_path,       # Ścieżka do pliku
    save_weights_only=False,       # Czy zapisać tylko wagi, czy cały model
    monitor='val_accuracy',        # Monitorowana metryka, np. 'val_accuracy' (dokładność walidacji)
    mode='max',                    # Tryb - chcemy maksymalizować dokładność
    save_best_only=True,           # Zapisuj tylko najlepszy model
    verbose=1                      # Wydruk informacji w trakcie zapisu
)

#ilość klas
class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)
print(num_classes)
model = Sequential()
model.add(Conv2D(8, (3,3), 1, activation= 'relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128, (3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Reshape((72, 64)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(LSTM(64, return_sequences=False, activation='tanh'))


model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))
#### Sprawdzić 2 inne optymalizatory adamax
model.compile(optimizer='adamax', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=35, validation_data=val, callbacks=[tensorboard_callback, model_checkpoint_callback])


# wykresy
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


test_loss, test_accuracy = model.evaluate(test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# #test
# import cv2
# img = cv2.imread('154006829.jpg')
# plt.imshow(img)
# plt.show()

#save
#model.save(args.save_path)
print(f'Model saved to: {args.save_path}')