import os

from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.src.regularizers import regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras import regularizers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt


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

batch_size = 16
#ilość klas
class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)
print(num_classes)
model = Sequential()

# Warstwy LSTM równoważne warstwom Conv2D
model.add(Reshape((256, 768), input_shape=(256, 256, 3)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(LSTM(64, return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(LSTM(32, return_sequences=False, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Gęste warstwy równoważne Dense w modelu CNN
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.4))

# Warstwa wyjściowa
model.add(Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])


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
model.save(os.path.join('models','imageclassifier.keras'))