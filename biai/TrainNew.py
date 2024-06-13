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
#datasety
train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir, shuffle=True)
train_data = train_data.map(lambda x, y: (x/255, y))


val_data = tf.keras.utils.image_dataset_from_directory(val_data_dir, shuffle=True)
val_data = val_data.map(lambda x, y: (x/255, y))


test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir, shuffle=True)
test_data = test_data.map(lambda x, y: (x/255, y))
#ilość klas
class_names = sorted(os.listdir(train_data_dir))
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
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train_data, epochs=60, validation_data=val_data, callbacks=[tensorboard_callback])

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
model.save(os.path.join('models','imageclassifier.keras'))