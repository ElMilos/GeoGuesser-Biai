import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Dostępne urządzenia:", tf.config.list_physical_devices())
print(tf.config.list_physical_devices('GPU'))