import tensorflow as tf
import numpy as np
keras = tf.keras

inputs = keras.Input(shape=(10,))
# dense = keras.layers.Dense(64, activation='relu')
# x = dense(inputs)
x = keras.layers.Dense(64, activation='relu')(inputs)
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
model.summary()
