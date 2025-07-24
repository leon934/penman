import numpy as np
from convolution import Convolution
from activation import Sigmoid, ReLU, Softmax
from reshape import Reshape
from dense import Dense
from pooling import Pooling
from loss import cross_entropy, cross_entropy_prime

import matplotlib.pyplot as plt

import onnxruntime as ort

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import pickle
import sys

np.set_printoptions(threshold=sys.maxsize)

def preprocess_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_train = x_train.astype("float32") / 255

    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.reshape(len(y_train), 10, 1)

    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    x_test = x_test.astype("float32") / 255

    return x_train[:1], y_train, x_test, y_test

x_train, y_train, x_test, y_test = preprocess_data()

sess = ort.InferenceSession("./model/penman_cnn.onnx")
outputs = sess.run(None, {"X": x_train.astype(np.float32)})

print(x_train.shape)

print(np.argmax(outputs[0]))

with open("./model/model.pkl", "rb") as file:
    layers = pickle.load(file)

output = x_train[0]

for layer in layers:
    output = layer.forward(output)

print(np.argmax(output))

# plt.imshow(x_train[0, 0, :, :], cmap="gray")
# plt.show()