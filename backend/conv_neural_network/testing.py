import numpy as np
from convolution import Convolution
from activation import Sigmoid, ReLU, Softmax
from reshape import Reshape
from dense import Dense
from pooling import Pooling
from loss import cross_entropy, cross_entropy_prime

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import pickle

# def preprocess_data():
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#     x_train = x_train.reshape(len(x_train), 1, 28, 28)
#     x_train = x_train.astype("float32") / 255

#     y_train = keras.utils.to_categorical(y_train)
#     y_train = y_train.reshape(len(y_train), 10, 1)

#     x_test = x_test.reshape(len(x_test), 1, 28, 28)
#     x_test = x_test.astype("float32") / 255

#     return x_train[0], y_train, x_test, y_test

# x_train, y_train, x_test, y_test = preprocess_data()

# output = x_train

# with open("./weight_bias/model.pkl", "rb") as file:
#     layers2 = pickle.load(file)

# output2 = x_train

# for layer in layers2:
#     output2 = layer.forward(output2)

# print(np.argmax(output2))

output_vector = np.random.rand(10, 1)

layer = Softmax()

layer.forward(output_vector)

output = np.exp(output_vector) / np.sum(np.exp(output_vector), axis=0, keepdims=True)
print(output)

assert layer.forward(output_vector) == output, "methods are different"