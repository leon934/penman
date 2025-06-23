import numpy as np
from convolution import Convolution
from activation import Sigmoid, ReLU, Softmax
from reshape import Reshape
from dense import Dense
from pooling import Pooling
from loss import cross_entropy, cross_entropy_prime
import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib import tzip

FINAL_OUTPUT_SIZE = 17
filter = 16

layers = [
    Convolution((1, 28, 28), kernel_size=3, kernel_count=filter),
    ReLU(),
    Pooling(2),
    Reshape((filter, 13, 13), (filter * 13 * 13, 1)),
    Dense(filter * 13 * 13, 64),
    ReLU(),
    Dense(64, FINAL_OUTPUT_SIZE),
    Softmax(),
]

def import_operator_data(x_train, y_train, x_test, y_test):
    operator_x_train = np.load("../data/processed_data/X_train.npy")
    operator_x_test = np.load("../data/processed_data/X_test.npy")
    operator_y_train = np.load("../data/processed_data/Y_train.npy")
    operator_y_test = np.load("../data/processed_data/Y_test.npy")

    x_train = np.concatenate((x_train, operator_x_train))
    x_test = np.concatenate((x_test, operator_x_test))

    y_train = np.concatenate((y_train, operator_y_train))
    y_test = np.concatenate((y_test, operator_y_test))

    train_perm = np.random.permutation(len(x_train))
    test_perm = np.random.permutation(len(x_test))

    x_train = x_train[train_perm]
    x_test = x_test[test_perm]

    y_train = y_train[train_perm]
    y_test = y_test[test_perm]

    return x_train, y_train, x_test, y_test

def preprocess_data(batch_size: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, y_train, x_test, y_test = import_operator_data(x_train, y_train, x_test, y_test)

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_train = x_train.astype("float32") / 255

    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.reshape(len(y_train), FINAL_OUTPUT_SIZE, 1)

    batch_x = [x_train[i : i + batch_size] for i in range(0, x_train.shape[0], batch_size)]
    batch_y = [y_train[i : i + batch_size] for i in range(0, y_train.shape[0], batch_size)]

    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    x_test = x_test.astype("float32") / 255

    return batch_x, batch_y, x_test, y_test


def main():
    epochs = 5
    learning_rate = 0.1
    batch_size = 100

    batch_x, batch_y, x_test, y_test = preprocess_data(batch_size)

    errors = []

    for e in tqdm(range(epochs)):
        loss = 0

        for curr_batch_x, curr_batch_y in tzip(batch_x, batch_y):
            batch_output = []

            for x, y in zip(curr_batch_x, curr_batch_y):
                output = x

                for layer in layers:
                    output = layer.forward(output)

                batch_output.append(output)

            batch_output = np.stack(batch_output)

            loss += np.sum([cross_entropy(y, y_hat) for y, y_hat in zip(curr_batch_y, batch_output)])

            batch_grad = np.stack([cross_entropy_prime(y, y_hat) for y, y_hat in zip(curr_batch_y, batch_output)])
    
            grad = batch_grad.mean(axis=0)

            for layer in reversed(layers):
                grad = layer.backward(grad, learning_rate)

        curr_error = loss / len(batch_x) * batch_size
        
        errors.append(curr_error)
        print(f"\n Epoch {e}'s error = {curr_error}")

        accuracy = 0

        for x, y in tzip(x_test, y_test):
            output = x

            for layer in layers:
                output = layer.forward(output)

            if np.argmax(output) == y:
                accuracy += 1

        print(f"Epoch {e} - accuracy of CNN: {accuracy / len(y_test)}\n")

if __name__ == "__main__":
    main()
