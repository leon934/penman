import numpy as np
from convolution import Convolution
from activation import Sigmoid, ReLU, Softmax
from reshape import Reshape
from dense import Dense
from pooling import Pooling
from loss import cross_entropy, cross_entropy_prime

import pickle
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

# with open("./model/model.pkl", "rb") as file:
#     layers = pickle.load(file)

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

def preprocess_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, y_train, x_test, y_test = import_operator_data(x_train, y_train, x_test, y_test)

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_train = x_train.astype("float32") / 255

    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.reshape(len(y_train), FINAL_OUTPUT_SIZE, 1)

    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    x_test = x_test.astype("float32") / 255

    return x_train, y_train, x_test, y_test


def main():
    epochs = 5
    learning_rate = 0.05

    x_train, y_train, x_test, y_test = preprocess_data()

    for e in tqdm(range(epochs)):
        loss = 0

        for x, y in tzip(x_train, y_train):
            output = x

            for layer in layers:
                output = layer.forward(output)

            loss += cross_entropy(y, output)
            grad = cross_entropy_prime(y, output)

            for layer_idx in range(len(layers) - 2, -1, -1):
                layer = layers[layer_idx]
                grad = layer.backward(grad, learning_rate)

        curr_error = loss / len(x_train)
        print(f"\n Epoch {e}'s error = {curr_error}")

        accuracy = 0

        for x, y in tzip(x_test, y_test):
            output = x

            for layer in layers:
                output = layer.forward(output)

            if np.argmax(output) == y:
                accuracy += 1

        accuracy /= len(y_test)

        print(f"Epoch {e} - accuracy of CNN: {accuracy}\n")

        with open("./model/best_accuracy", "r+") as file:
            best = file.read().strip()

            if float(best) < accuracy:
                print(f"Better weights and biases found with accuracy: {accuracy:.4f}. Saving weights and biases.")

                with open("./model/model.pkl", "wb") as pkl_file:
                    pickle.dump(layers, pkl_file)

                file.seek(0)
                file.write(f"{accuracy:.6f}")
                file.truncate()

if __name__ == "__main__":
    main()