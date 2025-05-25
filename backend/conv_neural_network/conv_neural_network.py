import numpy as np
from convolution import Convolution
from activation import Sigmoid, ReLU, Softmax
from reshape import Reshape
from dense import Dense
from loss import cross_entropy, cross_entropy_prime
import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib import tzip

final_output_size = 10

layers = [
    Convolution((1, 28, 28), kernel_size=3, kernel_count=5),
    ReLU(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 16),
    ReLU(),
    Dense(16, 10),
    Softmax(),
]

def preprocess_data(batch_size: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_train = x_train.astype("float32")

    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.reshape(len(y_train), 10, 1)

    batch_x = [x_train[i : i + batch_size] for i in range(0, x_train.shape[0], 100)]
    batch_y = [y_train[i : i + batch_size] for i in range(0, y_train.shape[0], 100)]

    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    x_test = x_test.astype("float32")

    return batch_x, batch_y, x_test, y_test


def main():
    epochs = 1
    learning_rate = 0.1
    batch_size = 100

    start = time.perf_counter()

    batch_x, batch_y, x_test, y_test = preprocess_data(batch_size)

    end = time.perf_counter()

    print(f"Time to create data: {end - start:.4f}")

    errors = []

    for _ in tqdm(range(epochs)):
        loss = 0

        for curr_batch_x, curr_batch_y in tzip(batch_x, batch_y):
            batch_output = []

            for x in curr_batch_x:
                output = x

                for layer in layers:
                    output = layer.forward(output)

                batch_output.append(output)

            batch_output = np.stack(batch_output)

            loss += np.sum(
                [cross_entropy(y, y_hat) for y, y_hat in zip(curr_batch_y, batch_output)]
            )

            batch_grad = (
                np.stack(
                    [cross_entropy_prime(y, y_hat) for y, y_hat in zip(curr_batch_y, batch_output)]
                ) / batch_size
            )
            grad = batch_grad.mean(axis=0)

            for layer in reversed(layers):
                grad = layer.backward(grad, learning_rate)

        curr_error = loss / len(batch_x) * batch_size
        
        errors.append(curr_error)
        print(f"Error = {curr_error}")

    accuracy = 0

    for x, y in tzip(x_test, y_test):
        output = x

        for layer in layers:
            output = layer.forward(output)

        if np.argmax(output) == y:
            accuracy += 1

    print(f"Final accuracy of CNN: {accuracy / len(y_test)}")


if __name__ == "__main__":
    main()
