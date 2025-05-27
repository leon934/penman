import numpy as np
from typing import List

class Dense():
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.normal(0, np.sqrt(2 / input_size), (output_size, input_size))

        self.bias = np.random.rand(output_size, 1)
        self.bias.fill(0.1)

    def forward(self, input: np.array):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_grad, learning_rate):
        weights_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)

        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * output_grad

        return input_grad