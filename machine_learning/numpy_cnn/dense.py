import numpy as np
from typing import List

class Dense():
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.normal(0, np.sqrt(2 / input_size), (output_size, input_size)).astype(np.float32)
        
        self.bias = np.full((output_size, 1), 0.1, dtype=np.float32)

        # Adam optimizer parameters; they start at 0 since there shouldn't be momentum/velocity.
        self.m_t_weight = 0
        self.v_t_weight = 0
        self.m_t_bias = 0
        self.v_t_bias = 0

        self.t = 0

        self.B_1 = 0.9
        self.B_2 = 0.999

    def forward(self, input: np.array):
        self.input = input

        print(np.dot(self.weights, self.input).shape)

        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_grad, learning_rate):
        weights_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)

        # Calculates new Adam optimizer parameters to account for momentum.
        eps = 1e-9
        self.t += 1

        self.m_t_weight = self.B_1 * self.m_t_weight + (1 - self.B_1) * weights_grad
        self.v_t_weight = self.B_2 * self.v_t_weight + (1 - self.B_2) * weights_grad ** 2

        self.m_t_bias = self.B_1 * self.m_t_bias + (1 - self.B_1) * output_grad
        self.v_t_bias = self.B_2 * self.v_t_bias + (1 - self.B_2) * output_grad ** 2

        # Correction to fix initial bias towards 0.
        self.m_t_weight = self.m_t_weight / (1 - self.B_1 ** self.t)
        self.v_t_weight = self.v_t_weight / (1 - self.B_2 ** self.t)

        self.m_t_bias   = self.m_t_bias / (1 - self.B_1 ** self.t)
        self.v_t_bias   = self.v_t_bias / (1 - self.B_2 ** self.t)

        # Previous parameter update without Adam.
        # self.weights -= learning_rate * weights_grad
        # self.bias -= learning_rate * output_grad

        self.weights -= learning_rate * self.m_t_weight / (np.sqrt(self.v_t_weight) + eps)
        self.bias -= learning_rate * self.m_t_bias / (np.sqrt(self.v_t_bias) + eps)

        return input_grad
    
    def save(self):
        pass    