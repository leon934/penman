import numpy as np

class Reshape():
    def __init__(self, input_shape: tuple[int], output_shape: tuple[int]):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_shape):
        return np.reshape(input_shape, self.output_shape)
    
    def backward(self, output_grad, alpha):
        return np.reshape(output_grad, self.input_shape)