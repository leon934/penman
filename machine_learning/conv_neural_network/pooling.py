import numpy as np

class Pooling():
    def __init__(self, pooling_shape: int):
        self.pooling_shape = (pooling_shape, pooling_shape)
        self.stride = pooling_shape

    def forward(self, input_matrix: np.array):
        if input_matrix.shape[1] % self.stride or input_matrix.shape[2] % self.stride:
            raise "Pooling where the stride does not evenly fit into matrix is not allowed."

        self.input = input_matrix
        self.input_shape = input_matrix.shape

        new_h = input_matrix.shape[1] // self.stride
        new_w = input_matrix.shape[2] // self.stride
        
        pooled = input_matrix.reshape(input_matrix.shape[0], new_h, self.stride, new_w, self.stride)
        self.output = pooled.max(axis=(2, 4))

        expanded = self.output[:, :, None, :, None]
        mask_blocks = (pooled == expanded)
        self.mask = mask_blocks.reshape(self.input_shape)

        return self.output

    def backward(self, output_gradient, learning_rate):
        base = np.ones((1, self.stride, self.stride))
        input_gradient = np.kron(output_gradient, base)

        return input_gradient * self.mask