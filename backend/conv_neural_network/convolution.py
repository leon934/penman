import numpy as np
from typing import List

class Convolution():
    def __init__(self, input_size: tuple[int], kernel_size: int, kernel_count: int, non_random=False):
        self.input_depth, self.input_height, self.input_width = input_size
        self.input_size = input_size

        self.kernel_size = kernel_size
        self.kernel_count = kernel_count

        self.output_shape = (kernel_count, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernels_shape = (kernel_count, self.input_depth, self.kernel_size, self.kernel_size)

        if non_random:
            self.kernels = np.zeros(self.kernels_shape)

            for i in range(self.kernels.shape[0]):
                self.kernels[i] = np.arange(0.1, 1, 0.1).round(1).reshape(self.kernel_size, self.kernel_size)
        else:
            self.kernels = np.random.normal(0, np.sqrt(2 / (kernel_size * kernel_size)), self.kernels_shape).astype(np.float32)

        self.bias = np.zeros(self.output_shape, dtype=np.float32)

    def init_kernel(self, input_size: tuple[int], kernel_size: int, kernel_count: int) -> list:
        '''
        The initialization is done with Xavier initialization, since it works better with sigmoid activation functions.

        TODO: Other types of initializations can be created as well (eg. He initialization for ReLU functions).
        '''
        
        # Gets standard deviation from input sizes.
        n_in = input_size[2] * kernel_size * kernel_size
        n_out = kernel_count * kernel_size * kernel_size
        std_dev = np.sqrt(2 / (n_in + n_out))

        kernels = []

        # Iterates through and generates random values with Xavier initialization for each kernel.
        for _ in range(kernel_count):
            kernel = np.random.normal(0, std_dev, kernel_size ** 2).reshape((kernel_size, kernel_size))
            kernels.append(kernel)
        
        return kernels

    def conv(self, matrix: np.array, kernel: np.array, full_conv: bool = False, do_conv: bool = False) -> np.array:
        '''
        Does the convolution between the matrix and the kernel. A full convolution can be specified with the optional arg.

        @params:
            1. matrix (int): to be convoluted
            2. kernel (np.array): used to convolute the matrix
            3. full_conv (bool): determines if a full convolution should be done; false by default
        
        @returns:
            1. feature_map (np.array): Feature map after applying kernel to matrix.

        Note that kernel's size should be less than or equal to the matrix' size.
        '''

        if do_conv:
            kernel = np.flipud(np.fliplr(kernel))

        if matrix.size < kernel.size:
            raise Exception("Matrix size less than kernel's size.")
        
        if full_conv:
            # Surrounds matrix with zeros to perform full convolution.
            pad = kernel.shape[0] - 1
            full_matrix = np.pad(matrix, ((pad, pad), (pad, pad)), constant_values=0)
            
            # Obtains the dimensions of the new feature map.
            dim = matrix.shape[0] + kernel.shape[0] - 1
            matrix = full_matrix
        else:
            # Obtains the dimensions of the new feature map.
            dim = matrix.shape[0] - kernel.shape[0] + 1
        
        feature_map = np.empty((dim, dim))

        # Goes through each sub-array of the original matrix and applies dot product of it to the kernel.
        for i in range(dim):
            for j in range(dim):
                patch = matrix[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                feature_map[i, j] = np.sum(patch * kernel)

        return feature_map
    
    def forward(self, matrices: np.array) -> np.array:
        '''
        Calculates the output layer of the convolution.

        @params:
            1. matrices (np.array): either an np.array or a list of them.
        
        @returns:
            1. feature_maps: List of features maps.
        '''

        self.input = matrices

        # Initialize output matrix.
        output = np.zeros(self.output_shape)

        # Applies cross-correlation with each kernel to each input matrix.
        for ki, kernel in enumerate(self.kernels):
            total = np.zeros((self.output_shape[1], self.output_shape[2]))
            
            for d in range(matrices.shape[0]):
                total += self.conv(matrices[d], kernel[d])
            
            output[ki] = total + self.bias[ki]

        self.output = output

        return self.output
 
    def backward(self, output_gradient: np.array, learning_rate: float) -> np.array:
        '''
        Calculates the gradient of the input with respect to the loss function.

        @params:
            1. output_gradient (np.array): change in error with respect to the output.
            2. learning_rate (float): factor to change the kernels and biases by.

        @returns:
            1. input_gradient: Returns the change in error with respect to output for the previous layer.
        '''

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_size)

        for i in range(self.kernel_count):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = self.conv(self.input[j], output_gradient[i])
                input_gradient[j] += self.conv(output_gradient[i], self.kernels[i, j], full_conv=True, do_conv=True)

        self.kernels -= learning_rate * kernels_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient