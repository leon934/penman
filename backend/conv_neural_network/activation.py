import numpy as np

class Activation():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Sigmoid(Activation):
    '''
    Activation function class that fits all values in between 1 and -1.

    @params:
        1. x (float): value to pass into the sigmoid function
    @returns:
        1. x (float): returns value after being passed into the sigmoid function
    '''
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))

        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    '''
    Activation function class that just returns the value if it is greater than 0.

    @params:
        1. x (float): value to pass into relu function.
    @returns:
        2. x (float): value after being passed into relu function.
    '''
    def __init__(self):
        def relu(x: np.array) -> np.array:
            return np.maximum(0, x)
        
        def relu_prime(x):
            return (x > 0).astype(x.dtype)

        super().__init__(relu, relu_prime)

class Softmax():
    '''
    Activation function class that raises each scalar value to the power of e and returns the probability that each vector occurs.

    @params:
        1. output_vector (np.array): a vector of the last layer's output containing the probabilities that the image is that number.
    @returns:
        1. output_vector (np.array): softmax representation of the original output_vector argument.
    '''

    def forward(self, output_vector: np.array) -> np.array:
        self.output = np.exp(output_vector) / np.sum(np.exp(output_vector), axis=0, keepdims=True)
        
        return self.output

    def backward(self, grad, learning_rate):
        inner = np.sum(grad * self.output, axis=0, keepdims=True)
        
        return self.output * (grad - inner)