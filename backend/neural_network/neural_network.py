import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

node_count = [784, 16, 10]

# TODO: Make the neural network into a class.
class NeuralNetwork:
    def __init__(self, training_data, node_count: list):
        self.training_data = training_data
        self.node_count = node_count
        self.weight_list, self.bias_list = self.init_nn()

        self.X_training_data, self.one_hot_Y = self.setup_data(training_data)

    # Initializes and randomizes weights and biases.
    def init_nn(self):
        self.weight_list, self.bias_list = [], []

        for i in range(1, len(self.node_count)):
            weight = np.random.rand(self.node_count[i], self.node_count[i - 1]) - 0.5
            bias = np.random.rand(self.node_count[i], 1) - 0.5

            self.weight_list.append(weight)
            self.bias_list.append(bias)

        return self.weight_list, self.bias_list

    def setup_data(self, training_data):
        X_training_data = np.array(training_data[:].T[1:]) / 255
        Y_training_data = np.array(training_data[:].T[:1])

        one_hot_Y = self.one_hot(Y_training_data)

        return X_training_data, one_hot_Y
    
    # One-hot-encodes the Y-values so that they can be compared to the output.
    def one_hot(self, Y, num_classes=10):
        one_hot_Y = np.zeros((num_classes, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    # Rectified linear unit function to add non-linearity to model.
    def ReLU(self, x):
        return np.maximum(0, x)

    def derivativeReLU(self, x):
        return x > 0
    
    # Transforms output vectors into probabilities.
    def softmax(self, output_vector: np.array) -> list:
        return np.exp(output_vector) / np.sum(np.exp(output_vector), axis=0, keepdims=True)
    
    def forward_propagation(self, input_matrix):
        '''
        Forward propagates through the neural network, resulting in a 10 x n matrix, where n is the amount of samples there are.

        Parameters:
        input_matrix: The input data matrix.
        weight_list: List of weight matrices for each layer.
        bias_list: List of bias vectors for each layer.

        Returns:
        preactivation_list: List of preactivation values for each layer.
        activation_list: List of activation values for each layer.
        '''

        # The "None" is a placeholder value to keep indices consistent later.
        preactivation_list, activation_list = [None], [input_matrix]
        
        for i in range(len(self.weight_list)):
            Z = self.weight_list[i] @ input_matrix + self.bias_list[i]

            preactivation_list.append(Z)
            activation_list.append(self.ReLU(Z))

            input_matrix = self.ReLU(Z)

        # Apply softmax to the output layer
        activation_list[-1] = self.softmax(preactivation_list[-1])

        return preactivation_list, activation_list

    def backward_propagation(self, batch_size, activation_list, preactivation_list):
        # Creates the output activation/preactivation (z/a).
        output_activation = activation_list[-1]

        dC_da = output_activation - self.one_hot_Y
        output_error = dC_da

        update_weight_list, update_bias_list = [output_error @ activation_list[-2].T / batch_size], [np.sum(output_error, axis=1, keepdims=True) / batch_size]

        for i in range(len(self.node_count) - 2, 0, -1):
            output_error = ((self.weight_list[i].T @ output_error) * self.derivativeReLU(preactivation_list[i]))

            dC_dW = output_error @ activation_list[i - 1].T / batch_size
            dC_dB = np.sum(output_error, axis=1, keepdims=True) / batch_size

            update_weight_list.append(dC_dW)
            update_bias_list.append(dC_dB)

        update_weight_list.reverse()
        update_bias_list.reverse()

        return update_weight_list, update_bias_list

    def update_params(self, update_weight_list, update_bias_list, training_rate):
        for i in range(len(self.weight_list)):
            self.weight_list[i] -= training_rate * update_weight_list[i]
            self.bias_list[i] -= training_rate * update_bias_list[i]

    def row_to_image(self, row, size=(28, 28), show=True):
        # Convert the row to a numpy array and reshape it
        pixels = np.array(row).reshape(size)
        
        # Normalize the pixel values to the range [0, 255]
        pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255).astype(np.uint8)
        
        # Create an image from the pixel array
        img = Image.fromarray(pixels, mode='L')
        
        if show:
            # Display the image using matplotlib
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.show()
        
        return img

    def test_sample(self, n, testing_data):
        # Extract the nth sample from the test dataset
        test_sample = np.array(testing_data.iloc[n, :]) / 255  # Normalize the sample
        
        # Reshape the sample to match the input shape
        test_sample = test_sample.reshape(784, 1)
        
        # Forward propagation
        activations = test_sample
        for i in range(len(self.weight_list)):
            Z, activations = self.forward_propagation(activations)
        
        # Apply softmax to the output layer
        output = self.softmax(Z)
        predicted_class = np.argmax(output, axis=0)
        print(output)
        
        # Print the result
        print(f"Predicted label: {predicted_class[0]}")
        
        # Visualize the sample
        self.row_to_image(test_sample.reshape(28, 28), show=True)

    # Finds the accuracy of the current batch.
    def get_accuracy(self, activation_array):
        predicted_labels = np.argmax(activation_array, axis=0)
        true_labels = np.argmax(self.one_hot_Y, axis=0)
        return np.sum(predicted_labels == true_labels) / true_labels.size

    def train(self, epochs, training_rate):
        accuracy_list = []
        length = []

        for epoch in range(epochs):
            preactivation_list, activation_list = self.forward_propagation(self.X_training_data)
            update_weight_list, update_bias_list = self.backward_propagation(self.X_training_data.shape[1], activation_list, preactivation_list)
            self.update_params(update_weight_list, update_bias_list, training_rate)
            accuracy = self.get_accuracy(activation_list[-1])
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy * 100:.2f}%')

            accuracy_list.append(accuracy)
            length.append(epoch)

        return accuracy_list

    def saveParams(self, weight_file_prefix, bias_file_prefix):
        for i, weight in enumerate(self.weight_list):
            np.save(f"./backend/neural_network/wb/{weight_file_prefix}_{i}.npy", weight)
        for i, bias in enumerate(self.bias_list):
            np.save(f"./backend/neural_network/wb/{bias_file_prefix}_{i}.npy", bias)
            
# # Deliverable: Number/operator given image.

