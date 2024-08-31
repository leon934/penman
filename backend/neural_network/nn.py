import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

node_count = [784, 16, 10]

# TODO: Make the neural network into a class.
# class NeuralNetwork:
#     def __init__(self, batch_size, training_rate, training_data, node_count):
#         self.batch_size = batch_size
#         self.training_rate = training_rate
#         self.training_data = training_data
#         self.node_count = node_count
#         self.weight_list, self.bias_list, self.activation_list = [], [], []

# Initializes and randomizes weights and biases.
def init_nn():
    weight_list, bias_list = [], []
    
    for i in range(1, len(node_count)):
        weight = np.random.rand(node_count[i], node_count[i - 1]) - 0.5
        bias = np.random.rand(node_count[i], 1) - 0.5

        weight_list.append(weight)
        bias_list.append(bias)

    return weight_list, bias_list

# Rectified linear unit function to add non-linearity to model.
def ReLU(x):
    return np.maximum(0, x)

def derivativeReLU(x):
    return x > 0

# Transforms output vectors into probabilities.
def softmax(output_vector: np.array) -> list:
    return np.exp(output_vector) / np.sum(np.exp(output_vector), axis=0, keepdims=True)

# Goes throught the neural network by multiplying by weights and adding bias.
def forward_propagation(input_matrix, weight_list, bias_list, X_training_data): # The "None" is a placeholder value to keep indices consistent later.
    preactivation_list, activation_list = [None], [X_training_data]
    
    for i in range(len(node_count) - 1):
        Z = weight_list[i] @ input_matrix + bias_list[i]

        preactivation_list.append(Z)
        activation_list.append(ReLU(Z))

        input_matrix = ReLU(Z)

    activation_list[-1] = softmax(preactivation_list[-1])

    return preactivation_list, activation_list

# One-hot-encodes the Y-values so that they can be compared to the output.
def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# Used to find the error of the output vector from the neural network.
def mean_squared_error(y_data, output_vector):
    return 2 / len(output_vector) * (output_vector - y_data) ^ 2

# TODO: Convert backpropagation to a function.
def backward_propagation(batch_size, one_hot_Y, activation_list, preactivation_list, weight_list):
    output_activation = activation_list[-1]
    output_preactivation = preactivation_list[-1]

    dC_da = 1 / batch_size * (output_activation - one_hot_Y)
    output_error = dC_da * derivativeReLU(output_preactivation)

    update_weight_list, update_bias_list = [output_error @ activation_list[-2].T], [np.sum(output_error, axis=1, keepdims=True)]

    for i in range(len(node_count) - 2, 0, -1):
        output_error = ((weight_list[i].T @ output_error) * derivativeReLU(preactivation_list[i]))

        dC_dW = output_error @ activation_list[i - 1].T
        dC_dB = np.sum(output_error, axis=1, keepdims=True)

        update_weight_list.append(dC_dW)
        update_bias_list.append(dC_dB)

    update_weight_list.reverse()
    update_bias_list.reverse()

    return update_weight_list, update_bias_list

def update_params(weight_list, update_weight_list, bias_list, update_bias_list, training_rate):
    for i in range(len(weight_list)):
        weight_list[i] -= training_rate * update_weight_list[i]
        bias_list[i] -= training_rate * update_bias_list[i]

def row_to_image(row, size=(28, 28), show=True):
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

def test_sample(n, testing_data, weight_list, bias_list):
        # Extract the nth sample from the test dataset
        test_sample = np.array(testing_data.iloc[n, :]) / 255  # Normalize the sample
        
        # Reshape the sample to match the input shape
        test_sample = test_sample.reshape(784, 1)
        
        # Forward propagation
        activations = test_sample
        for i in range(len(weight_list)):
            Z, activations = forward_propagation(weight_list[i], bias_list[i], activations)
        
        # Apply softmax to the output layer
        output = softmax(Z)
        predicted_class = np.argmax(output, axis=0)
        print(output)
        
        # Print the result
        print(f"Predicted label: {predicted_class[0]}")
        
        # Visualize the sample
        row_to_image(test_sample.reshape(28, 28), show=True)

# Finds the accuracy of the current batch.
def get_accuracy(Y_training_data, activation_array):
    activation_array = np.argmax(activation_array, 0)

    return np.sum(activation_array == Y_training_data) / Y_training_data.size

def main():
    # testing_data = pd.read_csv(r"C:\Users\leonl\Documents\GitHub\penman\mnist_dataset\test.csv")
    training_data = pd.read_csv(r"C:\Users\leonl\Documents\GitHub\penman\mnist_dataset\train.csv")
    batch_size = 42000
    training_rate = 0.25
    iterations = 500

    for i in range(5):
        iteration_list = []
        accuracy_list = []

        # Creates random weights and biases for each of the layers excluding the input.
        weight_list, bias_list = init_nn()

        for j in range(iterations):
            # X_training_data = np.array(training_data[batch_size * j:batch_size * (j + 1)].T[1:]) / 255
            # Y_training_data = np.array(training_data[batch_size * j:batch_size * (j + 1)].T[:1])

            X_training_data = np.array(training_data[:].T[1:]) / 255
            Y_training_data = np.array(training_data[:].T[:1])

            one_hot_Y = one_hot(Y_training_data)

            # Does the forward propagation and stores the preactivation and activation matrices.
            preactivation_list, activation_list = forward_propagation(X_training_data, weight_list, bias_list, X_training_data)

            # Performs backpropagation and stores the values in a list.
            update_weight_list, update_bias_list = backward_propagation(batch_size, one_hot_Y, activation_list, preactivation_list, weight_list)

            update_params(weight_list, update_weight_list, bias_list, update_bias_list, training_rate)

            if j % 10 == 0:
                print(f"Iteration: {j}")
                print(f"Accuracy: {get_accuracy(Y_training_data, activation_list[-1]) * 100:.2f}%")

                iteration_list.append(j / 10)
                accuracy_list.append(get_accuracy(Y_training_data, activation_list[-1]) * 100)
                
                print()

        plt.plot(iteration_list, accuracy_list, label=f"Line {i}")
    plt.ylim(0, 100)
    plt.show()

    # Test the function with an example index
    # test_sample(0, testing_data, weight_list, bias_list)

    # TODO for the future:
    #   Use stochastic gradient descent by cutting up the training and testing data.

if __name__ == "__main__":
    main()

# Deliverable: Number/operator given image.