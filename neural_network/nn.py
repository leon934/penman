import numpy as np
import pandas as pd

# Initializes and randomizes weights and biases.
def init_nn(num_node, output_vector_size):
    weight = np.random.rand(output_vector_size, num_node) - 0.5
    bias = np.random.rand(output_vector_size, 1) - 0.5

    return weight, bias

# Rectified linear unit function to add non-linearity to model.
def ReLU(x):
    return np.maximum(0, x)

# Transforms output vectors into probabilities.
def softmax(output_vector: np.array) -> list:
    return np.exp(output_vector) / np.sum(np.exp(output_vector), axis=0, keepdims=True)

# Goes throught the neural network by multiplying by weights and adding bias.
def forward_propagation(weight_matrix, bias, input_matrix):
    Z = (weight_matrix @ input_matrix) + bias

    return Z, ReLU(Z)

def main():
    training_data = pd.read_csv(r"C:\Users\leonl\Documents\GitHub\penman\mnist_dataset\train.csv")

    # Used for testing. (First 5 training samples)
    X_training_data = np.array(training_data[:5].T[1:]) / 255
    Y_training_data = np.array(training_data[:5].T[:1])

    node_count = [784, 16, 16, 10]
    weight_list, bias_list, Z_list = [], [], []

    # Creates radnom weights and biases for each of the layers excluding the input.
    for i in range(1, len(node_count)):
        curr_weight, curr_bias = init_nn(node_count[i - 1], node_count[i])

        weight_list.append(curr_weight)
        bias_list.append(curr_bias)

    # Does the forward propagation and stores the z matrices.
    for i in range(len(weight_list)):
        Z_list.append(forward_propagation(weight_list[i], bias_list[i], X_training_data))
        X_training_data = Z_list[i][1]

    print(softmax(X_training_data))
    print(np.sum(softmax(X_training_data)))

if __name__ == "__main__":
    main()



# Deliverable: Number/operator given image.