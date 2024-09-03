import neural_network as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def transformImage(image_location):
    '''
    Transforms a given image of an unsolved equation (eg. an image of 2 + 3) and resizes it down for further processing.

    Parameters:
    image_location: The location of the image to be processed.

    Returns:
    A scaled down numpy array of the image.
    '''

    # Load and preprocess the image.
    img = Image.open(image_location).convert('L')
    img_width, img_height = img.size
    img = img.resize((int(img_width / img_height * 28), 28)) if img_width > img_height else img.resize((28, int(img_height / img_width * 28)))

    # Done to allow the neural network to eventually process the individual symbols.
    img_array = 255 - np.array(img)
    img_array[img_array < 10] = 0

    return img_array

# FIXME: Currently only works if img_width < img_height.
def findSymbols(np_image_array) -> list :
    '''
    Scans through the numpy array to locate any symbols.

    Parameters:
    np_image_array: A numpy array of an image to process the symbols. Note that the background must be zeros for it to be processed.

    Returns:
    A list of numpy arrays of the symbols obtained from np_image_array.
    '''
    img_height, img_width = np_image_array.shape

    is_symbol = False
    symbol_list = []

    # Initialize an empty array with shape (img_height, 0) to store the columns
    symbol_array = np.empty((img_height, 0))

    # Iterate through columns to find symbols.
    for col in range(img_width):
        column_data = np_image_array[:, col][:, np.newaxis]  # Convert to 2D array (column vector)

        # Check if there are any non-zero values in the column, which are the "symbols".
        if np.any(column_data > 0):
            symbol_array = np.concatenate((symbol_array, column_data), axis=1)

            is_symbol = True
        elif is_symbol and np.any(column_data == 0):
            symbol_list.append(symbol_array)
            is_symbol = False

            symbol_array = np.empty((img_height, 0))

    return symbol_list

def resizeImage(symbol_list):
    '''
    Transforms all elements in a numpy array list into a 28x28 image to be processed by the neural network.

    Parameters:
    symbol_list: A numpy array list of the obtained symbols.

    Returns:
    A numpy array list of all symbols (28x28) that are flattened.
    '''
    resized_symbols = []

    for symbol in symbol_list:
        _, symbol_width = symbol.shape

        total_padding = 28 - symbol_width
        if total_padding > 0:
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding

            symbol = np.concatenate((np.zeros((symbol.shape[0], left_padding)), symbol), axis=1)
            symbol = np.concatenate((symbol, np.zeros((symbol.shape[0], right_padding))), axis=1)

        resized_symbols.append(symbol.flatten())

    return resized_symbols

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting the max for numerical stability
    return e_x / e_x.sum(axis=0)

def main():
    node_count = [784, 64, 32, 10]
    max_iteration = 50
    i = 0
    epochs = 110
    
    training_data = pd.read_csv(r"C:\Users\leonl\Documents\GitHub\penman\backend\mnist_dataset\train.csv")

    while i < max_iteration:
        number_model = nn.NeuralNetwork(training_data, node_count)

        accuracy_trend = number_model.train(epochs, 0.1)

        plt.plot(range(epochs), accuracy_trend, label=f'Iteration {i + 1}')

        if accuracy_trend[-1] > 0.85:
            number_model.saveParams('weight', 'bias')
            break    
        
        i = i + 1

    plt.ylim(0, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend')
    plt.legend()
    plt.show()

    np.set_printoptions(linewidth=141, suppress=True)
    

    img = transformImage(r"C:\Users\leonl\Documents\GitHub\penman\backend\uploads\Screenshot.png")
    symbol_list = findSymbols(img)
    flattened_symbol_list = resizeImage(symbol_list)

    curr_img = flattened_symbol_list[2]
    print(curr_img)

    curr_img /= 255

    for i in range(len(node_count) - 1):
        weight = np.load(f'C:/Users/leonl/Documents/GitHub/penman/backend/neural_network/wb/weight_{i}.npy')
        bias = np.load(f'C:/Users/leonl/Documents/GitHub/penman/backend/neural_network/wb/bias_{i}.npy')

        if i == 0:
            curr_img = (weight @ curr_img).reshape(node_count[1], 1) + bias
        else:
            curr_img = (weight @ curr_img) + bias

    probabilities = softmax(curr_img)

    print(probabilities)
    print(np.argmax(probabilities))


if __name__ == '__main__':
    main()