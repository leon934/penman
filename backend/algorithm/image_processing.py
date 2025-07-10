from PIL import Image
import numpy as np
import cv2

def transform_single_digit(img: Image):
    '''
    Transforms a given image of an unsolved equation (eg. an image of 2 + 3) and resizes it down for further processing.

    Parameters:
    image_location: The location of the image to be processed.

    Returns:
    A scaled down numpy array of the image.
    '''

    # Load and preprocess the image.
    img = img.convert('L')

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Done to allow the neural network to eventually process the individual symbols by inverting the colors.
    img_array = 255 - np.array(img)     
    img_array[img_array < 10] = 0

    return img_array

def resize_image(img: Image):
    '''
    Transforms a given image of an unsolved equation (eg. an image of 2 + 3) and resizes it down for further processing.

    Parameters:
    image_location: The location of the image to be processed.

    Returns:
    A scaled down numpy array of the image.
    '''

    # FIXME: Function should eventually be reformed to transform images that have more than three symbols.

    # Load and preprocess the image.
    img = img.convert('L')

    img_width, img_height = img.size

    if img_width > img_height:
        img = img.resize((int(img_width / img_height * 28), 28))
    else:
        img = img.resize((28, int(img_height / img_width * 28)))

    # Inverts colors and filters out any subtle dark colors.
    img_array = 255 - np.array(img)     
    img_array[img_array < 10] = 0

    return img_array

def find_contours(image: Image):
    # Finds the contours by obtaining only the outermost shapes.
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

# FIXME: Currently only works if img_width < img_height.
def find_symbols(np_image_array) -> list :
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