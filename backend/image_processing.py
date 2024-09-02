from PIL import Image
import numpy as np

def resizeImage():
    # Load and preprocess the image.
    img = Image.open(r'.\uploads\Screenshot.png').convert('L').resize((28, 28))
    img_array = 255 - np.array(img)
    img_array[img_array < 10] = 0

    # Deliverable: A 1D array with all pixel values.
    flattened_img = img_array.flatten()

    return img_array

resizeImage()