from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch

from typing import Literal, List

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
    img_array = img_array.astype("float32") / 255

    return img_array

def convert_to_numpy(img: Image) -> np.ndarray:
    # Load and preprocess the image.
    img = img.convert('L')

    # Inverts colors and filters out any subtle dark colors.
    img_array = 255 - np.array(img)     
    img_array[img_array < 10] = 0

    return img_array

def resize_image(img: Image) -> Image:
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
        img = img.resize((28, int(img_height / img_width * 28)))
    else:
        img = img.resize((int(img_width / img_height * 28), 28))

    # Inverts colors and filters out any subtle dark colors.

    return img

def erode(image: Image) -> Image:
    numpy_image = np.array(image)

    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(numpy_image, kernel=kernel, iterations=1)

    return Image.fromarray(eroded_image)

def find_symbols(image: Image) -> List[Image.Image]:
    '''
    finds the symbols by making images "thicker" then uses cv2's findContours to return a list of found symbols
    '''

    image_np = convert_to_numpy(image)

    # Uses a kernel and morphology to group closely positioned objects.
    kernel_size = (100, 60)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    morphed = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorts them from left to right since findContours does not sort them in that order by default.
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    symbols = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        symbol = image_np[y:y + h, x:x + w]
        symbols.append(Image.fromarray(symbol))

    return symbols

def harsh_gamma_enhance(image: Image, gamma=0.3, target_max=255):
    image = np.array(image)

    nonzero_mask = image > 1e-6
    result = image.copy().astype(float)
    
    if np.any(nonzero_mask):
        nonzero_vals = image[nonzero_mask]

        normalized = (nonzero_vals - nonzero_vals.min()) / (nonzero_vals.max() - nonzero_vals.min())
        gamma_corrected = np.power(normalized, gamma)
        
        result[nonzero_mask] = gamma_corrected * target_max
    
    result[image <= 1e-6] = 0

    return Image.fromarray(result)

def resize_symbols(image: Image, mode: Literal["stretch", "pad"]) -> Image:
    np_image = np.array(image)

    h, w = np_image.shape

    if mode == "pad":
        # Finds the amount of padding required to make the image a square.
        pad_w = max(w, h) - w
        pad_h = max(w, h) - h

        left = pad_w // 2
        right = pad_w - left

        top = pad_h // 2
        bottom = pad_h - top
    elif mode == "stretch":
        # Finds the amount of padding required to make the image a square.
        if w > h:
            pad_h = w - h

            top = pad_h // 2
            bottom = pad_h - top

            left = right = 0
        elif h > w:
            pad_w = h - w

            left = pad_w // 2
            right = pad_w - left
            
            top = bottom = 0
        else:
            top = bottom = left = right = 0
    else:
        raise Exception("Mode must be either 'pad' or 'stretch'.")

    # Creating a border makes the image nxn, to be resized properly.
    padded_image = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    padded_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(padded_image)

def normalize(image: Image) -> Image:
    numpy_image = np.array(image)

    normalized_image = numpy_image / 255

    return Image.fromarray(normalized_image)

def process_image(
        img: Image,
        pipeline: transforms.Compose
    ) -> torch.Tensor:
    '''
    img is the original image
    '''

    symbols = find_symbols(img)
        
    tensors = []
    for symbol in symbols:
        tensor = pipeline(symbol)
        tensors.append(tensor)

    return torch.stack(tensors)