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
    # Uses a kernel and morphology to group closely positioned objects.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

    return boxes

def harsh_gamma_enhance(data, gamma=0.3, target_max=255):
    nonzero_mask = data > 1e-6
    result = data.copy().astype(float)
    
    if np.any(nonzero_mask):
        nonzero_vals = data[nonzero_mask]

        normalized = (nonzero_vals - nonzero_vals.min()) / (nonzero_vals.max() - nonzero_vals.min())
        gamma_corrected = np.power(normalized, gamma)
        
        result[nonzero_mask] = gamma_corrected * target_max
    
    result[data <= 1e-6] = 0
    return result.astype(np.uint8)

def pad_image(image: np.ndarray, rect_boxes: list) -> np.ndarray:
    images = np.ndarray((len(rect_boxes), 1, 28, 28))

    for i, rect in enumerate(rect_boxes):
        x, y, w, h = rect

        curr_image = image[y:y + h, x:x + w]

        # Finds the amount of padding required to make the image a square.
        pad_w = 28 - w
        pad_h = 28 - h

        left = pad_w // 2
        right = pad_w - left

        top = pad_h // 2
        bottom = pad_h - top

        # Creating a border makes the image nxn, to be resized properly.
        padded_image = cv2.copyMakeBorder(curr_image, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        padded_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_CUBIC)

        images[i] = padded_image

    return images

def stretch_image(image: np.ndarray, rect_boxes: list) -> np.ndarray:
    images = np.ndarray((len(rect_boxes), 1, 28, 28))

    for i, rect in enumerate(rect_boxes):
        x, y, w, h = rect

        curr_image = image[y:y + h, x:x + w]

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

        padded_image = cv2.copyMakeBorder(curr_image, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        padded_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_CUBIC)

        images[i] = padded_image

    return images

def normalize(image: np.ndarray) -> np.ndarray:
    return image / 255

def transform_image(image: Image):
    # Resizes and inverts image so the larger dimension is 28 and is the proper format for the CNN.
    resized_image = resize_image(image)
    contour_boxes = find_contours(resized_image)

    image_arr = stretch_image(resized_image, contour_boxes)

    pipeline =              [stretch_image, cv2.erode, harsh_gamma_enhance, normalize]
    pipeline_default_args = [
        {"rect_boxes": contour_boxes},
        {"kernel": None, "iterations": 1},
        {},
        {}
    ]

    for i, image in enumerate(resized_image):
        for j in range(len(pipeline)):
            image = pipeline[j](image, **pipeline_default_args[j])

        image_arr[i] = image

    return image_arr

def process_image(image: Image):
    pipeline =              [cv2.erode, harsh_gamma_enhance, normalize]
    pipeline_default_args = [
        {"kernel": None, "iterations": 1},
        {},
        {}
    ]

    # Resizes and inverts image so the larger dimension is 28 and is the proper format for the CNN.
    resized_image = resize_image(image)
    contour_boxes = find_contours(resized_image)

    image_arr = stretch_image(resized_image, contour_boxes)

    for i, image in enumerate(image_arr):
        for j in range(len(pipeline)):
            image = pipeline[j](image, **pipeline_default_args[j])

        image_arr[i] = image

    return image_arr