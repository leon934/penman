from PIL import Image
import numpy as np

def transform_image(img: Image):
    '''
    Transforms a given image of an unsolved equation (eg. an image of 2 + 3) and resizes it down for further processing.

    Parameters:
    image_location: The location of the image to be processed.

    Returns:
    A scaled down numpy array of the image.
    '''

    # FIXME: Function should eventually be reformed to transform images that have more than three symbols.

    # Load and preprocess the image.
    # img = Image.open(image_location).convert('L')
    img = img.convert('L')

    # img_width, img_height = img.size

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # if img_width > img_height:
    #     img = img.resize((int(img_width / img_height * 28), 28))
    # else:
    #     img = img.resize((28, int(img_height / img_width * 28)))

    # Done to allow the neural network to eventually process the individual symbols by inverting the colors.
    img_array = 255 - np.array(img)     
    img_array[img_array < 10] = 0

    return img_array