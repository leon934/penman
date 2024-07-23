import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

# sigmoid replacement
def ReLU(x):
    return np.maximum(0, x)

def showDataSet(index):
    imagefile = './mnist_dataset/train-images.idx3-ubyte'
    imagearray = idx2numpy.convert_from_file(imagefile)

    plt.imshow(imagearray[index], cmap=plt.cm.binary)
    plt.show()