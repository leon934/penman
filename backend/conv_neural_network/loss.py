import numpy as np

def cross_entropy(encoded: np.array, actual: np.array):
    eps = 1e-12
    actual = np.clip(actual, eps, 1 - eps)

    return -np.sum(np.log(actual) * encoded) / encoded.shape[0]

def cross_entropy_prime(encoded: float, actual: float):
    return (actual - encoded) / encoded.shape[0]