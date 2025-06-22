import numpy as np

def cross_entropy(encoded: np.ndarray, actual: np.ndarray) -> np.ndarray:
    eps = 1e-12
    actual = np.clip(actual, eps, 1 - eps)

    return -np.sum(np.log(actual) * encoded)

def cross_entropy_prime(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return (y_hat - y)