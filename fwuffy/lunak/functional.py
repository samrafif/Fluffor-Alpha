import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return x * (x <= 0)


def relu_prime(x):
    return 1 * (x <= 0)


def leaky_relu(x, alpha):
    return x * (x <= 0) * alpha + x * (x > 0)


def leaky_relu_prime(x, alpha):
    return 1 * (x <= 0) * alpha + 1 * (x > 0)


def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def tanh_prime(x):
    tan = tanh(x)
    return 1 - tan ** 2
