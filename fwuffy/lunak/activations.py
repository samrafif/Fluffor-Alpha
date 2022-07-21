from .base import Function
from .functional import *


class Sigmoid(Function):
    def forwards(self, x):
        return sigmoid(x)

    def backwards(self, dy):
        return dy * self.grads["X"]

    def local_grads(self, x):
        dx = sigmoid_prime(x)
        grads = {"X": dx}
        self.grads = grads
        return grads


class ReLU(Function):
    def forwards(self, x):
        return relu(x)

    def backwards(self, dy):
        return dy * self.grads["X"]

    def local_grads(self, x):
        dx = relu_prime(x)
        grads = {"X": dx}
        self.grads = grads
        return grads


class Tanh(Function):
    def forwards(self, x):
        return tanh(x)

    def backwards(self, dy):
        return dy * self.grads["X"]

    def local_grads(self, x):
        dx = tanh_prime(x)
        grads = {"X": dx}
        self.grads = grads
        return grads


class Softmax(Function):
    def forwards(self, X):
        exp_x = np.exp(X)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache["X"] = X
        self.cache["output"] = probs
        return probs

    def backwards(self, dY):
        dX = []

        for dY_row, grad_row in zip(dY, self.grads["X"]):
            dX.append(np.dot(dY_row, grad_row))

        return np.array(dX)

    def local_grads(self, X):
        grad = []

        for prob in self.cache["output"]:
            prob = prob.reshape(-1, 1)
            grad_row = -np.dot(prob, prob.T)
            grad_row_diagonal = prob * (1 - prob)
            np.fill_diagonal(grad_row, grad_row_diagonal)
            grad.append(grad_row)

        grad = np.array(grad)
        grads = {"X": grad}
        self.grads = grads
        return grads


activations_dict = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "softmax": Softmax,
    "tanh": Tanh,
}
