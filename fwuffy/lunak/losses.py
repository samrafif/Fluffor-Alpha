import numpy as np

from .base import Function


class Loss(Function):
    def forwards(self, x, y):
        """
        Forward pass. computes the loss between x and y.

        Params:
            x: numpy.ndarray of shape (batches, n_out), predictions
            y: numpy.ndarray of shape (batches, n_out), ground truth

        """
        pass

    def backwards(self):
        """
        Bacward pass. returns the local gradients.

        Params:
            x: numpy.ndarray of shape (batches, n_out), predictions
            y: numpy.ndarray of shape (batches, n_out), ground truth

        """
        return self.grads["X"]

    def local_grads(self, x, y):
        """
        Forward pass, Local gradients, calculates the local gradients in respect to x

        Params:
            x: numpy.ndarray of shape (batches, n_out), predictions
            y: numpy.ndarray of shape (batches, n_out), ground truth
        """
        pass


class MeanSquaredError(Loss):
    def forwards(self, x, y):
        """
        Forward pass. computes the mean squared error between x and y.

        Params:
            x: numpy.ndarray of shape (batches, n_out), predictions
            y: numpy.ndarray of shape (batches, n_out), ground truth

        Returns:
            mean_loss: numpy.float, the MSE loss between x and y

        """
        sum_loss = np.sum((x - y) ** 2, axis=1, keepdims=True)
        mean_loss = np.mean(sum_loss)
        return mean_loss

    def local_grads(self, x, y):
        """
        Forward pass, Local gradients, calculates the local gradients in respect to x

        Params:
            x: numpy.ndarray of shape (batches, n_out), predictions
            y: numpy.ndarray of shape (batches, n_out), ground truth

        Returns:
            gradsx: numpy.ndarray of shape (batches, n_out), gradients of MSE w.r.t x
        """
        grads = {"X": 2 * (x - y) / x.shape[0]}
        return grads


class CrossEntropyLoss(Loss):
    def forwards(self, X, y):
        """
        Computes the cross entropy loss of x with respect to y.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, 1). Should contain class labels
                for each data point in x.
        Returns:
            crossentropy_loss: numpy.float. Cross entropy loss of x with respect to y.
        """
        # calculating crossentropy
        exp_x = np.exp(X)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        log_probs = -np.log([probs[i, y[i]] for i in range(len(probs))])
        crossentropy_loss = np.mean(log_probs)

        # caching for backprop
        self.cache["probs"] = probs
        self.cache["y"] = y

        return crossentropy_loss

    def local_grads(self, X, Y):
        probs = self.cache["probs"]
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {"X": (probs - ones) / float(len(X))}
        return grads
