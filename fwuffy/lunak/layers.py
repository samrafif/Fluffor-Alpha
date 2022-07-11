import numpy as np
import math
from typing import Optional

from .activations import activations_dict, leaky_relu, leaky_relu_prime
from .base import Function


class Layer(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_dims: int
        self.out_dims: int
        self.name = self.__class__.__name__.lower()
        self.params = {}
        self.param_updates = {}

    def _init_params(*args, **kwargs):
        pass

    def init_layer(self, idx, **kwargs):
        self.name += str(idx)
        self.name += f" ({self.__class__.__name__})"

    def _update_params(self, lr):
        """
        Updates the trainable parameters using the corresponding global gradients
        computed during the Backpropogation

        Params:
            lr: float. learning rate.
        """
        for key, _ in self.params.items():
            self.params[key] -= lr * self.param_updates[key]

    def __repr__(self):
        return f"<{self.__class__.__name__}: in_dim: {self.in_dims}, out_dims: {self.out_dims}>"


class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def init_layer(self, idx):
        super().init_layer(idx)
        
        if isinstance(self.in_dims, tuple):
            self.out_dims = int(np.prod(self.in_dims))
            return
        
        self.out_dims = self.in_dims
    
    def forwards(self, x):
        self.cache["shape"] = x.shape
        batch = x.shape[0]
        return x.reshape(batch, -1)

    def backwards(self, dy):
        return dy.reshape(self.cache["shape"])


class LeakyReLU(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def init_layer(self, idx):
        super().init_layer(idx)
        self.out_dims = self.in_dims

    def forwards(self, x):
        return leaky_relu(x, self.alpha)

    def backwards(self, dy):
        return dy * self.grads["X"]

    def local_grads(self, x):
        dx = leaky_relu_prime(x, self.alpha)
        grads = {"X": dx}
        return grads


class PRelu(Layer):
    def __init__(self):
        super().__init__()

    def _init_params(self):

        self.params["al"] = np.zeros((self.in_dims, self.out_dims))

    def init_layer(self, idx):
        super().init_layer(idx)
        self.out_dims = self.in_dims
        self._init_params()

    def forwards(self, x):
        return np.dot(x * (x <= 0), self.params["al"]) + x * (x > 0)
    
    def backwards(self, dy):
        dy_prev = dy * self.grads["x"]
        dal = self.grads["al"].T.dot(dy)
        
        self.param_updates = {"al": dal}
        return dy_prev
    
    def local_grads(self, x):
        dx = np.dot(1 * (x <= 0), self.params["al"]) + 1 * (x > 0)
        dal = x * (x <= 0)
        grads = {"x": dx, "al": dal}
        return grads


class Linear(Layer):
    def __init__(self, out_dims, in_dims=None, activation=None):
        super().__init__()

        if activation not in activations_dict.keys():
            if activation is not None:
                raise ValueError(f"Activation function {activation} not supported")

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.activation = activation
        self.activation_f = activations_dict[activation]() if activation else None

    def __repr__(self):
        reprs = super().__repr__()[:-1]
        reprs += f", activation: {self.activation}>"
        return reprs

    def _init_params(self, in_dims, out_dims, activation):

        if activation in ("sigmoid", "tanh", "softmax"):
            scale = 1 / math.sqrt(in_dims)

        if activation == "relu" or activation is None:
            scale = math.sqrt(2.0 / in_dims)

        self.params["W"] = scale * np.random.randn(in_dims, out_dims)
        self.params["b"] = np.zeros((1, out_dims))

    def init_layer(self, idx):
        super().init_layer(idx)
        self._init_params(self.in_dims, self.out_dims, self.activation)

    def forwards(self, x):
        """
        Forward pass for the Linear layer.

        Args:
            X: numpy.ndarray of shape (batch, in_dim) containing
                the input value.
        Returns:
            Y: numpy.ndarray of shape of shape (batch, out_dim) containing
                the output value.
        """
        if x.shape[-1] != self.in_dims:
            raise ValueError(
                f"Expected {self.name} input dims to be {self.in_dims}, got {x.shape} instead"
            )

        z = np.dot(x, self.params["W"]) + self.params["b"]

        a = self.activation_f(z) if self.activation_f else z

        return a

    def backwards(self, dy):
        """
        Backward pass for the Linear layer.
        Args:
            dY: numpy.ndarray of shape (batch, n_out). Global gradient
                backpropagated from the next layer.
        Returns:
            dX: numpy.ndarray of shape (batch, n_out). Global gradient
                of the Linear layer.
        """
        dz = self.activation_f.backwards(dy) if self.activation_f else dy
        dy_prev = dz.dot(self.grads["x"].T)

        dw = self.grads["w"].T.dot(dz)
        db = np.sum(dz, axis=0, keepdims=True)

        self.param_updates = {"W": dw, "b": db}
        return dy_prev

    def local_grads(self, x):
        """
        Local gradients of the Linear layer at X.
        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing the
                input data.
        Returns:
            grads: dictionary of local gradients with the following items:
                X: numpy.ndarray of shape (batch, in_dim).
                W: numpy.ndarray of shape (batch, in_dim).
                b: numpy.ndarray of shape (batch, 1).
        """
        gradx_l = self.params["W"]
        gradw_l = x
        gradb_l = np.ones_like(self.params["b"])

        grads = {"x": gradx_l, "w": gradw_l, "b": gradb_l}
        return grads


class Conv2D(Layer):
    def __init__(self, out_channels, in_channels, kernel_size=3, stride=1, padding=0, activation=None):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding


# TODO: Write RNN class wrapper, to automate reccurent loop
class RNN(Layer):
    def __init__(
        self, cell, return_sequences=False, return_state=False, reversed=False
    ):
        super().__init__()


class RNNCell(Layer):
    def __init__(self, units, in_dims=None, activation="tanh"):
        super().__init__()

        if activation not in activations_dict.keys():
            if activation is not None:
                raise ValueError(f"Activation function {activation} not supported")

        self.in_dims = in_dims
        self.out_dims = units
        self.state_dims = units
        self.activation = activation
        self.activation_f = activations_dict[activation]()

    def _init_params(self, in_dims, state_dims, out_dims, activation):

        if activation in ("sigmoid", "tanh", "softmax"):
            scale = 1 / math.sqrt(in_dims)

        if activation == "relu" or activation is None:
            scale = math.sqrt(2.0 / in_dims)

        # Input params
        self.params["Wx"] = scale * np.random.randn(state_dims, in_dims)

        # Hidden params
        self.params["Ws"] = scale * np.random.randn(state_dims, state_dims)
        self.params["bs"] = np.zeros((state_dims, 1))

        # Output params
        self.params["Wy"] = scale * np.random.randn(out_dims, state_dims)
        self.params["by"] = np.zeros((out_dims, 1))

    def init_layer(self, idx):
        super().init_layer(idx)
        self._init_params(self.in_dims, self.state_dims, self.out_dims, self.activation)

    def forwards(self, x, state, tanh):
        """
        Forward pass for a RNN cell.

        Args:
            x: numpy.ndarray of shape (in_dim) containing
                the input value.

            state: numpy.ndarray of shape (units) containing
                the previous state.

        Returns:
            y: numpy.ndarray of shape (units) containing
                the output value.

            new_state: numpy.ndarray of shape (units) containing
                the new computed state.
        """
        x = np.dot(self.params["Wx"], x)
        state = np.dot(self.params["Ws"], state) + self.params["bs"]
        tanh_in = x + state
        new_state = tanh(tanh_in)

        y = np.dot(self.params["Wy"], new_state) + self.params["by"]

        return y, new_state

    def backwards(
        self, dy, ds_prev, prev_param_updates, inputs, prev_state, curr_state, tanh
    ):

        dwy = np.dot(dy, curr_state.T)
        dby = dy
        dsa = np.dot(self.grads["dsa"].T, dy) + ds_prev

        dtanh = tanh.backwards(dsa)
        dbs = dtanh
        dws = np.dot(dtanh, prev_state.T)
        dwx = np.dot(dtanh, inputs.T)
        dx = np.dot(self.grads["dx"].T, dtanh)
        ds_prev = np.dot(self.grads["dsp"].T, dtanh)

        param_updates = [
            x + y for x, y in zip(prev_param_updates, [dwy, dws, dwx, dby, dbs])
        ]
        param_updates = [np.clip(grad, -1, 1, out=grad) for grad in param_updates]

        self.param_updates = {
            "Wy": param_updates[0],
            "Ws": param_updates[1],
            "Wx": param_updates[2],
            "by": param_updates[3],
            "bs": param_updates[4],
        }

        return dx, ds_prev, param_updates

    def local_grads(self, x, state, tanh):

        dsa = self.params["Wy"]
        ds_p = self.params["Ws"]
        dx = self.params["Wx"]

        grads = {"dx": dx, "dsa": dsa, "dsp": ds_p}
        return grads

    def _update_params(self, lr, param_updates=None):
        """
        Updates the trainable parameters using the corresponding global gradients
        computed during the Backpropogation

        Params:
            lr: float. learning rate.
            param_updates: dict | None, parameter gradients to override the internal gradients.
        """
        if param_updates is None:
            return super()._update_params(lr)

        for key, _ in self.params.items():
            self.params[key] -= lr * param_updates[key]


class LSTMCell(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
