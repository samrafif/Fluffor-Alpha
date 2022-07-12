from itertools import product
import numpy as np
import math
from typing import Optional

from .activations import activations_dict, leaky_relu, leaky_relu_prime
from .base import Function
from .utils import zero_pad


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
    def __init__(self, init=0.25):
        super().__init__()
        
        self.init = init

    def _init_params(self, init):

        self.params["al"] = np.tile(init, (self.in_dims, self.out_dims))

    def init_layer(self, idx):
        super().init_layer(idx)
        self.out_dims = self.in_dims
        self._init_params(self.init)

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


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride)
        )
        self.padding = padding
    
    def init_layer(self, idx):
        super().init_layer(idx)
        
        self.in_channels = self.in_dims[0]
        self.out_channels = self.in_channels
        self.out_dims = (
            self.out_channels,
            1 + ((self.in_dims[1] + 2 * self.padding)-self.kernel_size[0]) // self.stride[0],
            1 + ((self.in_dims[2] + 2 * self.padding)-self.kernel_size[1]) // self.stride[1]
            )
    
    def forwards(self, x):
        if self.padding:
            x = zero_pad(x, self.padding, (2,3))
        
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        
        out_shape = (n, self.out_channels, 1 + (h-kh) // self.stride[0], 1 + (w-kw) // self.stride[1])
        
        grads = np.zeros_like(x)
        y = np.zeros(out_shape)
        for h, w in product(range(out_shape[2]), range(out_shape[3])):
            h_offset, w_offset = h * kh, w * kw
            
            curr_field = x[:, :, h_offset: h_offset+kh, w_offset: w_offset+kw]
            y[:, :, h, w] = np.max(curr_field, axis=(2,3))
            
            for h, w in product(range(kh), range(kw)):
                
                grads[:, :, h_offset: h_offset+kh, w_offset: w_offset+kw] = (
                    x[:, :, h_offset: h_offset+kh, w_offset: w_offset+kw] >= y[:, :, h, w]
                )

        self.grads["x"] = grads[:, :, self.padding: -self.padding, self.padding: -self.padding]
        
        return y
    
    def backwards(self, dy):
        dy = np.repeat(
            np.repeat(dy, repeats=self.kernel_size[0], axis=2), repeats=self.kernel_size[1], axis=3
        )
        return self.grads["x"] * dy
    
    def local_grads(self, x):
        return self.grads

class Conv2D(Layer):
    def __init__(self, out_channels, in_channels=None, kernel_size=3, stride=1, padding=0, activation=None):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride)
        )
        self.padding = padding
        self.activation = activation
        self.activation_f = activations_dict[activation]() if activation else None
    
    def _init_params(self, in_channels, out_channels, kernel_size, activation):
        if activation in ("sigmoid", "tanh", "softmax"):
            scale = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1])

        if activation == "relu" or activation is None:
            scale = math.sqrt(2.0 / in_channels * kernel_size[0] * kernel_size[1])
        
        self.params["W"] = scale * np.random.randn(out_channels, in_channels, *kernel_size)
        
        self.params["b"] = np.zeros((out_channels, 1))
    
    def init_layer(self, idx):
        super().init_layer(idx)
        
        self.in_channels = self.in_dims[0]
        
        self.out_dims = (
            self.out_channels,
            1 + ((self.in_dims[1] + 2 * self.padding)-self.kernel_size[0]) // self.stride[0],
            1 + ((self.in_dims[2] + 2 * self.padding)-self.kernel_size[1]) // self.stride[1]
            )
        self._init_params(self.in_channels, self.out_channels, self.kernel_size, self.activation)
    
    def forward(self, x):
        """
        Forward pass for Conv2D layer
        
        params:
            x: np.ndarray of shape (batches, channels, height, width)
        
        returns:
            y: np.ndarray of shape (batches, channels_out, height_out, width_out)
        """
        
        if self.padding:
            x = zero_pad(x, self.padding, (2,3))
        
        self.cache["x"] = x
        
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        
        out_shape = (n, self.out_channels, 1 + (h-kh) // self.stride[0], 1 + (w-kw) // self.stride[1])
        y = np.zeros(out_shape)
        for b in range(n):
            for ch in range(c):
                for h, w in product(range(out_shape[2]), range(out_shape[3])):
                    h_offset, w_offset = h * self.stride[0], w * self.stride[1]
                    curr_field = x[n, :, h_offset: h_offset + kh, w_offset: w_offset + kw]
                    
                    y[b, ch, h, w] = np.sum(self.params["W"][ch] * curr_field) + self.params["b"]
        
        a = self.activation_f(y) if self.activation else y
        
        return a
    
    def backwards(self, dy):
        
        dy = self.activation_f.backwards(dy) if self.activation else dy
        x = self.cache["x"]
        
        # Calculate global gradient
        dx = np.zeros_like(x)
        
        n, c, h, w = dx.shape
        kh, kw = self.kernel_size
        for b in range(n):
            for ch in range(c):
                for h, w in product(range(dy.shape[2]), range(dy.shape[3])):
                    h_offset, w_offset = h * self.stride[0] + kh, w * self.stride[1] + kw
                    
                    dx[n, :, h_offset: h_offset + kh, w_offset: w_offset + kw] += (
                        self.params["W"][c] * dy[b, ch, h, w]
                    )
        
        # Calculate global gradients w.r.t weights
        dw = np.zeros_like(self.params["W"])
        for ch_o in range(self.out_channels):
            for ch_i in range(self.in_channels):
                for h, w in product(range(kh), range(kw)):
                    curr_field = x[:, ch_i, h: h-kh: self.stride[0], w: w-kw: self.stride[1]]
                    dy_field = dy[:, ch_o]
                    
                    dw[ch_o, ch_i, h, w] = np.sum(curr_field * dy_field)

        # Calculate global gradients w.r.t biases
        db = np.sum(dy, axis=(0, 2, 3)).reshape(-1, 1)
        
        self.param_updates = {"W": dw, "b": db}
        
        return dx[:,:, self.padding: -self.padding, self.padding: -self.padding]


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
