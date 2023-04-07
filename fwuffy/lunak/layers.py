from itertools import product
import numpy as np
import math
from typing import Optional

from .activations import Softmax, Tanh, activations_dict, leaky_relu, leaky_relu_prime
from .base import Function
from .utils import one_hot_encoding, zero_pad

scales = {
    "relu": lambda in_dims: math.sqrt(2.0 / in_dims),
    "sigmoid": lambda in_dims: 1 / math.sqrt(in_dims),
    "tanh": lambda in_dims: 1 / math.sqrt(in_dims),
    "softmax": lambda in_dims: 1 / math.sqrt(in_dims),
    None: lambda in_dims: 1,
}


class Layer(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_dims: int
        self.out_dims: int
        self.trainable = False
        self.name = self.__class__.__name__.lower()
        self.params = {}
        self.param_updates = {}

    def _init_params(*args, **kwargs):
        pass

    def init_layer(self, idx, **kwargs):
        self.name += str(idx)
        self.out_dims = (
            self.in_dims if getattr(self, "out_dims", None) is None else self.out_dims
        )

    def _update_params(self, lr):
        """
        Updates the trainable parameters using the corresponding global gradients
        computed during the Backpropogation

        Params:
            lr: float. learning rate.
        """
        if not self.trainable:
            return

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

    def forwards(self, x):
        self.cache["shape"] = x.shape
        batch = x.shape[0]
        return x.reshape(batch, -1)

    def backwards(self, dy):
        return dy.reshape(self.cache["shape"])


class Reshape(Layer):
    def __init__(self, new_shape):
        super().__init__()

        self.new_shape = new_shape

    def init_layer(self, idx):
        super().init_layer(idx)

        self.out_dims = self.new_shape[-1]

    def forwards(self, x):
        self.cache["shape"] = x.shape

        return x.reshape(self.new_shape)

    def backwards(self, dy):
        return dy.reshape(self.cache["shape"])


class LeakyReLU(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

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
        self.trainable = True

    def _init_params(self, init):
        self.params["al"] = np.tile(init, (self.in_dims, self.out_dims))

    def init_layer(self, idx):
        super().init_layer(idx)
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
        self.trainable = True
        self.activation = activation
        self.activation_f = activations_dict[activation]() if activation else None

    def __repr__(self):
        reprs = super().__repr__()[:-1]
        reprs += f", activation: {self.activation}>"
        return reprs

    def _init_params(self, in_dims, out_dims, activation):
        scale = scales[activation](in_dims)

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


class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()

        self.rate = rate

    def forwards(self, x):
        mask = np.random.binomial([np.ones((1, self.out_dims))], 1 - self.rate)[0] * (
            1.0 / (1 - self.rate)
        )
        self.mask = mask

        masked = x * mask
        return masked

    def backwards(self, dy):
        return dy * self.mask


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()

        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

    def init_layer(self, idx):
        super().init_layer(idx)

        self.in_channels = self.in_dims[0]
        self.out_channels = self.in_channels
        self.out_dims = (
            self.out_channels,
            1
            + ((self.in_dims[1] + 2 * self.padding) - self.kernel_size[0])
            // self.stride[0],
            1
            + ((self.in_dims[2] + 2 * self.padding) - self.kernel_size[1])
            // self.stride[1],
        )

    def forwards(self, x):
        if self.padding:
            x = zero_pad(x, self.padding, (2, 3))

        n, c, h, w = x.shape
        kh, kw = self.kernel_size

        out_shape = (
            n,
            self.out_channels,
            1 + (h - kh) // self.stride[0],
            1 + (w - kw) // self.stride[1],
        )

        grads = np.zeros_like(x)
        y = np.zeros(out_shape)
        for h, w in product(range(out_shape[2]), range(out_shape[3])):
            h_offset, w_offset = h * kh, w * kw

            curr_field = x[:, :, h_offset : h_offset + kh, w_offset : w_offset + kw]
            y[:, :, h, w] = np.max(curr_field, axis=(2, 3))

            for khs, kws in product(range(kh), range(kw)):
                grads[:, :, h_offset + khs, w_offset + kws] = (
                    x[:, :, h_offset + khs, w_offset + kws] >= y[:, :, h, w]
                )

        self.grads["x"] = grads

        return y

    def backwards(self, dy):
        dy = np.repeat(
            np.repeat(dy, repeats=self.kernel_size[0], axis=2),
            repeats=self.kernel_size[1],
            axis=3,
        )
        return self.grads["x"] * dy

    def local_grads(self, x):
        return self.grads


class BatchNorm2D(Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.epsilon = eps

        self.trainable = True

    def _init_params(self, n_channels):
        self.params["gm"] = np.ones((1, n_channels, 1, 1))
        self.params["bt"] = np.zeros((1, n_channels, 1, 1))

    def init_layer(self, idx):
        super().init_layer(idx)
        self._init_params(self.in_dims[0])

    def forwards(self, x):
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True) + self.epsilon
        invvar = 1.0 / var
        sqrt_invvar = np.sqrt(invvar)

        centered = x - mean
        scaled = centered * sqrt_invvar
        b_normalized = scaled * self.params["gm"] + self.params["bt"]

        # caching intermediate results for backprop
        self.cache["mean"] = mean
        self.cache["var"] = var
        self.cache["invvar"] = invvar
        self.cache["sqrt_invvar"] = sqrt_invvar
        self.cache["centered"] = centered
        self.cache["scaled"] = scaled
        self.cache["normalized"] = b_normalized

        return b_normalized

    def backwards(self, dy):
        dgm = np.sum(self.cache["scaled"] * dy, axis=(0, 2, 3), keepdims=True)
        dbt = np.sum(dy, axis=(0, 2, 3), keepdims=True)

        self.param_updates = {"gm": dgm, "bt": dbt}
        dy = self.params["gm"] * dy
        dx = self.grads["X"] * dy

        return dx

    def local_grads(self, x):
        n, c, h, w = x.shape
        ppc = h * w

        dsqrt_invvar = self.cache["centered"]
        dinvvar = (2.0 * np.sqrt(self.cache["var"])) * dsqrt_invvar
        dvar = (-1.0 / self.cache["var"] ** 2) * dinvvar
        ddenom = self.cache["centered"] * (2 * (ppc - 1) / ppc**2) * dvar

        dcentered = self.cache["sqrt_invvar"]
        dnum = ppc * dcentered

        dx = ddenom + dnum

        grads = {"X": dx}
        return grads


class Conv2D(Layer):
    def __init__(
        self,
        out_channels,
        in_channels=None,
        kernel_size=3,
        stride=1,
        padding=0,
        activation=None,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.trainable = True
        self.padding = padding
        self.activation = activation
        self.activation_f = activations_dict[activation]() if activation else None

    def _init_params(self, in_channels, out_channels, kernel_size, activation):
        scale = scales[activation](in_channels * kernel_size[0] * kernel_size[1])

        self.params["W"] = scale * np.random.randn(
            out_channels, in_channels, *kernel_size
        )

        self.params["b"] = np.zeros((out_channels, 1))

    def init_layer(self, idx):
        super().init_layer(idx)

        self.in_channels = self.in_dims[0]

        self.out_dims = (
            self.out_channels,
            1
            + ((self.in_dims[1] + 2 * self.padding) - self.kernel_size[0])
            // self.stride[0],
            1
            + ((self.in_dims[2] + 2 * self.padding) - self.kernel_size[1])
            // self.stride[1],
        )
        self._init_params(
            self.in_channels, self.out_channels, self.kernel_size, self.activation
        )

    def forwards(self, x):
        """
        Forward pass for Conv2D layer

        params:
            x: np.ndarray of shape (batches, channels, height, width)

        returns:
            y: np.ndarray of shape (batches, channels_out, height_out, width_out)
        """

        if self.padding:
            x = zero_pad(x, self.padding, (2, 3))

        self.cache["x"] = x

        n, c, h, w = x.shape
        kh, kw = self.kernel_size

        # TODO: Please vectorize.
        out_shape = (
            n,
            self.out_channels,
            1 + (h - kh) // self.stride[0],
            1 + (w - kw) // self.stride[1],
        )
        y = np.zeros(out_shape)
        for b in range(n):
            for ch in range(self.out_channels):
                for h, w in product(range(out_shape[2]), range(out_shape[3])):
                    h_offset, w_offset = h * self.stride[0], w * self.stride[1]
                    curr_field = x[
                        b, :, h_offset : h_offset + kh, w_offset : w_offset + kw
                    ]

                    y[b, ch, h, w] = (
                        np.sum(self.params["W"][ch] * curr_field) + self.params["b"][ch]
                    )

        a = self.activation_f(y) if self.activation else y

        return a

    def backwards(self, dy):
        dy = self.activation_f.backwards(dy) if self.activation else dy
        x = self.cache["x"]

        # Calculate global gradient
        dx = np.zeros_like(x)

        n, c, H, W = dx.shape
        kh, kw = self.kernel_size
        for b in range(n):
            for ch in range(self.out_channels):
                for h, w in product(range(dy.shape[2]), range(dy.shape[3])):
                    h_offset, w_offset = h * self.stride[0], w * self.stride[1]
                    dx[b, :, h_offset : h_offset + kh, w_offset : w_offset + kw] += (
                        self.params["W"][ch] * dy[b, ch, h, w]
                    )

        # Calculate global gradients w.r.t weights
        dw = np.zeros_like(self.params["W"])
        for ch_o in range(self.out_channels):
            for ch_i in range(self.in_channels):
                for h, w in product(range(kh), range(kw)):
                    curr_field = x[
                        :,
                        ch_i,
                        h : H - kh + h + 1 : self.stride[0],
                        w : W - kw + w + 1 : self.stride[1],
                    ]
                    dy_field = dy[:, ch_o]

                    dw[ch_o, ch_i, h, w] = np.sum(curr_field * dy_field)

        # Calculate global gradients w.r.t biases
        db = np.sum(dy, axis=(0, 2, 3)).reshape(-1, 1)

        self.param_updates = {"W": dw, "b": db}

        return dx[:, :, self.padding : -self.padding, self.padding : -self.padding]


class RNN(Layer):
    def __init__(self, cell, return_sequences=False, return_state=False, reverse=False):
        super().__init__()

        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.reverse = reverse
        self.trainable = True

        self.states = []

        self.out_sequences = []
        self.activation_ins = []

    def init_layer(self, idx):
        super().init_layer(idx)
        self.cell.in_dims = self.in_dims[-2]
        self.cell.init_layer(idx)
        temp = list(self.in_dims)
        temp[-2] = self.cell.out_dims
        self.out_dims = tuple(temp)

    def forwards(self, x):
        self.states = []

        self.out_sequences = []
        self.cache["x"] = x
        self.activation_ins = []
        # TODO: 😩 Please implement parallel mini-batching, and please reduce the number of loops
        for seq in x:
            out_seq = []
            states = [self.cell.state]
            activation_ins = []

            seq = reversed(seq) if self.reverse else seq
            for el in seq:
                out, state, act_in = self.cell(el)

                out_seq.append(out)
                states.append(state)
                activation_ins.append(act_in)
            self.cell.reset_state()
            self.out_sequences.append((out_seq))
            self.states.append(states)
            self.activation_ins.append(activation_ins)

        y = (
            np.array(self.out_sequences)
            if self.return_sequences
            else np.array(self.out_sequences[-1])
        )
        return y if not self.return_state else (y, self.states)

    def backwards(self, dy):
        dxs = []
        param_updates = [0 for _ in range(self.cell.param_len)]
        for dseq_idx, dseq in enumerate(dy):
            ds_prev = self.cell.state_delta()
            dxseq = []

            dseq = reversed(dseq) if self.reverse else dseq
            for dyel_idx, dyel in reversed(list(enumerate(dseq))):
                dx, ds_prev, param_updates = self.cell.backwards(
                    dyel,
                    ds_prev,
                    param_updates,
                    self.cache["x"][dseq_idx][dyel_idx],
                    self.states[dseq_idx][dyel_idx],
                    self.states[dseq_idx][dyel_idx + 1],
                    self.activation_ins[dseq_idx][dyel_idx],
                )
                dxseq.append(dx)
            dxs.append(dxseq)
        # param_updates = [grad/len(dy) for grad in param_updates]
        self.param_updates = [np.clip(grad, -1, 1) for grad in param_updates]
        # print(self.param_updates)
        return np.array(dxs)

    def _update_params(self, lr):
        self.cell.param_updates = self.param_updates
        self.cell._update_params(lr)


class RNNCell(Layer):
    def __init__(self, units, in_dims=None, activation="tanh"):
        super().__init__()

        if activation not in activations_dict.keys():
            if activation is not None:
                raise ValueError(f"Activation function {activation} not supported")

        self.in_dims = in_dims
        self.out_dims = units
        self.state_dims = units
        self.trainable = True
        self.activation = activation
        self.activation_f = activations_dict[activation]()
        self.state = np.zeros((self.state_dims, 1))

        self.param_len = 5

    def _init_params(self, in_dims, state_dims, out_dims, activation):
        scale = scales[activation](in_dims)

        # Input params
        self.params["Wx"] = scale * np.random.randn(state_dims, in_dims)

        # Hidden params
        self.params["Ws"] = scale * np.random.randn(state_dims, state_dims)
        self.params["bs"] = np.zeros((state_dims, 1))

        # Output params
        self.params["Wy"] = scale * np.random.randn(out_dims, state_dims)
        self.params["by"] = np.zeros((out_dims, 1))

    def init_layer(self, idx):
        self._init_params(self.in_dims, self.state_dims, self.out_dims, self.activation)
        super().init_layer(idx)

    def forwards(self, x):
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
        state = np.dot(self.params["Ws"], self.state) + self.params["bs"]
        tanh_in = x + state
        new_state = self.activation_f(tanh_in)
        self.state = new_state

        y = np.dot(self.params["Wy"], new_state) + self.params["by"]

        return y, new_state, tanh_in

    def backwards(
        self, dy, ds_prev, prev_param_updates, inputs, prev_state, curr_state, tanh_in
    ):
        dwy = np.dot(dy, curr_state.T)
        dby = dy
        dsa = np.dot(self.grads["dsa"].T, dy) + ds_prev

        self.activation_f.local_grads(tanh_in)
        dtanh = self.activation_f.backwards(dsa)
        dbs = dtanh
        dws = np.dot(dtanh, prev_state.T)
        dwx = np.dot(dtanh, inputs.T)
        dx = np.dot(self.grads["dx"].T, dtanh)
        ds_prev = np.dot(self.grads["dsp"].T, dtanh)

        param_updates = [
            x + y for x, y in zip(prev_param_updates, [dwy, dws, dwx, dby, dbs])
        ]

        return dx, ds_prev, param_updates

    def local_grads(self, x):
        dsa = self.params["Wy"]
        ds_p = self.params["Ws"]
        dx = self.params["Wx"]

        grads = {"dx": dx, "dsa": dsa, "dsp": ds_p}
        return grads

    def _update_params(self, lr):
        self.param_updates = {
            "Wy": self.param_updates[0],
            "Ws": self.param_updates[1],
            "Wx": self.param_updates[2],
            "by": self.param_updates[3],
            "bs": self.param_updates[4],
        }
        super()._update_params(lr)

    def reset_state(self):
        self.state = np.zeros((self.state_dims, 1))

    def state_delta(self):
        return np.zeros((self.state_dims, 1))


class LSTMCell(Layer):
    def __init__(
        self, units, in_dims=None, activation="tanh", recurrent_activation="sigmoid"
    ):
        super().__init__()

        self.in_dims = in_dims
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.trainable = True

        self.activation_f = activations_dict.get(activation)()
        self.recurrent_activation_f = activations_dict.get(recurrent_activation)()

        self.state_c = np.zeros((self.units, 1))
        self.state_h = np.zeros((self.units, 1))
        self.state = (self.state_c, self.state_h)
        self.inter_states = ()

        self.param_len = 8

    def _init_params(
        self, in_dims, concat_dims, state_dims, activation, recurrent_activation
    ):
        rscale = scales[recurrent_activation](concat_dims)
        scale = scales[activation](concat_dims)

        self.params["Wf"] = np.random.randn(state_dims, concat_dims) * rscale
        self.params["Wc"] = np.random.randn(state_dims, concat_dims) * scale
        self.params["Wi"] = np.random.randn(state_dims, concat_dims) * rscale
        self.params["Wo"] = np.random.randn(state_dims, concat_dims) * rscale

        self.params["bg"] = np.zeros((state_dims, 1))
        self.params["bc"] = np.zeros((state_dims, 1))
        self.params["bi"] = np.zeros((state_dims, 1))
        self.params["bo"] = np.zeros((state_dims, 1))

    def init_layer(self, idx):
        super().init_layer(idx)

        self.out_dims = self.units
        self.concat_dims = self.in_dims + self.units

        self._init_params(
            self.in_dims,
            self.concat_dims,
            self.units,
            self.activation,
            self.recurrent_activation,
        )

    def forwards(self, x):
        xcon = np.concatenate((x, self.state_h), axis=0)

        a_sf = np.dot(self.params["Wf"], xcon) + self.params["bg"]
        sf = self.recurrent_activation_f(a_sf)
        a_sc = np.dot(self.params["Wc"], xcon) + self.params["bc"]
        sc = self.activation_f(a_sc)
        a_si = np.dot(self.params["Wi"], xcon) + self.params["bi"]
        si = self.recurrent_activation_f(a_si)
        a_so = np.dot(self.params["Wo"], xcon) + self.params["bo"]
        so = self.recurrent_activation_f(a_so)

        sc = sc * self.state_c + (sf * si)
        sh = self.activation_f(sc) * so

        self.state_c, self.state_h = sc, sh
        self.state = (sc, sh)
        self.inter_states = (a_sf, a_sc, a_si, a_so, xcon)

        return sh, self.state, self.inter_states

    def backwards(
        self, dy, ds_prev, prev_param_updates, inputs, prev_state, curr_state, act_ins
    ):
        sc_p, sh_p = prev_state
        sc, sh = curr_state
        f, c, i, o, xcon = act_ins
        dsc_p, dsh_p = ds_prev
        dsh = dsh_p + dy

        dsc = (
            dsh * self.recurrent_activation_f(o) * self.activation_f.local_grads(c)["X"]
        ) + dsc_p
        do = self.activation_f(sc) * dsh
        df = sc_p * dsc
        dc = self.recurrent_activation_f(i) * dsc
        di = self.activation_f(c) * dsc

        df_in = self.recurrent_activation_f.local_grads(f)["X"] * df
        di_in = self.recurrent_activation_f.local_grads(i)["X"] * di
        do_in = self.recurrent_activation_f.local_grads(o)["X"] * do
        dc_in = self.activation_f.local_grads(c)["X"] * dc

        dWf = np.dot(df_in, xcon.T)
        dWc = np.dot(dc_in, xcon.T)
        dWi = np.dot(di_in, xcon.T)
        dWo = np.dot(do_in, xcon.T)
        dbf = df_in
        dbc = dc_in
        dbi = di_in
        dbo = do_in

        param_updates = [dWf, dWc, dWi, dWo, dbf, dbc, dbi, dbo]
        param_updates = [a + b for a, b in zip(prev_param_updates, param_updates)]

        dxcon = np.zeros_like(xcon)
        dxcon += np.dot(self.grads["dxcf"].T, df_in)
        dxcon += np.dot(self.grads["dxcc"].T, dc_in)
        dxcon += np.dot(self.grads["dxci"].T, di_in)
        dxcon += np.dot(self.grads["dxco"].T, do_in)

        dsc_next = dsc * f
        dsh_next = dxcon[self.in_dims :]

        dx = dxcon[: self.in_dims]

        ds_next = (dsc_next, dsh_next)

        return dx, ds_next, param_updates

    def local_grads(self, x):
        dxcf = self.params["Wf"]
        dxcc = self.params["Wc"]
        dxci = self.params["Wi"]
        dxco = self.params["Wo"]

        grads = {"dxcf": dxcf, "dxcc": dxcc, "dxci": dxci, "dxco": dxco}
        return grads

    def _update_params(self, lr):
        self.param_updates = {
            "Wf": self.param_updates[0],
            "Wc": self.param_updates[1],
            "Wi": self.param_updates[2],
            "Wo": self.param_updates[3],
            "bg": self.param_updates[4],
            "bc": self.param_updates[5],
            "bi": self.param_updates[6],
            "bo": self.param_updates[7],
        }
        super()._update_params(lr)

    def reset_state(self):
        self.state_c = np.zeros((self.units, 1))
        self.state_h = np.zeros((self.units, 1))
        self.state = (self.state_c, self.state_h)
        self.inter_states = ()

    def state_delta(self):
        dstate_c = np.zeros((self.units, 1))
        dstate_h = np.zeros((self.units, 1))
        dstate = (dstate_c, dstate_h)
        return dstate


class Embedding(Layer):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.softmax = Softmax()

    def init_layer(self, idx):
        super().init_layer(idx)

        self.out_dims = list(self.in_dims)
        self.out_dims[-2] = self.embedding_dim
        self.out_dims = tuple(self.out_dims)

        self._init_params(self.embedding_dim, self.vocab_size)

    def _init_params(self, embedding_dim, vocab_size):
        self.params["We"] = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

    def forwards(self, x):
        ohenc = np.array([one_hot_encoding(tok, self.vocab_size) for tok in x])
        self.cache["shape"] = ohenc.shape
        new_shape = list(ohenc.shape)
        new_shape[-2] = self.embedding_dim
        self.cache["X"] = ohenc

        embed = np.dot(ohenc.reshape((-1, self.vocab_size)), self.params["We"])

        return embed.reshape(new_shape)

    def local_grads(self, x):
        dxe = self.params["We"]

        grads = {"dxe": dxe}
        return grads

    def backwards(self, dy):
        dy_prev = dy.reshape((-1, self.embedding_dim)).dot(self.grads["dxe"].T)

        dw = dy.reshape((-1, self.embedding_dim)).T.dot(
            self.cache["X"].reshape((-1, self.vocab_size))
        )

        self.param_updates["We"] = dw
        return dy_prev.reshape(self.cache["shape"])


# TODO: Implement self attention
class SelfAttention(Layer):
    def __init__(self, attention_dim=None):
        super().__init__()

        self.softmax = Softmax()
        if not attention_dim:
            self.attention_dim = self.in_dims

    def _init_params(self, in_dims, attention_dim):
        self.params["wq"] = np.random.randn(in_dims, attention_dim)
        self.params["wk"] = np.random.randn(in_dims, attention_dim)
        self.params["wv"] = np.random.randn(in_dims, attention_dim)

    def init_layer(
        self,
        idx,
    ):
        self._init_params(self.in_dims, self.attention_dim)
        super().init_layer(idx)

    def forwards(self, x):
        q = np.dot(x, self.params["wq"])
        k = np.dot(x, self.params["wk"])
        v = np.dot(x, self.params["wv"])

        pre_scores = q.dot(k.T)
        scores = self.softmax(pre_scores)

        return scores
