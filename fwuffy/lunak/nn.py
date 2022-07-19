import inspect
import typing
import numpy as np
import json
from collections import defaultdict

from tabulate import tabulate

from .base import Function
from .layers import Layer
from .losses import Loss


class Model:
    def __init__(self):
        pass


class Sequential(Model):
    def __init__(self, layers: list[Layer], input_dim: typing.Union[int, tuple], loss: Loss):
        super().__init__()

        if not isinstance(loss, Loss):
            ValueError("loss function should be an instance of lunak.losses.Loss")
        for layer in layers:
            if not isinstance(layer, Function):
                raise ValueError(
                    "layer should be an instance of lunak.layers.Layer or lunak.base.Function"
                )

        self.loss_func = loss
        self.input_dim = input_dim
        self.layers = self._init_layers(layers)

    def _init_layers(self, layers: list[Layer]):

        curr_input_dim = self.input_dim
        name_counter = defaultdict(int)

        for layer in layers:
            layer.in_dims = curr_input_dim
            l_name = layer.name

            layer.init_layer(name_counter[l_name])
            name_counter[l_name] += 1

            curr_input_dim = layer.out_dims

        return layers

    def __call__(self, x):
        return self.forward(x)

    def loss(self, x, y):

        return self.loss_func(x, y)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x

    def backwards(self):

        dy = self.loss_func.backwards()

        for layer in reversed(self.layers):
            dy = layer.backwards(dy)
        return dy

    def apply_gradients(self, lr):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer._update_params(lr)

    def summary(self):

        null = np.array([])

        layer_names = [layer.name for layer in self.layers]
        output_dims = [layer.out_dims for layer in self.layers]

        param_n = [
            np.prod(layer.params.get("W", null).shape)
            + np.prod(layer.params.get("b", null).shape)
            for layer in self.layers
        ]
        trainable_params = sum(
            [
                np.prod(layer.params.get("W", null).shape)
                + np.prod(layer.params.get("b", null).shape)
                if layer.trainable else 0
                for layer in self.layers
            ]
        )
        untrainable_params = sum(
            [
                np.prod(layer.params.get("W", null).shape)
                + np.prod(layer.params.get("b", null).shape)
                if not layer.trainable else 0
                for layer in self.layers
            ]
        )

        print(f'Model: "{self.__class__.__name__.lower()}"')
        print(
            tabulate(
                zip(layer_names, output_dims, param_n),
                headers=["Layer (type)", "Output dim", "Param #"],
                tablefmt="grid",
            )
        )
        print(f"Trainable params: {trainable_params}")
        print(f"Non trainable params: {untrainable_params}")

    def save(self, path):
        saved_model = {}
        saved_model["model_name"] = self.__class__.__name__
        saved_model["layers_count"] = len(self.layers)
        saved_model["input_dims"] = self.input_dim
        saved_model["loss_func"] = self.loss_func.name
        saved_model["layers"] = []

        for layer in self.layers:
            saved_layer = {}
            saved_layer["layer_name"] = layer.name
            saved_layer["layer_type"] = layer.__class__.__name__
            args = {
                arg: layer.__dict__.get(arg)
                for arg in inspect.signature(layer.__class__.__init__).parameters.keys()
            }
            args.pop("self")
            saved_layer["layer_args"] = args
            saved_layer["layer_params"] = {
                key: val.tolist() for key, val in layer.params.items()
            }
            saved_model["layers"].append(saved_layer)

        with open(path + ".json", "w") as f:
            json.dump(saved_model, f)
