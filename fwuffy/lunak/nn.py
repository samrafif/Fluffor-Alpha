import inspect
import pickle
import sys
import typing
import numpy as np
import json
import pathlib
import zipfile
import tempfile
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

        layer_names = [layer.name + f" ({layer.__class__.__name__})" for layer in self.layers]
        output_dims = [layer.out_dims for layer in self.layers]

        param_n = [
            sum([
                np.prod(val.shape) for val in layer.params.values()
            ])
            for layer in self.layers
        ]
        trainable_params = sum(
            [
                sum([
                    np.prod(val.shape) for val in layer.params.values()
                ])
                if layer.trainable else 0
                for layer in self.layers
            ]
        )
        untrainable_params = sum(
            [
                sum([
                    np.prod(val.shape) for val in layer.params.values()
                ])
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
        with tempfile.TemporaryDirectory() as tmpdirname:
            directory = pathlib.Path(tmpdirname)
            with zipfile.ZipFile(f"{path}.zip", mode="w") as archive:
                saved_model = {
                    "model_name": self.__class__.__name__,
                    "layers_count": len(self.layers),
                    "input_dims": self.input_dim,
                    "loss_func": self.loss_func.name,
                    "layers": []
                }

                for layer in self.layers:
                    args = {
                        arg: layer.__dict__.get(arg)
                        for arg in inspect.signature(layer.__class__.__init__).parameters.keys()
                    }
                    args.pop("self")
                    saved_layer = {
                        "layer_name": layer.name,
                        "layer_type": layer.__class__.__name__,
                        "layer_params": list(layer.params.keys())
                    }
                    saved_model["layers"].append(saved_layer)

                    dp = directory / layer.name
                    dp.mkdir()
                    
                    args_fp = dp / "args"
                    with args_fp.open("wb") as f:
                        pickle.dump(args, f)
                    
                    for key, val in layer.params.items():
                        fp = dp / key
                        np.save(fp, val)

                fp = directory / "mnfst.json"
                with fp.open("w") as f:
                    json.dump(saved_model, f)

                for file_path in  [f for f in directory.resolve().glob('**/*') if f.is_file()]: 
                    path = "\\".join(file_path.parts[len(directory.parts):])
                    archive.write(file_path, arcname=path)

def load_model(path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        directory = pathlib.Path(tmpdirname)
        with zipfile.ZipFile(f"{path}.zip", mode="r") as archive:
            archive.extractall(directory)
            manisfest_fp = directory / "mnfst.json"
            with manisfest_fp.open("r") as mnsft_f:
                mnsft = json.load(mnsft_f)
            
            model_class = globals()[mnsft["model_name"]]
            model_loss_class = getattr(sys.modules["fwuffy.lunak.losses"], mnsft["loss_func"])
            model_inp_dim = mnsft["input_dims"]
            model = model_class([], model_inp_dim, model_loss_class())
            
            layers = []
            
            for layer in mnsft["layers"]:
                layer_class = getattr(sys.modules["fwuffy.lunak.layers"], layer["layer_type"])
                layer_name = layer["layer_name"]
                
                layer_args_fp = directory / layer_name / "args"
                with layer_args_fp.open("rb") as f:
                    layer_args = pickle.load(f)
                
                layer_obj = layer_class(**layer_args)
                layer_obj.name = layer_name
                layer_params = layer["layer_params"]
                params = {}
                for param in layer_params:
                    param_fp = directory / layer_name / (param + ".npy")
                    param_d = np.load(param_fp)
                    params[param] = param_d
                layer_obj.params = params
                
                layers.append(layer_obj)
            
            model.layers = layers
            return model
