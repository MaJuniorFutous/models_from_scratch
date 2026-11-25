from typing import Any
import numpy as np
from modules.layers import LinearLayer
from modules.autograd import Node

class Network:

    def __init__(self, layers: list, layer_cls = LinearLayer, labels: bool = False) -> None:
        self.layers = [layer_cls(
            in_features=layers[i], 
            out_features=layers[i+1],
            label=f'{layer_cls.__name__} {i+1}' if labels else None
            ) for i in range(len(layers) - 1)]

    def __repr__(self) -> str:
            return f"Net (Layers ({[layer for layer in self.layers]}))"

    def __call__(self, x: list | np.ndarray, no_act: set | None = None):
        for i, layer in enumerate(self.layers): 
            x = layer(x, act=False if i in no_act else True)
        return x[0] if len(x) == 1 else x # if only one feature

    def parameters(self)-> list[Any | Node]:
        return [param for layer in self.layers for param in layer.parameters()]

    def zero_grad(self) -> None:
        for param in self.parameters(): param.grad = 0

class NetworkVectorized:

    def __init__(self, layers: list, layer_cls = LinearLayer, labels: bool = False):
        self.layers = [layer_cls(
            in_features=layers[i], 
            out_features=layers[i+1],
            label=f'{layer_cls.__name__} {i+1}' if labels else None
            ) for i in range(len(layers) - 1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU on hidden layers
            if i != len(self.layers) - 1:
                x = [
                    [node.relu() for node in row]
                    for row in x
                ]
        return x[0] if len(x) == 1 else x # if only one feature

    def __repr__(self) -> str:
        return f"Net (Layers ({[layer for layer in self.layers]}))"

    def parameters(self):
        return [
            param
            for layer in self.layers
            for param in layer.parameters()
        ]

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0