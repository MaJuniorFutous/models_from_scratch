import numpy as np
from modules import Neuron, RNNCell

class LinearLayer:

    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 neuron_cls = Neuron,
                 label: str | None = None):
        self.in_features = in_features
        self.out_features = out_features
        self.label = label
        self.neurons = [
            neuron_cls(
                n=self.in_features, 
                label=None if not label else label + f' | neuron: {i+1}'
            ) for i in range(self.out_features)]

    def __repr__(self) -> str:
        return f"{type(self).__name__} (in: {self.in_features}, out: {self.out_features})"

    def __call__(self, x: list | np.ndarray):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

class RNNLinearLayer(LinearLayer):

    def __init__(
        self,
        in_features,
        out_features,
        neuron_cls=RNNCell,
        label=None
    ):
        super().__init__(
            in_features,
            out_features,
            neuron_cls,
            label
        )

    def reset_state(self):
        for neuron in self.neurons: neuron.reset_state()