import random
from .autograd import Node

random.seed(42)

class Neuron:

    def __init__(
            self, 
            n: int,
            weights: list | None = None,
            bias: float | None = None,
            label: str | None = None) -> None:
        self.n_inputs = n
        self.label = label
        self.weights = weights if weights else [
            Node(
                data=random.uniform(-1, 1),
                label=f"{label} | weight: {i+1}" if label else None
            ) for i in range(n)
        ]
        self.bias = bias if bias is not None else \
            Node(
                random.uniform(-1, 1), 
                label=label + f" | bias" if label else None
            )

    def _forward_linear(self, X):
        assert len(X) == self.n_inputs
        return sum(
            (xi*wi for xi, wi in zip(X, self.weights)),
            start=self.bias
        )

    def __call__(self, X):
        return self._forward_linear(X)

    def parameters(self)-> list | list[Node]:
        return self.weights + [self.bias]

class RNNCell(Neuron):

    def __init__(self, 
                 n, 
                 weights = None, 
                 bias = None,
                 recurrent_state: float = 0.0,
                 recurrent_weight: float | None = None,
                 label: str | None = None
                 ) -> None:
        super().__init__(n, weights, bias, label)
        self.recurrent_state = Node(data=recurrent_state, label='recurrent state')
        self.recurrent_weight = Node(data=recurrent_weight) if recurrent_weight is not None \
            else Node(data=random.uniform(-1, 1), label='recurrent weight')

    def __call__(self, X):
        activation = (
            self._forward_linear(X)
            + self.recurrent_state*self.recurrent_weight
        )
        self.recurrent_state = activation
        return activation

    def parameters(self):
        return super().parameters() + [self.recurrent_weight]

    def reset_state(self):
        self.recurrent_state = Node(data=0.0)