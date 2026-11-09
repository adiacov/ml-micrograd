import random

from micrograd.engine import Value


class Neuron:
    """A neuron is a function o a sequence of inputs, weights and a bias"""

    def __init__(self, nin, label=""):
        """
        nin: int, the number of inputs
        """
        # if label create the prefix with hyphen, empty otherwise. Ex. N-
        self.label = "" if len(label) > 0 else f"{label}-"
        self.w = [
            Value(random.uniform(-1, 1), label=f"{self.label}V{i}") for i in range(nin)
        ]
        self.b = Value(random.uniform(-1, 1), label=f"{label}-b")

    def __call__(self, x):
        """x - input data"""
        act = sum([wi * xi for wi, xi in zip(self.w, x)]) + self.b  # activation
        out = act.tahn()  # normalization
        return out

    def parameters(self):
        return [p for p in self.w]


class Layer:
    """A layer of Neurons.
    nin: number of inputs to a each neuron;
    nout: number of outputs of the layer
    """

    def __init__(self, nin, nout, label=""):
        # if label create the prefix with hyphen, empty otherwise. Ex. L-
        self.label = "" if len(label) > 0 else f"{label}-"
        self.neurons = [Neuron(nin, label=f"{self.label}N{i}") for i in range(nout)]

    def __call__(self, x):
        """x - input data"""
        outs = [n(x) for n in self.neurons]
        out = outs[0] if len(outs) == 1 else outs
        return out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi Layer Perceptron.
    nin - int, number of inputs
    nouts - list of ints, number of inner layers, plus one final layer. Ex.: [4, 1] - 4 inner, 1 out.

    Each inner layer outputs, is the input to the next layer.
    """

    def __init__(self, nin, nouts):
        # if label create the prefix with hyphen, empty otherwise. Ex. M-
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], label=f"L{i}") for i in range(len(nouts))
        ]

    def __call__(self, x):
        """x - input data"""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
