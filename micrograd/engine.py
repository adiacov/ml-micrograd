import math


class Value:
    """Value is the single scalar node in a computational graph."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._prev = _children
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op="+")
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def tahn(self):
        # Hyperbolic tangent definition: (exp(2*x) - 1) / (exp(2*x) + 1)
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), "tahn")
        return out
