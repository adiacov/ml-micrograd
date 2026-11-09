class Value:
    """Value is the single scalar node in a computational graph."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
        self._prev = _children
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        return self + other

    def tahn(self):
        # Hyperbolic tangent definition: (exp(2*x) - 1) / (exp(2*x) + 1)
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), "tahn")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """Calculate gradients for each Value object in a graph in topological order starting from the last one."""
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)  # add node to seen nodes
                for child in node._prev:
                    build_topo(child)
                topo.append(node)  # append not to the end of topo

        build_topo(self)

        # initialize grad (this is the right most Value object on which backward is called)
        self.grad = 1.0
        # calculate gradients in reversed order, starting with last
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = 0.0
