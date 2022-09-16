import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data  # np.array
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def sigmoid(self):
        def sig_np(x: np.array):
            return 1.0 / (1.0 + np.exp(-x))

        sig = sig_np(self.data)
        out = Tensor(sig, (self,), "sigmoid")

        def _backward():
            self.grad += sig * (1.0 - sig) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node: Tensor):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            # TODO: make this more robust for different vector / matrix shapes
            self.grad += out.grad.dot(other.data.T)
            other.grad += self.data[None, ...].T.dot(out.grad[None, ...])

        out._backward = _backward

        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out
