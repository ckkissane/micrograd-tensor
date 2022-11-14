import numpy as np
import micrograd


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data  # np.array
        self.grad = np.zeros_like(self.data)
        self.shape = data.shape
        self.ndim = data.ndim
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
            if out.grad.ndim == 1:
                other.grad += self.data[None, ...].T.dot(out.grad[None, ...])
            else:
                other.grad += self.data.T.dot(out.grad)

        out._backward = _backward

        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += micrograd.unbroadcast(out.grad, self.data.shape)
            other.grad += micrograd.unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __truediv__(self, other):
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += micrograd.unbroadcast(out.grad / other.data, self.shape)
            other.grad += micrograd.unbroadcast(
                -(self.data / other.data**2) * out.grad, other.shape
            )

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += out.grad * (out.data > 0)

        out._backward = _backward

        return out

    def reshape(self, newshape):
        out = Tensor(np.reshape(self.data, newshape), (self,), "reshape")

        def _backward():
            self.grad += np.reshape(out.grad, self.data.shape)

        out._backward = _backward

        return out

    def transpose(self, dim0, dim1):
        out = Tensor(self.data.swapaxes(dim0, dim1), (self,), "transpose")

        def _backward():
            self.grad += out.grad.swapaxes(dim0, dim1)

        out._backward = _backward

        return out

    def softmax(self, dim):
        maxes = self.data.max(axis=dim, keepdims=True)
        softmax_num = np.exp(self.data - maxes)
        softmax_denom = softmax_num.sum(axis=dim, keepdims=True)
        s = softmax_num / softmax_denom
        out = Tensor(s, (self,), "softmax")

        def _backward():
            a = np.swapaxes(s, dim, -1)
            local_deriv = micrograd.diag_embed(a).data - (
                np.expand_dims(a, axis=-2) * np.expand_dims(a, axis=-1)
            )
            local_deriv = np.swapaxes(local_deriv, dim, -1)
            self.grad = self.grad + out.grad.dot(local_deriv)

        out._backward = _backward

        return out
