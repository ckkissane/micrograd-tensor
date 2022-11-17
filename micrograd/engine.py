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

    def backward(self, gradient=None):
        topo = []
        visited = set()

        def build_topo(node: Tensor):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = gradient if gradient is not None else np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            # TODO: make this more robust for different vector / matrix shapes
            if out.grad.ndim == 1:  # vector matrix product
                self.grad += out.grad.dot(other.data.T)
                other.grad += self.data[None, ...].T.dot(out.grad[None, ...])
            elif out.grad.ndim == 2:  # standard 2D  matmul
                self.grad += out.grad.dot(other.data.T)
                other.grad += self.data.T.dot(out.grad)
            else:  # batched matmul
                self.grad += micrograd.unbroadcast(
                    out.grad @ other.data.swapaxes(-1, -2), self.shape
                )
                other.grad += micrograd.unbroadcast(
                    self.data.swapaxes(-1, -2) @ out.grad, other.shape
                )

        out._backward = _backward

        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += micrograd.unbroadcast(out.grad, self.data.shape)
            other.grad += micrograd.unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __sub__(self, other):
        out = Tensor(
            self.data - other.data,
            (
                self,
                other,
            ),
            "-",
        )

        def _backward():
            self.grad += micrograd.unbroadcast(out.grad, self.shape)
            other.grad += micrograd.unbroadcast(-out.grad, other.shape)

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += micrograd.unbroadcast(other.data * out.grad, self.shape)
            other.grad += micrograd.unbroadcast(self.data * out.grad, other.shape)

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
            # TODO: there must be a cleaner way?
            swapped_s = np.swapaxes(s, dim, -1)
            local_deriv = micrograd.diag_embed(swapped_s).data - (
                np.expand_dims(swapped_s, axis=-2) * np.expand_dims(swapped_s, axis=-1)
            )
            swapped_out_grad = np.swapaxes(out.grad, dim, -1)
            reshaped_out_grad = swapped_out_grad.reshape(-1, 1, swapped_s.shape[-1])
            reshaped_local_deriv = local_deriv.reshape(
                -1, swapped_s.shape[-1], swapped_s.shape[-1]
            )
            res = reshaped_out_grad @ reshaped_local_deriv
            res = res.reshape(swapped_out_grad.shape).swapaxes(dim, -1)
            self.grad += res

        out._backward = _backward

        return out

    def mean(self, dim=None, keepdims=False):
        out = Tensor(self.data.mean(axis=dim, keepdims=keepdims), (self,), "mean")

        def _backward():
            if not dim:
                denom = self.data.size
            elif isinstance(
                dim,
                (
                    tuple,
                    list,
                ),
            ):
                denom = np.prod([self.data.shape[d] for d in dim])
            else:
                denom = self.data.shape[dim]
            out_grad = out.grad if keepdims else np.expand_dims(out.grad, axis=dim)
            self.grad += out_grad / denom

        out._backward = _backward

        return out

    def var(self, dim=None, unbiased=True, keepdims=False):
        out = Tensor(
            self.data.var(axis=dim, ddof=unbiased, keepdims=keepdims), (self,), "var"
        )

        def _backward():
            if not dim:
                denom = self.data.size
            elif isinstance(
                dim,
                (
                    tuple,
                    list,
                ),
            ):
                denom = np.prod([self.data.shape[d] for d in dim])
            else:
                denom = self.data.shape[dim]

            if unbiased:
                denom -= 1

            out_grad = out.grad if keepdims else np.expand_dims(out.grad, axis=dim)
            mu = self.data.mean(axis=dim, keepdims=True)
            local_deriv = (2.0 / denom) * (self.data - mu)
            self.grad += local_deriv * out_grad

        out._backward = _backward

        return out
