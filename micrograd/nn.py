import numpy as np
from .engine import Tensor


class Module:
    def parameters(self):
        return []


class Sigmoid(Module):
    def __call__(self, x: Tensor):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Linear(Module):
    def __init__(self, nin: int, nout: int):
        self.nin, self.nout = nin, nout
        self.weight = Tensor(
            np.random.uniform(
                -1.0,
                1.0,
                (
                    nin,
                    nout,
                ),
            )
        )
        self.bias = Tensor(np.zeros(nout))

    def __call__(self, x: Tensor):
        xW = x.matmul(self.weight)
        y = xW + self.bias
        return y

    def __repr__(self):
        return f"Linear(nin={self.nin}, nout={self.nout})"

    def parameters(self):
        return [self.weight, self.bias]
