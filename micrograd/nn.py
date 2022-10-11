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


class ReLU(Module):
    def __call__(self, x: Tensor):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Linear(Module):
    def __init__(self, nin: int, nout: int):
        self.nin, self.nout = nin, nout
        k = 1 / nin
        self.weight = Tensor(
            np.random.uniform(
                -np.sqrt(k),
                np.sqrt(k),
                (
                    nin,
                    nout,
                ),
            )
        )
        self.bias = Tensor(
            np.random.uniform(
                -np.sqrt(k),
                np.sqrt(k),
                (nout,),
            )
        )

    def __call__(self, x: Tensor):
        xW = x.matmul(self.weight)
        y = xW + self.bias
        return y

    def __repr__(self):
        return f"Linear(nin={self.nin}, nout={self.nout})"

    def parameters(self):
        return [self.weight, self.bias]
