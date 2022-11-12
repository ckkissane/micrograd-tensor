import numpy as np
from .engine import Tensor
from .functional import conv2d


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


class Conv2d(Module):
    def __init__(self, C_in, C_out, K, stride=1, padding=0):
        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.stride = stride
        self.padding = padding

        k = 1.0 / (C_in * K * K)
        self.weight = Tensor(
            np.random.uniform(-np.sqrt(k), np.sqrt(k), (K, K, C_in, C_out))
        )

    def __call__(self, Z: Tensor):
        """
        Args:
            Z: Tensor(N, H, W, C_in)
        """
        return conv2d(Z, self.weight, stride=self.stride, padding=self.padding)

    def __repr__(self):
        return f"Conv2d(C_in={self.C_in}, C_out={self.C_out}, K={self.K}, stride={self.stride}, padding={self.padding}"
