import numpy as np
from .engine import Tensor
from .functional import conv2d, layer_norm, embedding
from typing import List


def get_parameters(obj) -> List[Tensor]:
    if isinstance(obj, Module):
        return obj.parameters()
    elif isinstance(obj, Tensor):
        return [obj]
    elif isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        res = []
        for value in obj:
            res.extend(get_parameters(value))
        return res
    elif isinstance(obj, dict):
        res = []
        for value in obj.values():
            res.extend(get_parameters(value))
        return res
    else:
        return []


class Module:
    def parameters(self) -> List[Tensor]:
        return get_parameters(self.__dict__)

    def __repr__(self):
        res = [type(self).__name__ + "("]
        for child in self.__dict__.values():
            child_str = repr(child) + ","
            child_lines = child_str.split("\n")
            for i, line in enumerate(child_lines):
                child_lines[i] = "\t" + line
            child_str = "\n".join(child_lines)
            res.append(child_str)
        res.append(")")
        return "\n".join(res)


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
        return f"Conv2d(C_in={self.C_in}, C_out={self.C_out}, K={self.K}, stride={self.stride}, padding={self.padding})"

    def parameters(self):
        return [self.weight]


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = Tensor(np.ones(normalized_shape))
            self.bias = Tensor(np.zeros(normalized_shape))

    def __call__(self, x: Tensor) -> Tensor:
        return layer_norm(
            x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps
        )

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    def parameters(self):
        res = []
        if self.weight:
            res.append(self.weight)
        if self.bias:
            res.append(self.bias)
        return res


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim))

    def __call__(self, x: Tensor) -> Tensor:
        return embedding(x, self.weight)

    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"

    def parameters(self):
        return [self.weight]


class Sequential(Module):
    def __init__(self, *args):
        self.layers = list(args)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        res = ["Sequential("]
        for child in self.layers:
            res.append("\t" + repr(child) + ",")
        res.append(")")
        return "\n".join(res)
