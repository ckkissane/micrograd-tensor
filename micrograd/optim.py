from typing import List
from engine import Tensor
import numpy as np


class SGD:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def __repr__(self):
        return f"SGD(lr={self.lr})"

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
