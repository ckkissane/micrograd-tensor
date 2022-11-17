from typing import List
from .engine import Tensor
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


class Adam:
    def __init__(self, params: List[Tensor], lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr

        self.t = 0
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def __repr__(self):
        return f"Adam(lr={self.lr})"

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g**2
            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)
            p.data -= self.lr * m_hat / np.sqrt(v_hat + self.eps)
