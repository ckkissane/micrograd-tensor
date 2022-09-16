import numpy as np
from engine import Tensor


def cross_entropy(input: Tensor, target: int):
    softmax_num = np.exp(input.data)
    softmax_denom = np.sum(softmax_num)
    probs = softmax_num / softmax_denom

    p = probs[target]
    out = Tensor([-np.log(p)], (input,), "cross_entropy")

    def _backward():
        target_one_hot = np.zeros_like(probs)
        target_one_hot[target] = 1
        input.grad += (probs - target_one_hot) * out.grad

    out._backward = _backward

    return out
