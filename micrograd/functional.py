import numpy as np
from .engine import Tensor


def cross_entropy(input: Tensor, target: int):
    # input: (C,) or (N, C)
    # target: () or (N)
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


def batched_cross_entropy(input: Tensor, target):
    # input: (N, C)
    # target: (N)
    softmax_num = np.exp(input.data)
    softmax_denom = np.sum(softmax_num, axis=-1)
    softmax_denom = np.expand_dims(softmax_denom, axis=-1)
    probs = softmax_num / softmax_denom

    p = np.array([probs[i, c] for i, c in enumerate(target)])
    out = Tensor(-np.log(p).mean(keepdims=True), (input,), "cross_entropy")

    def _backward():
        batch_size = probs.shape[0]
        target_one_hot = np.zeros_like(probs)
        for i, c in enumerate(target):
            target_one_hot[i][c] = 1
        input.grad += (probs - target_one_hot) * out.grad / batch_size

    out._backward = _backward

    return out  # (N)


def unbroadcast(grad, shape):
    # grad: np.array
    # shape: tuple
    new_shape = (1,) * (len(grad.shape) - len(shape)) + shape
    axs = tuple(i for i, d in enumerate(new_shape) if d == 1)
    out_grad = grad.sum(axis=axs, keepdims=True)
    return out_grad.reshape(shape)
