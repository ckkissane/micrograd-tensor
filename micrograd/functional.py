import numpy as np
from .engine import Tensor


def cross_entropy(input: Tensor, target: int):
    """
    Computes the cross entropy loss between input logits and target

    Args:
        input (micrograd.Tensor) raw logits of shape (C)
        target (np.array) Ground truth class indices of shape ()
    """
    input_max = input.data.max(axis=-1, keepdims=True)
    softmax_num = np.exp(input.data - input_max)
    softmax_denom = np.sum(softmax_num)
    probs = softmax_num / softmax_denom
    p = probs[target]
    out = Tensor(-np.log(p).mean(keepdims=True), (input,), "cross_entropy")

    def _backward():
        input.grad = np.copy(probs)
        input.grad[target] -= 1.0

    out._backward = _backward

    return out


def batched_cross_entropy(input: Tensor, target):
    """
    Computes the cross entropy loss between input logits and target

    Args:
        input (micrograd.Tensor) raw logits of shape (N, C)
        target (np.array) Ground truth class indices of shape (N)
    """
    input_max = input.data.max(axis=-1, keepdims=True)
    softmax_num = np.exp(input.data - input_max)
    softmax_denom = np.sum(softmax_num, axis=-1, keepdims=True)
    probs = softmax_num / softmax_denom
    p = probs[np.arange(probs.shape[0]), target]
    out = Tensor(-np.log(p).mean(keepdims=True), (input,), "cross_entropy")

    def _backward():
        batch_size = probs.shape[0]
        input.grad = np.copy(probs)
        input.grad[np.arange(batch_size), target] -= 1.0
        input.grad /= batch_size

    out._backward = _backward

    return out


def unbroadcast(grad, shape):
    # grad: np.array
    # shape: tuple
    new_shape = (1,) * (len(grad.shape) - len(shape)) + shape
    axs = tuple(i for i, d in enumerate(new_shape) if d == 1)
    out_grad = grad.sum(axis=axs, keepdims=True)
    return out_grad.reshape(shape)
