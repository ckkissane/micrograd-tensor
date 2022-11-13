from .engine import Tensor
import numpy as np


def unbroadcast(grad, shape):
    new_shape = (1,) * (len(grad.shape) - len(shape)) + shape
    axs = tuple(i for i, d in enumerate(new_shape) if d == 1)
    out_grad = grad.sum(axis=axs, keepdims=True)
    return out_grad.reshape(shape)


def where(condition, x, y) -> Tensor:
    """
    Args:
        condition: Tensor[bool]. When True, yield x, else yield y
        x: Tensor
        y: Tensor
    """
    out = Tensor(
        np.where(condition.data, x.data, y.data),
        (
            x,
            y,
        ),
        "where",
    )

    def _backward():
        x.grad += unbroadcast(np.where(condition.data, out.grad, 0.0), x.shape)
        y.grad += unbroadcast(np.where(condition.data, 0.0, out.grad), y.shape)

    out._backward = _backward
    return out
