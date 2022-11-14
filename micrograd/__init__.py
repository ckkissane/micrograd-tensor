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


def diag_embed(x, offset=0, dim1=-1, dim2=-2):
    nDims = x.ndim + 1
    dim1 = dim1 + (dim1 < 0) * nDims
    dim2 = dim2 + (dim2 < 0) * nDims
    new_dim_len = abs(offset) + x.shape[-1]
    sizes = list(x.shape)
    sizes.pop()
    sizes.insert(min(dim1, dim2), new_dim_len)
    sizes.insert(max(dim1, dim2), new_dim_len)
    res = np.zeros(sizes)
    print("res", res.shape)
    diag = res.diagonal(offset, dim1, dim2)
    diag.setflags(write=1)
    np.copyto(diag, x.data)
    return Tensor(res)
