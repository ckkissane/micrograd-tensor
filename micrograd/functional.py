import numpy as np
from .engine import Tensor
import micrograd


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


def batched_cross_entropy(input: Tensor, target: Tensor) -> Tensor:
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
    p = probs[np.arange(probs.shape[0]), target.data]
    out = Tensor(-np.log(p).mean(keepdims=True), (input,), "cross_entropy")

    def _backward():
        batch_size = probs.shape[0]
        input.grad = np.copy(probs)
        input.grad[np.arange(batch_size), target.data] -= 1.0
        input.grad /= batch_size

    out._backward = _backward

    return out


def conv2d(Z, weight, stride=1, padding=0):
    """
    Args:
        Z: Tensor(N, H, W, C_in) input image
        weight: Tensor(K, K, C_in, C_out) conv filters
        stride: int
        padding: int (padding to add to each size of image)
    """

    def conv2d_forward(Z, weight, stride=1, padding=0):
        N, H, W, C_in = Z.shape
        K, _, _, C_out = weight.shape

        padded_Z = np.pad(
            Z, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant"
        )
        Ns, Hs, Ws, Cs = padded_Z.strides
        inner_dim = K * K * C_in
        H_out = (H + 2 * padding - K) // stride + 1
        W_out = (W + 2 * padding - K) // stride + 1
        A = np.lib.stride_tricks.as_strided(
            padded_Z,
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs * stride, Ws * stride, Hs, Ws, Cs),
        ).reshape(-1, inner_dim)
        out = A @ weight.reshape(-1, C_out)
        return out.reshape(N, H_out, W_out, C_out)

    out_data = conv2d_forward(Z.data, weight.data, stride=stride, padding=padding)
    out = Tensor(out_data, (Z, weight), "conv2d")

    def _backward():
        def dilate(x, dilation):
            """
            Args:
                x: np.array(N, H, W, C_in)
                dilation: int
            Returns:
                out: np.array(N, H*(dilation+1), W*(dilation+1), C_in)
            """
            N, H, W, C_in = x.shape
            dilate_len = dilation + 1
            dilate_shape = (N, H * dilate_len, W * dilate_len, C_in)
            out = np.zeros(dilate_shape)
            # TODO: optimize this?
            for i in range(H):
                for j in range(W):
                    out[:, i * dilate_len, j * dilate_len, :] = x[:, i, j, :]
            return out

        dilated_out_grad = dilate(out.grad, stride - 1)
        _, doH, _, _ = dilated_out_grad.shape
        K, _, _, _ = weight.data.shape
        _, iH, _, _ = Z.data.shape

        flipped_weight = np.flip(np.flip(weight.data, 0), 1).transpose(0, 1, 3, 2)
        out_padding = (iH - doH + K - 1) // 2
        Z.grad += conv2d_forward(dilated_out_grad, flipped_weight, padding=out_padding)

        weight.grad += conv2d_forward(
            Z.data.transpose(3, 1, 2, 0),
            dilated_out_grad.transpose(1, 2, 0, 3),
            padding=padding,
        ).transpose(1, 2, 0, 3)

    out._backward = _backward

    return out


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # TODO: analytical derivative?
    if weight is None:
        weight = Tensor(np.ones(normalized_shape))
    if bias is None:
        bias = Tensor(np.zeros(normalized_shape))

    axes = tuple(-i for i in range(1, len(normalized_shape) + 1))
    eps = Tensor(np.array(eps))
    out = (input - input.mean(dim=axes, keepdims=True)) / micrograd.sqrt(
        input.var(dim=axes, unbiased=False, keepdims=True) + eps
    ) * weight + bias
    return out


def embedding(input: Tensor, weight: Tensor) -> Tensor:
    out = Tensor(weight.data[input.data], (weight,), "embedding")

    def _backward():
        reshaped_input = input.data.reshape(-1)
        reshaped_out_grad = out.grad.reshape(-1)
        # TODO: speed up
        for idx, g in zip(reshaped_input, reshaped_out_grad):
            weight.grad[idx] += g

    out._backward = _backward

    return out
