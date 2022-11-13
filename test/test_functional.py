import torch
import numpy as np
from micrograd.engine import Tensor
import micrograd.functional as F


def test_cross_entropy_forward():
    ma = Tensor(np.random.randn(5))
    mtarget = 2
    mout = F.cross_entropy(ma, mtarget)

    ta = torch.as_tensor(ma.data)
    ttarget = torch.tensor(mtarget)
    tout = torch.nn.functional.cross_entropy(ta, ttarget)
    assert np.allclose(mout.data, tout.numpy())


def test_cross_entropy_backward():
    ma = Tensor(np.random.randn(5))
    mtarget = 2
    mout = F.cross_entropy(ma, mtarget)
    mout.backward()

    ta = torch.as_tensor(ma.data)
    ta.requires_grad = True
    ttarget = torch.tensor(mtarget)
    tout = torch.nn.functional.cross_entropy(ta, ttarget)
    tout.backward(gradient=torch.ones_like(tout))
    assert np.allclose(ma.grad, ta.grad.numpy())


def test_batched_cross_entropy_forward():
    batch_size = 2
    ma = Tensor(np.random.randn(batch_size, 5))
    mtarget = np.ones(batch_size, dtype=int)
    mout = F.batched_cross_entropy(ma, mtarget)

    ta = torch.as_tensor(ma.data)
    ttarget = torch.as_tensor(mtarget)
    tout = torch.nn.functional.cross_entropy(ta, ttarget)
    assert np.allclose(mout.data, tout.numpy())


def test_batched_cross_entropy_backward():
    batch_size = 2
    ma = Tensor(np.random.randn(batch_size, 5))
    mtarget = np.ones(batch_size, dtype=int)
    mout = F.batched_cross_entropy(ma, mtarget)
    mout.backward()

    ta = torch.as_tensor(ma.data)
    ta.requires_grad = True
    ttarget = torch.as_tensor(mtarget)
    tout = torch.nn.functional.cross_entropy(ta, ttarget)
    tout.backward(gradient=torch.ones_like(tout))
    assert np.allclose(ma.grad, ta.grad.numpy())


# helper function to test conv
def conv_ref(Z, weight, stride=1, padding=0):
    # NHWC -> NCHW
    Z_torch = torch.as_tensor(Z).permute(0, 3, 1, 2)
    # (k, k, iC, oC) -> (oC, iC, k, k)
    weight_torch = torch.as_tensor(weight).permute(3, 2, 0, 1)
    out = torch.nn.functional.conv2d(
        Z_torch, weight_torch, stride=stride, padding=padding
    )
    return out.permute(0, 2, 3, 1).contiguous().numpy()


def test_conv2d_forward_no_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    out = F.conv2d(my_Z, my_weight)

    torch_out = conv_ref(Z, weight)
    assert np.allclose(out.data, torch_out)


def test_conv2d_forward_with_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)
    padding = 1

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    out = F.conv2d(my_Z, my_weight, padding=padding)

    torch_out = conv_ref(Z, weight, padding=padding)
    assert np.allclose(out.data, torch_out)


def test_conv2d_forward_with_stride_no_padding():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)
    stride = 2

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    out = F.conv2d(my_Z, my_weight, stride=stride)

    torch_out = conv_ref(Z, weight, stride=stride)
    assert np.allclose(out.data, torch_out)


def test_conv2d_forward_with_stride_and_padding():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)
    stride = 2
    padding = 1

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    out = F.conv2d(my_Z, my_weight, stride=stride, padding=padding)

    torch_out = conv_ref(Z, weight, stride=stride, padding=padding)
    assert np.allclose(out.data, torch_out)


# helper function to test conv backward
def conv_ref_backward(Z, weight, stride=1, padding=0):
    # NHWC -> NCHW
    Z_torch = torch.tensor(Z).permute(0, 3, 1, 2)
    Z_torch.requires_grad = True
    # (k, k, iC, oC) -> (oC, iC, k, k)
    weight_torch = torch.tensor(weight).permute(3, 2, 0, 1)
    weight_torch.requires_grad = True
    out = torch.nn.functional.conv2d(
        Z_torch, weight_torch, stride=stride, padding=padding
    )
    out.backward(gradient=torch.ones_like(out))
    return (
        weight_torch.grad.permute(2, 3, 1, 0).contiguous().numpy(),
        Z_torch.grad.permute(0, 2, 3, 1).contiguous().numpy(),
    )


def test_conv2d_backward_no_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    my_out = F.conv2d(my_Z, my_weight)
    my_out.backward()

    torch_dweight, torch_dZ = conv_ref_backward(Z, weight)
    assert np.allclose(my_weight.grad, torch_dweight)
    assert np.allclose(my_Z.grad, torch_dZ)


def test_conv2d_backward_with_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)
    padding = 1

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    my_out = F.conv2d(my_Z, my_weight, padding=padding)
    my_out.backward()

    torch_dweight, torch_dZ = conv_ref_backward(Z, weight, padding=padding)
    assert np.allclose(my_weight.grad, torch_dweight)
    assert np.allclose(my_Z.grad, torch_dZ)


def test_conv2d_backward_with_stride_no_padding():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)
    stride = 2

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    my_out = F.conv2d(my_Z, my_weight, stride=stride)
    my_out.backward()

    torch_dweight, torch_dZ = conv_ref_backward(Z, weight, stride=stride)
    assert np.allclose(my_weight.grad, torch_dweight)
    assert np.allclose(my_Z.grad, torch_dZ)


def test_conv2d_backward_with_stride_and_padding():
    Z = np.random.randn(100, 32, 32, 8)
    weight = np.random.randn(3, 3, 8, 16)
    stride = 2
    padding = 1

    my_Z = Tensor(Z)
    my_weight = Tensor(weight)

    my_out = F.conv2d(my_Z, my_weight, stride=stride, padding=padding)
    my_out.backward()

    torch_dweight, torch_dZ = conv_ref_backward(
        Z, weight, stride=stride, padding=padding
    )
    assert np.allclose(my_weight.grad, torch_dweight)
    assert np.allclose(my_Z.grad, torch_dZ)
