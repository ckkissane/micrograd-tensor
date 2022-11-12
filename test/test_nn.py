from micrograd.engine import Tensor
import micrograd.nn as nn
import torch
import numpy as np


def test_sigmoid_forward():
    msig = nn.Sigmoid()
    ma = Tensor(np.random.randn(5))
    mb = msig(ma)

    tsig = torch.nn.Sigmoid()
    ta = torch.from_numpy(ma.data)
    tb = tsig(ta)
    assert np.allclose(mb.data, tb.numpy())


def test_sigmoid_backward():
    msig = nn.Sigmoid()
    ma = Tensor(np.random.randn(5))
    mb = msig(ma)
    mb.backward()

    tsig = torch.nn.Sigmoid()
    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = tsig(ta)
    tb.backward(gradient=torch.ones_like(tb))
    assert np.allclose(ma.grad, ta.grad.numpy())


def test_relu_forward():
    mrelu = nn.ReLU()
    ma = Tensor(np.random.randn(5))
    mb = mrelu(ma)

    trelu = torch.nn.ReLU()
    ta = torch.from_numpy(ma.data)
    tb = trelu(ta)
    assert np.allclose(mb.data, tb.numpy())


def test_relu_backward():
    mrelu = nn.ReLU()
    ma = Tensor(np.random.randn(5))
    mb = mrelu(ma)
    mb.backward()

    trelu = torch.nn.ReLU()
    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = trelu(ta)
    tb.backward(gradient=torch.ones_like(tb))
    assert np.allclose(ma.grad, ta.grad.numpy())


def test_linear_forward():
    mlin = nn.Linear(2, 3)
    ma = Tensor(np.random.randn(2))
    mb = mlin(ma)

    tlin = torch.nn.Linear(2, 3)
    tlin.bias = torch.nn.Parameter(torch.from_numpy(mlin.bias.data))
    tlin.weight = torch.nn.Parameter(torch.from_numpy(mlin.weight.data.T))
    ta = torch.from_numpy(ma.data)
    tb = tlin(ta)
    assert np.allclose(mb.data, tb.detach().numpy())


def test_linear_backward():
    mlin = nn.Linear(2, 3)
    ma = Tensor(np.random.randn(2))
    mb = mlin(ma)
    mb.backward()

    tlin = torch.nn.Linear(2, 3)
    tlin.bias = torch.nn.Parameter(torch.from_numpy(mlin.bias.data))
    tlin.weight = torch.nn.Parameter(torch.from_numpy(mlin.weight.data.T))
    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = tlin(ta)
    tb.backward(gradient=torch.ones_like(tb))
    assert np.allclose(ma.grad, ta.grad.numpy())


def test_batched_linear_forward():
    batch_size = 16
    mlin = nn.Linear(2, 3)
    ma = Tensor(np.random.randn(batch_size, 2))
    mb = mlin(ma)

    tlin = torch.nn.Linear(2, 3)
    tlin.bias = torch.nn.Parameter(torch.from_numpy(mlin.bias.data))
    tlin.weight = torch.nn.Parameter(torch.from_numpy(mlin.weight.data.T))
    ta = torch.from_numpy(ma.data)
    tb = tlin(ta)
    assert np.allclose(mb.data, tb.detach().numpy())


def test_batched_linear_backward():
    batch_size = 16
    mlin = nn.Linear(2, 3)
    ma = Tensor(np.random.randn(batch_size, 2))
    mb = mlin(ma)
    mb.backward()

    tlin = torch.nn.Linear(2, 3)
    tlin.bias = torch.nn.Parameter(torch.from_numpy(mlin.bias.data))
    tlin.weight = torch.nn.Parameter(torch.from_numpy(mlin.weight.data.T))
    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = tlin(ta)
    tb.backward(gradient=torch.ones_like(tb))
    assert np.allclose(ma.grad, ta.grad.numpy())


def test_conv_forward_no_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    my_conv = nn.Conv2d(C_in, C_out, K)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)

    torch_conv = torch.nn.Conv2d(C_in, C_out, K, bias=False)
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_out = torch_conv(torch_Z).permute(0, 2, 3, 1)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_conv_forward_with_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    padding = 1
    my_conv = nn.Conv2d(C_in, C_out, K, padding=padding)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)

    torch_conv = torch.nn.Conv2d(C_in, C_out, K, padding=padding, bias=False)
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_out = torch_conv(torch_Z).permute(0, 2, 3, 1)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_conv_forward_with_stride_no_padding():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    stride = 2
    my_conv = nn.Conv2d(C_in, C_out, K, stride=stride)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)

    torch_conv = torch.nn.Conv2d(C_in, C_out, K, stride=stride, bias=False)
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_out = torch_conv(torch_Z).permute(0, 2, 3, 1)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_conv_forward_with_stride_and_padding():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    stride = 2
    padding = 1
    my_conv = nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)

    torch_conv = torch.nn.Conv2d(
        C_in, C_out, K, stride=stride, padding=padding, bias=False
    )
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_out = torch_conv(torch_Z).permute(0, 2, 3, 1)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_conv_backward_no_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    my_conv = nn.Conv2d(C_in, C_out, K)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)
    my_out.backward()

    torch_conv = torch.nn.Conv2d(C_in, C_out, K, bias=False)
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_Z.requires_grad = True
    torch_out = torch_conv(torch_Z)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_Z.grad, torch_Z.grad.permute(0, 2, 3, 1).numpy())


def test_conv_backward_with_padding_no_stride():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    padding = 1
    my_conv = nn.Conv2d(C_in, C_out, K, padding=padding)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)
    my_out.backward()

    torch_conv = torch.nn.Conv2d(C_in, C_out, K, padding=padding, bias=False)
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_Z.requires_grad = True
    torch_out = torch_conv(torch_Z)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_Z.grad, torch_Z.grad.permute(0, 2, 3, 1).numpy())


def test_conv_backward_with_stride_no_padding():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    stride = 2
    my_conv = nn.Conv2d(C_in, C_out, K, stride=stride)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)
    my_out.backward()

    torch_conv = torch.nn.Conv2d(C_in, C_out, K, stride=stride, bias=False)
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_Z.requires_grad = True
    torch_out = torch_conv(torch_Z)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_Z.grad, torch_Z.grad.permute(0, 2, 3, 1).numpy())


def test_conv_backward_with_stride_and_padding():
    Z = np.random.randn(100, 32, 32, 8)
    K, C_in, C_out = 3, 8, 16
    stride = 2
    padding = 1
    my_conv = nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding)

    my_Z = Tensor(Z)
    my_out = my_conv(my_Z)
    my_out.backward()

    torch_conv = torch.nn.Conv2d(
        C_in, C_out, K, stride=stride, padding=padding, bias=False
    )
    torch_conv.weight = torch.nn.Parameter(
        torch.from_numpy(my_conv.weight.data).permute(3, 2, 0, 1)
    )
    torch_Z = torch.from_numpy(Z).permute(0, 3, 1, 2)
    torch_Z.requires_grad = True
    torch_out = torch_conv(torch_Z)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_Z.grad, torch_Z.grad.permute(0, 2, 3, 1).numpy())
