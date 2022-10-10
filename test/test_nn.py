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
