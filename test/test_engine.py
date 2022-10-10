from micrograd.engine import Tensor
import numpy as np
import torch
import pytest

def test_add_forward():
    ma = Tensor(np.random.randn(5))
    mb = Tensor(np.random.randn(5))
    mc = ma + mb

    ta = torch.from_numpy(ma.data)
    tb = torch.from_numpy(mb.data)
    tc = ta + tb

    assert np.allclose(mc.data, tc.numpy())
    
def test_add_backward():
    ma = Tensor(np.random.randn(5))
    mb = Tensor(np.random.randn(5))
    mc = ma + mb
    mc.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = torch.from_numpy(mb.data)
    tb.requires_grad = True
    tc = ta + tb
    tc.backward(gradient=torch.ones_like(tc))

    assert np.allclose(ma.grad, ta.grad.numpy())
    assert np.allclose(mb.grad, tb.grad.numpy())

def test_broadcasted_add_forward():
    ma = Tensor(np.random.randn(5, 5))
    mb = Tensor(np.random.randn(5))
    mc = ma + mb

    ta = torch.from_numpy(ma.data)
    tb = torch.from_numpy(mb.data)
    tc = ta + tb

    assert np.allclose(mc.data, tc.numpy())

def test_broadcasted_add_backward():
    ma = Tensor(np.random.randn(5, 5))
    mb = Tensor(np.random.randn(5))
    mc = ma + mb
    mc.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = torch.from_numpy(mb.data)
    tb.requires_grad = True
    tc = ta + tb
    tc.backward(gradient=torch.ones_like(tc))

    assert np.allclose(ma.grad, ta.grad.numpy())
    assert np.allclose(mb.grad, tb.grad.numpy())

def test_sigmoid_forward():
    ma = Tensor(np.random.randn(2, 2))
    mb = ma.sigmoid()

    ta = torch.from_numpy(ma.data)
    tb = ta.sigmoid()

    assert np.allclose(mb.data, tb.numpy())

def test_sigmoid_backward():
    ma = Tensor(np.random.randn(2, 2))
    mb = ma.sigmoid()
    mb.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = ta.sigmoid()
    tb.backward(gradient=torch.ones_like(tb))

    assert np.allclose(ma.grad, ta.grad.numpy())

def test_matmul_foward():
    ma = Tensor(np.random.randn(2))
    mb = Tensor(np.random.randn(2, 2))
    mc = ma.matmul(mb)

    ta = torch.from_numpy(ma.data)
    tb = torch.from_numpy(mb.data)
    tc = ta @ tb

    assert np.allclose(mc.data, tc.numpy())

def test_matmul_backward():
    ma = Tensor(np.random.randn(2))
    mb = Tensor(np.random.randn(2, 2))
    mc = ma.matmul(mb)
    mc.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = torch.from_numpy(mb.data)
    tb.requires_grad = True
    tc = ta @ tb
    tc.backward(gradient=torch.ones_like(tc))

    assert np.allclose(ma.grad, ta.grad.numpy())
    assert np.allclose(mb.grad, tb.grad.numpy())

def test_batched_matmul_foward():
    batch_size = 16
    ma = Tensor(np.random.randn(batch_size, 2))
    mb = Tensor(np.random.randn(2, 2))
    mc = ma.matmul(mb)

    ta = torch.from_numpy(ma.data)
    tb = torch.from_numpy(mb.data)
    tc = ta @ tb

    assert np.allclose(mc.data, tc.numpy())

def test_batched_matmul_backward():
    batch_size = 16
    ma = Tensor(np.random.randn(batch_size, 2))
    mb = Tensor(np.random.randn(2, 2))
    mc = ma.matmul(mb)
    mc.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = torch.from_numpy(mb.data)
    tb.requires_grad = True
    tc = ta @ tb
    tc.backward(gradient=torch.ones_like(tc))

    assert np.allclose(ma.grad, ta.grad.numpy())
    assert np.allclose(mb.grad, tb.grad.numpy())
