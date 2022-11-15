import torch
import micrograd
from micrograd.engine import Tensor
import numpy as np


def test_where_forward():
    batch_size = 1
    num_heads = 2
    seq_len = 5
    mask_val = -1e4
    q_ind = np.arange(seq_len)[..., None]
    k_ind = np.arange(seq_len)[None, ...]
    scores = np.random.randn(batch_size, num_heads, seq_len, seq_len)

    my_condition = Tensor(q_ind < k_ind)
    my_x = Tensor(mask_val * np.ones_like(scores))
    my_y = Tensor(scores)
    my_out = micrograd.where(my_condition, my_x, my_y)

    torch_condition = torch.as_tensor(q_ind) < torch.as_tensor(k_ind)
    torch_y = torch.as_tensor(scores)
    torch_x = torch.as_tensor(my_x.data)
    torch_out = torch.where(torch_condition, torch_x, torch_y)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_where_backward():
    batch_size = 1
    num_heads = 2
    seq_len = 5
    mask_val = -1e4
    q_ind = np.arange(seq_len)[..., None]
    k_ind = np.arange(seq_len)[None, ...]
    scores = np.random.randn(batch_size, num_heads, seq_len, seq_len)

    my_condition = Tensor(q_ind < k_ind)
    my_x = Tensor(mask_val * np.ones_like(scores))
    my_y = Tensor(scores)
    my_out = micrograd.where(my_condition, my_x, my_y)
    my_out.backward()

    torch_condition = torch.as_tensor(q_ind) < torch.as_tensor(k_ind)
    torch_y = torch.as_tensor(scores)
    torch_y.requires_grad = True
    torch_x = torch.as_tensor(my_x.data)
    torch_x.requires_grad = True
    torch_out = torch.where(torch_condition, torch_x, torch_y)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_y.grad, torch_y.grad.numpy())
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_where_forward_broadcasted():
    batch_size = 1
    num_heads = 2
    seq_len = 5
    mask_val = -1e4
    q_ind = np.arange(seq_len)[..., None]
    k_ind = np.arange(seq_len)[None, ...]
    scores = np.random.randn(batch_size, num_heads, seq_len, seq_len)

    my_condition = Tensor(q_ind < k_ind)
    my_x = Tensor(np.array(mask_val))
    my_y = Tensor(scores)
    my_out = micrograd.where(my_condition, my_x, my_y)

    torch_condition = torch.as_tensor(q_ind) < torch.as_tensor(k_ind)
    torch_y = torch.as_tensor(scores)
    torch_x = torch.tensor(mask_val)
    torch_out = torch.where(torch_condition, torch_x, torch_y)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_where_backward_broadcasted():
    batch_size = 1
    num_heads = 2
    seq_len = 5
    mask_val = -1e4
    q_ind = np.arange(seq_len)[..., None]
    k_ind = np.arange(seq_len)[None, ...]
    scores = np.random.randn(batch_size, num_heads, seq_len, seq_len)

    my_condition = Tensor(q_ind < k_ind)
    my_x = Tensor(np.array(mask_val))
    my_y = Tensor(scores)
    my_out = micrograd.where(my_condition, my_x, my_y)
    my_out.backward()

    torch_condition = torch.as_tensor(q_ind) < torch.as_tensor(k_ind)
    torch_y = torch.as_tensor(scores)
    torch_y.requires_grad = True
    torch_x = torch.tensor(mask_val)
    torch_x.requires_grad = True
    torch_out = torch.where(torch_condition, torch_x, torch_y)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_y.grad, torch_y.grad.numpy())
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_diag_embed_matrix():
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    my_x = Tensor(x)
    my_out = micrograd.diag_embed(my_x)

    torch_x = torch.as_tensor(x)
    torch_out = torch.diag_embed(torch_x)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_diag_embed_tensor():
    x = np.random.randn(1, 2, 5, 5)
    my_x = Tensor(x)
    my_out = micrograd.diag_embed(my_x)

    torch_x = torch.as_tensor(x)
    torch_out = torch.diag_embed(torch_x)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_sqrt_forward():
    x = np.arange(1, 4)

    my_x = Tensor(x)
    my_out = micrograd.sqrt(my_x)

    torch_x = torch.as_tensor(x)
    torch_out = torch.sqrt(torch_x)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_sqrt_backward():
    x = np.arange(1, 4, dtype=np.float32)

    my_x = Tensor(x)
    my_out = micrograd.sqrt(my_x)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch.sqrt(torch_x)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())
