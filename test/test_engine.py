from micrograd.engine import Tensor
import numpy as np
import torch


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


def test_sub_forward():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(5, 10, 32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a - my_b

    torch_a = torch.as_tensor(a)
    torch_b = torch.as_tensor(b)
    torch_out = torch_a - torch_b
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_sub_forward_broadcast():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a - my_b

    torch_a = torch.as_tensor(a)
    torch_b = torch.as_tensor(b)
    torch_out = torch_a - torch_b
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_sub_backward():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(5, 10, 32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a - my_b
    my_out.backward()

    torch_a = torch.as_tensor(a)
    torch_a.requires_grad = True
    torch_b = torch.as_tensor(b)
    torch_b.requires_grad = True
    torch_out = torch_a - torch_b
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_a.grad, torch_a.grad.numpy())
    assert np.allclose(my_b.grad, torch_b.grad.numpy())


def test_sub_backward_broadcast():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a - my_b
    my_out.backward()

    torch_a = torch.as_tensor(a)
    torch_a.requires_grad = True
    torch_b = torch.as_tensor(b)
    torch_b.requires_grad = True
    torch_out = torch_a - torch_b
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_a.grad, torch_a.grad.numpy())
    assert np.allclose(my_b.grad, torch_b.grad.numpy())


def test_mul_forward():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(5, 10, 32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a * my_b

    torch_a = torch.as_tensor(a)
    torch_b = torch.as_tensor(b)
    torch_out = torch_a * torch_b
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_mul_forward_broadcast():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a * my_b

    torch_a = torch.as_tensor(a)
    torch_b = torch.as_tensor(b)
    torch_out = torch_a * torch_b
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_mul_backward():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(5, 10, 32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a * my_b
    my_out.backward()

    torch_a = torch.as_tensor(a)
    torch_a.requires_grad = True
    torch_b = torch.as_tensor(b)
    torch_b.requires_grad = True
    torch_out = torch_a * torch_b
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_a.grad, torch_a.grad.numpy())
    assert np.allclose(my_b.grad, torch_b.grad.numpy())


def test_mul_backward_broadcast():
    a = np.random.randn(5, 10, 32)
    b = np.random.randn(32)

    my_a = Tensor(a)
    my_b = Tensor(b)
    my_out = my_a * my_b
    my_out.backward()

    torch_a = torch.as_tensor(a)
    torch_a.requires_grad = True
    torch_b = torch.as_tensor(b)
    torch_b.requires_grad = True
    torch_out = torch_a * torch_b
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_a.grad, torch_a.grad.numpy())
    assert np.allclose(my_b.grad, torch_b.grad.numpy())


def test_div_forward():
    my_a = Tensor(np.random.randn(5))
    my_b = Tensor(np.random.randn(5))
    my_c = my_a / my_b

    torch_a = torch.from_numpy(my_a.data)
    torch_b = torch.from_numpy(my_b.data)
    torch_c = torch_a / torch_b

    assert np.allclose(my_c.data, torch_c.numpy())


def test_div_backward():
    my_a = Tensor(np.random.randn(5))
    my_b = Tensor(np.random.randn(5))
    my_c = my_a / my_b
    my_c.backward()

    torch_a = torch.from_numpy(my_a.data)
    torch_a.requires_grad = True
    torch_b = torch.from_numpy(my_b.data)
    torch_b.requires_grad = True
    torch_c = torch_a / torch_b
    torch_c.backward(gradient=torch.ones_like(torch_c))

    assert np.allclose(my_a.grad, torch_a.grad.numpy())
    assert np.allclose(my_b.grad, torch_b.grad.numpy())


def test_broadcasted_div_forward():
    my_a = Tensor(np.random.randn(5, 5))
    my_b = Tensor(np.random.randn(5))
    my_c = my_a / my_b

    torch_a = torch.from_numpy(my_a.data)
    torch_b = torch.from_numpy(my_b.data)
    torch_c = torch_a / torch_b

    assert np.allclose(my_c.data, torch_c.numpy())


def test_broadcasted_div_backward():
    my_a = Tensor(np.random.randn(5, 5))
    my_b = Tensor(np.random.randn(5))
    my_c = my_a / my_b
    my_c.backward()

    torch_a = torch.from_numpy(my_a.data)
    torch_a.requires_grad = True
    torch_b = torch.from_numpy(my_b.data)
    torch_b.requires_grad = True
    torch_c = torch_a / torch_b
    torch_c.backward(gradient=torch.ones_like(torch_c))

    assert np.allclose(my_a.grad, torch_a.grad.numpy())
    assert np.allclose(my_b.grad, torch_b.grad.numpy())


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


def test_relu_forward():
    ma = Tensor(np.random.randn(5))
    mb = ma.relu()

    ta = torch.from_numpy(ma.data)
    tb = ta.relu()

    assert np.allclose(mb.data, tb.numpy())


def test_relu_backward():
    ma = Tensor(np.random.randn(5))
    mb = ma.relu()
    mb.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = ta.relu()
    tb.backward(gradient=torch.ones_like(tb))

    assert np.allclose(ma.grad, ta.grad.numpy())


def test_batched_relu_forward():
    batch_size = 16
    ma = Tensor(np.random.randn(batch_size, 5))
    mb = ma.relu()

    ta = torch.from_numpy(ma.data)
    tb = ta.relu()

    assert np.allclose(mb.data, tb.numpy())


def test_batched_relu_backward():
    batch_size = 16
    ma = Tensor(np.random.randn(batch_size, 5))
    mb = ma.relu()
    mb.backward()

    ta = torch.from_numpy(ma.data)
    ta.requires_grad = True
    tb = ta.relu()
    tb.backward(gradient=torch.ones_like(tb))

    assert np.allclose(ma.grad, ta.grad.numpy())


def test_reshape_forward():
    a = np.arange(6, dtype=np.float32)
    my_a = Tensor(a)
    new_shape = (3, 2)
    my_out = my_a.reshape(new_shape)

    torch_a = torch.from_numpy(a)
    torch_out = torch_a.reshape(new_shape)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_reshape_backward():
    a = np.arange(6, dtype=np.float32)
    my_a = Tensor(a)
    new_shape = (3, 2)
    my_out = my_a.reshape(new_shape)
    my_out.backward()

    torch_a = torch.from_numpy(a)
    torch_a.requires_grad = True
    torch_out = torch_a.reshape(new_shape)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_a.grad, torch_a.grad.numpy())


def test_tranpose_forward():
    k = np.random.randn(1, 2, 5, 4)

    my_k = Tensor(k)
    my_out = my_k.transpose(-1, -2)

    torch_k = torch.as_tensor(k)
    torch_out = torch_k.transpose(-1, -2)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_tranpose_backward():
    k = np.random.randn(1, 2, 5, 4)

    my_k = Tensor(k)
    my_out = my_k.transpose(-1, -2)
    my_out.backward()

    torch_k = torch.as_tensor(k)
    torch_k.requires_grad = True
    torch_out = torch_k.transpose(-1, -2)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_k.grad, torch_k.grad.numpy())


def test_softmax_forward():
    x = np.random.randn(1, 2, 5, 5)

    my_x = Tensor(x)
    my_out = my_x.softmax(dim=-1)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.softmax(dim=-1)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_softmax_backward_vector():
    x = np.random.randn(5)
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.softmax(dim=dim)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.softmax(dim=dim)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_softmax_backward_matrix():
    x = np.random.randn(5, 5)
    dim = 0

    my_x = Tensor(x)
    my_out = my_x.softmax(dim=dim)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.softmax(dim=dim)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_softmax_backward_tensor():
    x = np.random.randn(1, 2, 5, 5)
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.softmax(dim=dim)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.softmax(dim=dim)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_mean_forward():
    x = np.random.randn(1, 2, 5, 5)
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dim)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.mean(dim=dim)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_mean_forward_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dim, keepdims=True)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.mean(dim=dim, keepdims=True)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_mean_forward_multi_dim():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dims)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.mean(dim=dims)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_mean_forward_multi_dim_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dims, keepdims=True)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.mean(dim=dims, keepdims=True)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_mean_backward():
    x = np.random.randn(1, 2, 5, 5)
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dim)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.mean(dim=dim)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_mean_backward_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dim, keepdims=True)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.mean(dim=dim, keepdims=True)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_mean_backward_multi_dim():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dims)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.mean(dim=dims)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_mean_backward_multi_dim_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)

    my_x = Tensor(x)
    my_out = my_x.mean(dim=dims, keepdims=True)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.mean(dim=dims, keepdims=True)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_unbiased_forward():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = True
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dim, unbiased=unbiased)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_biased_forward():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = False
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dim, unbiased=unbiased)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_unbiased_forward_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = True
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased, keepdims=True)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dim, unbiased=unbiased, keepdims=True)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_biased_forward_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = False
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased, keepdims=True)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dim, unbiased=unbiased, keepdims=True)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_forward_unbiased_multi_dim():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = True

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, unbiased=unbiased)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dims, unbiased=unbiased)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_forward_biased_multi_dim():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = False

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, unbiased=unbiased)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dims, unbiased=unbiased)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_forward_unbiased_multi_dim_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = True

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, keepdims=True, unbiased=unbiased)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dims, keepdims=True, unbiased=unbiased)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_forward_biased_multi_dim_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = False

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, keepdims=True, unbiased=unbiased)

    torch_x = torch.as_tensor(x)
    torch_out = torch_x.var(dim=dims, keepdims=True, unbiased=unbiased)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_var_biased_backward():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = False
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dim, unbiased=unbiased)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_unbiased_backward():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = True
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dim, unbiased=unbiased)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_biased_backward_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = False
    keepdims = True
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased, keepdims=keepdims)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dim, unbiased=unbiased, keepdims=keepdims)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_unbiased_backward_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    unbiased = False
    keepdims = True
    dim = -1

    my_x = Tensor(x)
    my_out = my_x.var(dim=dim, unbiased=unbiased, keepdims=keepdims)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dim, unbiased=unbiased, keepdims=keepdims)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_backward_unbiased_multi_dim():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = True

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, unbiased=unbiased)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dims, unbiased=unbiased)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_backward_biased_multi_dim():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = False

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, unbiased=unbiased)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dims, unbiased=unbiased)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_backward_unbiased_multi_dim_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = True
    keepdims = True

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, unbiased=unbiased, keepdims=keepdims)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dims, unbiased=unbiased, keepdims=keepdims)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())


def test_var_backward_biased_multi_dim_keepdims():
    x = np.random.randn(1, 2, 5, 5)
    dims = (-1, -2, -3)
    unbiased = False
    keepdims = True

    my_x = Tensor(x)
    my_out = my_x.var(dim=dims, unbiased=unbiased, keepdims=keepdims)
    my_out.backward()

    torch_x = torch.as_tensor(x)
    torch_x.requires_grad = True
    torch_out = torch_x.var(dim=dims, unbiased=unbiased, keepdims=keepdims)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_x.grad, torch_x.grad.numpy())
