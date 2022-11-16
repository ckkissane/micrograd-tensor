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
    assert np.allclose(mlin.weight.grad, tlin.weight.grad.T.detach().numpy())
    assert np.allclose(mlin.bias.grad, tlin.bias.grad.detach().numpy())


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
    assert np.allclose(mlin.weight.grad, tlin.weight.grad.T.detach().numpy())
    assert np.allclose(mlin.bias.grad, tlin.bias.grad.detach().numpy())


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
    assert np.allclose(
        my_conv.weight.grad, torch_conv.weight.grad.permute(2, 3, 1, 0).numpy()
    )


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
    assert np.allclose(
        my_conv.weight.grad, torch_conv.weight.grad.permute(2, 3, 1, 0).numpy()
    )


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
    assert np.allclose(
        my_conv.weight.grad, torch_conv.weight.grad.permute(2, 3, 1, 0).numpy()
    )


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
    assert np.allclose(
        my_conv.weight.grad, torch_conv.weight.grad.permute(2, 3, 1, 0).numpy()
    )


def test_module_parameters():
    class CNN(nn.Module):
        def __init__(self):
            self.conv1 = nn.Conv2d(1, 6, K=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(6, 16, K=5, stride=2)

            self.lin1 = nn.Linear(400, 120)
            self.lin2 = nn.Linear(120, 80)
            self.lin3 = nn.Linear(80, 10)

        def manual_params(self):
            return [
                self.conv1.weight,
                self.conv2.weight,
                self.lin1.weight,
                self.lin1.bias,
                self.lin2.weight,
                self.lin2.bias,
                self.lin3.weight,
                self.lin3.bias,
            ]

    model = CNN()
    assert set(model.manual_params()) == set(model.parameters())


def test_module_parameters_with_list():
    class MLP(nn.Module):
        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = []
            for i in range(len(nouts)):
                self.layers.append(nn.Linear(sz[i], sz[i + 1]))
                if i != len(nouts) - 1:
                    self.layers.append(nn.ReLU())

        def manual_params(self):
            return [p for layer in self.layers for p in layer.parameters()]

    model = MLP(10, [20, 10])
    assert set(model.manual_params()) == set(model.parameters())


def test_layer_norm_forward():
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = np.random.randn(batch, sentence_length, embedding_dim)
    elementwise_affine = False

    my_embedding = Tensor(embedding)
    my_layernorm = nn.LayerNorm((embedding_dim,), elementwise_affine=elementwise_affine)
    my_out = my_layernorm(my_embedding)

    torch_embedding = torch.as_tensor(embedding)
    torch_layernorm = torch.nn.LayerNorm(
        (embedding_dim,), elementwise_affine=elementwise_affine
    )
    torch_out = torch_layernorm(torch_embedding)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_layer_norm_forward_elementwise_affine():
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = np.random.randn(batch, sentence_length, embedding_dim)
    elementwise_affine = True

    my_embedding = Tensor(embedding)
    my_layernorm = nn.LayerNorm((embedding_dim,), elementwise_affine=elementwise_affine)
    my_out = my_layernorm(my_embedding)

    torch_embedding = torch.as_tensor(embedding)
    torch_layernorm = torch.nn.LayerNorm(
        (embedding_dim,), elementwise_affine=elementwise_affine
    )
    torch_layernorm.weight = torch.nn.Parameter(
        torch.as_tensor(my_layernorm.weight.data)
    )
    torch_layernorm.bias = torch.nn.Parameter(torch.as_tensor(my_layernorm.bias.data))
    torch_out = torch_layernorm(torch_embedding)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_layer_norm_forward_image():
    N, C, H, W = 20, 5, 10, 10
    input = np.random.randn(N, C, H, W)
    normalized_shape = [C, H, W]
    elementwise_affine = True

    my_input = Tensor(input)
    my_layernorm = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)
    my_out = my_layernorm(my_input)

    torch_input = torch.as_tensor(input)
    torch_layernorm = torch.nn.LayerNorm(
        normalized_shape, elementwise_affine=elementwise_affine
    )
    torch_layernorm.weight = torch.nn.Parameter(
        torch.as_tensor(my_layernorm.weight.data)
    )
    torch_layernorm.bias = torch.nn.Parameter(torch.as_tensor(my_layernorm.bias.data))
    torch_out = torch_layernorm(torch_input)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_layer_norm_backward_elementwise_affine():
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = np.random.randn(batch, sentence_length, embedding_dim)
    elementwise_affine = True

    my_embedding = Tensor(embedding)
    my_layernorm = nn.LayerNorm((embedding_dim,), elementwise_affine=elementwise_affine)
    my_out = my_layernorm(my_embedding)
    my_out.backward()

    torch_embedding = torch.as_tensor(embedding)
    torch_embedding.requires_grad = True
    torch_layernorm = torch.nn.LayerNorm(
        (embedding_dim,), elementwise_affine=elementwise_affine
    )
    torch_layernorm.weight = torch.nn.Parameter(
        torch.as_tensor(my_layernorm.weight.data)
    )
    torch_layernorm.weight.requires_grad = True
    torch_layernorm.bias = torch.nn.Parameter(torch.as_tensor(my_layernorm.bias.data))
    torch_layernorm.bias.requires_grad = True
    torch_out = torch_layernorm(torch_embedding)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_embedding.grad, torch_embedding.grad.numpy())
    assert np.allclose(my_layernorm.weight.grad, torch_layernorm.weight.grad.numpy())
    assert np.allclose(my_layernorm.bias.grad, torch_layernorm.bias.grad.numpy())


def test_layer_norm_backward_image():
    N, C, H, W = 20, 5, 10, 10
    input = np.random.randn(N, C, H, W)
    normalized_shape = [C, H, W]
    elementwise_affine = True

    my_input = Tensor(input)
    my_layernorm = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)
    my_out = my_layernorm(my_input)
    my_out.backward()

    torch_input = torch.as_tensor(input)
    torch_input.requires_grad = True
    torch_layernorm = torch.nn.LayerNorm(
        normalized_shape, elementwise_affine=elementwise_affine
    )
    torch_layernorm.weight = torch.nn.Parameter(
        torch.as_tensor(my_layernorm.weight.data)
    )
    torch_layernorm.weight.requires_grad = True
    torch_layernorm.bias = torch.nn.Parameter(torch.as_tensor(my_layernorm.bias.data))
    torch_layernorm.bias.requires_grad = True
    torch_out = torch_layernorm(torch_input)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_input.grad, torch_input.grad.numpy())
    assert np.allclose(my_layernorm.weight.grad, torch_layernorm.weight.grad.numpy())
    assert np.allclose(my_layernorm.bias.grad, torch_layernorm.bias.grad.numpy())


def test_embedding_forward():
    input = np.array([[1, 2, 4, 5], [4, 3, 2, 9]])
    num_embeddings, embedding_dim = 10, 3

    my_input = Tensor(input)
    my_embedding = nn.Embedding(num_embeddings, embedding_dim)
    my_out = my_embedding(my_input)

    torch_input = torch.as_tensor(input)
    torch_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    torch_embedding.weight = torch.nn.Parameter(
        torch.as_tensor(my_embedding.weight.data)
    )
    torch_out = torch_embedding(torch_input)
    assert np.allclose(my_out.data, torch_out.detach().numpy())


def test_embedding_backward():
    input = np.array([[1, 2, 4, 5], [4, 3, 2, 9]])
    num_embeddings, embedding_dim = 10, 3

    my_input = Tensor(input)
    my_embedding = nn.Embedding(num_embeddings, embedding_dim)
    my_out = my_embedding(my_input)
    my_out.backward()

    torch_input = torch.as_tensor(input)
    torch_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    torch_embedding.weight = torch.nn.Parameter(
        torch.as_tensor(my_embedding.weight.data)
    )
    torch_out = torch_embedding(torch_input)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    assert np.allclose(my_embedding.weight.grad, torch_embedding.weight.grad.numpy())
