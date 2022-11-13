from micrograd.engine import Tensor
import micrograd.optim as optim
import micrograd.nn as nn
import numpy as np
import torch


def test_vanilla_sgd():
    inp = np.random.randn(10)
    lr = 0.1
    num_iters = 5

    my_lin = nn.Linear(10, 20)
    my_sgd = optim.SGD(my_lin.parameters(), lr=lr)
    my_inp = Tensor(inp)

    torch_lin = torch.nn.Linear(10, 20)
    torch_lin.bias = torch.nn.Parameter(torch.from_numpy(my_lin.bias.data))
    torch_lin.weight = torch.nn.Parameter(torch.from_numpy(my_lin.weight.data.T))
    torch_sgd = torch.optim.SGD(torch_lin.parameters(), lr=lr)
    torch_inp = torch.as_tensor(inp)
    torch_inp.requires_grad = True

    for _ in range(num_iters):
        my_out = my_lin(my_inp)
        my_out.backward()
        my_sgd.step()

    for _ in range(num_iters):
        torch_out = torch_lin(torch_inp)
        torch_out.backward(gradient=torch.ones_like(torch_out))
        torch_sgd.step()

    assert np.allclose(my_lin.bias.data, torch_lin.bias.data.detach().numpy())
    assert np.allclose(my_lin.weight.data, torch_lin.weight.data.T.detach().numpy())
