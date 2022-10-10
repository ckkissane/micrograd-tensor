import pytest
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

def test_cross_entropy_backward():
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

def test_unbroadcast_add():
    A = np.ones((5, 4))
    B = np.ones(4,)
    out = A + B
    assert F.unbroadcast(out, A.shape).shape == A.shape
    assert F.unbroadcast(out, B.shape).shape == B.shape
        
