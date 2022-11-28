from micrograd.engine import Tensor
from micrograd.data import Dataset, DataLoader, MnistDataset
import numpy as np
import torch

# toy dataset for testing
class MyReverseDataset(Dataset):
    def __init__(self, ndigit, size):
        self.ndigit = ndigit
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = np.arange(idx, idx + self.ndigit)
        y = np.flip(x, (-1,))
        return Tensor(x), Tensor(y)


class TorchReverseDataset(torch.utils.data.Dataset):
    def __init__(self, ndigit, size):
        self.ndigit = ndigit
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.arange(idx, idx + self.ndigit)
        y = torch.flip(x, (-1,))
        return x, y


def test_data_loader_types_and_shapes():
    size = 100
    batch_size = 4  # must divide size for this test
    ndigit = 2
    my_dataset = MyReverseDataset(ndigit, size)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    for x, y in my_dataloader:
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        assert x.shape == (
            batch_size,
            ndigit,
        )
        assert y.shape == (
            batch_size,
            ndigit,
        )


def test_data_loader_doesnt_drop_last():
    size = 100
    batch_size = 6  # must not divide size for this test
    ndigit = 2
    my_dataset = MyReverseDataset(ndigit, size)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    last_x, last_y = None, None
    for x, y in my_dataloader:
        last_x, last_y = x, y

    assert last_x.shape[0] == size % batch_size
    assert last_y.shape[0] == size % batch_size


def test_data_loader_no_shuffle():
    batch_size = 4
    size = 100
    ndigit = 2
    my_dataset = MyReverseDataset(ndigit, size)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    torch_dataset = TorchReverseDataset(ndigit, size)
    torch_dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size)

    for ((my_x, my_y), (torch_x, torch_y)) in zip(my_dataloader, torch_dataloader):
        assert np.allclose(my_x.data, torch_x.detach().numpy())
        assert np.allclose(my_y.data, torch_y.detach().numpy())


def test_data_loader_shuffle_types_and_shapes():
    batch_size = 4
    size = 100
    ndigit = 2
    my_dataset = MyReverseDataset(ndigit, size)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    for x, y in my_dataloader:
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        assert x.shape == (
            batch_size,
            ndigit,
        )
        assert y.shape == (
            batch_size,
            ndigit,
        )


def test_data_loader_shuffle():
    # batch size needs to be one for this test
    batch_size = 1
    ndigit = 2
    size = 1
    my_dataset = MyReverseDataset(ndigit, size)

    my_dataloader_unshuffled = DataLoader(
        my_dataset, batch_size=batch_size, shuffle=False
    )
    unshuffled_list = []
    unshuffled_set = set()

    my_dataloader_shuffled = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    shuffled_list = []
    shuffled_set = set()

    for ((x1, y1), (x2, y2)) in zip(my_dataloader_unshuffled, my_dataloader_shuffled):
        unshuffled_list.append((x1, y1))
        shuffled_list.append((x2, y2))
        # hack since np.array isn't hashable
        for r1, r2 in zip(x1.data, y1.data):
            unshuffled_set.add(tuple(r1))
            unshuffled_set.add(tuple(r2))
        for r1, r2 in zip(x2.data, y2.data):
            shuffled_set.add(tuple(r1))
            shuffled_set.add(tuple(r2))

    assert unshuffled_list != shuffled_list
    assert shuffled_set == unshuffled_set


def test_mnist_train_len():
    train_dataset = MnistDataset(train=True)
    assert len(train_dataset) == 60000


def test_mnist_test_len():
    test_dataset = MnistDataset(train=False)
    assert len(test_dataset) == 10000


def test_mnist_train_shape():
    train_dataset = MnistDataset(train=True)
    for x, y in train_dataset:
        assert x.shape == (
            1,
            28,
            28,
        )
        assert y.shape == ()


def test_mnist_test_shape():
    test_dataset = MnistDataset(train=False)
    for x, y in test_dataset:
        assert x.shape == (
            1,
            28,
            28,
        )
        assert y.shape == ()


def test_mnist_train_type():
    train_dataset = MnistDataset(train=True)
    for x, y in train_dataset:
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)


def test_mnist_test_type():
    test_dataset = MnistDataset(train=False)
    for x, y in test_dataset:
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)


def test_mnist_dataloader():
    train_dataset = MnistDataset(train=True)
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=2)
    for x, y in train_loader:
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        assert x.shape == (
            batch_size,
            1,
            28,
            28,
        )
        assert y.shape == (batch_size,)
