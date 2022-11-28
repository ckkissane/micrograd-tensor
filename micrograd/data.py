from typing import Optional
from .engine import Tensor
import numpy as np
import random
import math
import gzip
import os


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


class MnistDataset(Dataset):
    def __init__(self, train: bool = True):
        self.train = train

        self.x_train = (
            self._parse(
                os.path.dirname(__file__) + "/mnist/train-images-idx3-ubyte.gz"
            )[0x10:]
            .reshape((-1, 28, 28, 1))
            .astype(np.float32)
            / 256
        )
        self.y_train = self._parse(
            os.path.dirname(__file__) + "/mnist/train-labels-idx1-ubyte.gz"
        )[8:]

        self.x_test = (
            self._parse(os.path.dirname(__file__) + "/mnist/t10k-images-idx3-ubyte.gz")[
                0x10:
            ]
            .reshape((-1, 28, 28, 1))
            .astype(np.float32)
            / 256
        )
        self.y_test = self._parse(
            os.path.dirname(__file__) + "/mnist/t10k-labels-idx1-ubyte.gz"
        )[8:]

    def _parse(self, file):
        return np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()

    def __len__(self):
        return self.x_train.shape[0] if self.train else self.x_test.shape[0]

    def __getitem__(self, item):
        return (
            (Tensor(self.x_train[item]), Tensor(self.y_train[item]))
            if self.train
            else (Tensor(self.x_test[item]), Tensor(self.y_test[item]))
        )


class DataLoader:
    def __init__(
        self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        self.item = 0
        self.indices = [i for i in range(len(self.dataset))]
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.item < len(self.dataset):
            x_batch, y_batch = [], []
            for _ in range(min(self.batch_size, len(self.dataset) - self.item)):
                idx = self.indices[self.item]
                x_tensor, y_tensor = self.dataset[idx]
                x_batch.append(x_tensor.data)
                y_batch.append(y_tensor.data)
                self.item += 1
            x_numpy = np.stack(x_batch, axis=0)
            y_numpy = np.stack(y_batch, axis=0)
            return Tensor(x_numpy), Tensor(y_numpy)
        else:
            self.item = 0
            raise StopIteration
