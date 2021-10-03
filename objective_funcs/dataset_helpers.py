from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision as tv


def get_dataloaders(data_dir: str, dataset_type: str, requires_validation: bool, device: torch.device) \
        -> Tuple[Dataset, DataLoader, DataLoader, DataLoader]:
    if dataset_type.lower() == "cifar10":
        dataset = CIFAR10
        train_key = {'train': True}
        test_key = {'train': False}
    elif dataset_type.lower() == "cifar100":
        dataset = CIFAR100
        train_key = {'train': True}
        test_key = {'train': False}
    elif dataset_type.lower() == "svhn":
        dataset = SVHN
        train_key = {'split': 'train'}
        test_key = {'split': 'test'}
    else:
        raise KeyError

    train = dataset(data_dir, device, download=True, **train_key)
    if requires_validation:
        val_size = int(0.1 * len(train))
        train_size = len(train) - val_size
        train, val = random_split(train, [train_size, val_size])
    test = dataset(data_dir, device, download=True, **test_key)

    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None if not requires_validation else DataLoader(val, batch_size=5000, shuffle=False,
                                                                 num_workers=0)
    train_eval_loader = DataLoader(train, batch_size=5000, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=5000, shuffle=False, num_workers=0)
    return train, train_eval_loader, val_loader, test_loader


def process_data(data_np: np.ndarray, targets_np: np.ndarray, device: torch.device):
    # Scale data to [0,1] floats
    data_np = data_np / 255

    # Normalize data
    data_np = (data_np - data_np.mean(axis=(0, 1, 2))) / data_np.std(axis=(0, 1, 2))

    # NHWC -> NCHW
    data_np = data_np.transpose((0, 3, 1, 2))

    # Numpy -> Torch
    data = torch.tensor(data_np, dtype=torch.float32)
    targets = torch.tensor(targets_np, dtype=torch.long)

    # Put both data and targets on GPU in advance
    return data.to(device), targets.to(device)


# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class CIFAR10(tv.datasets.CIFAR10):
    def __init__(self, data_dir: str, device: torch.device, *args, **kwargs):
        super().__init__(data_dir, *args, **kwargs)
        self.data, self.targets = process_data(self.data, np.array(self.targets), device)

    # Don't convert to PIL like torchvision default
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class CIFAR100(tv.datasets.CIFAR100):
    def __init__(self, data_dir: str, device: torch.device, *args, **kwargs):
        super().__init__(data_dir, *args, **kwargs)
        self.data, self.targets = process_data(self.data, np.array(self.targets), device)

    # Don't convert to PIL like torchvision default
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class SVHN(tv.datasets.SVHN):
    def __init__(self, data_dir: str, device: torch.device, *args, **kwargs):
        super().__init__(data_dir, *args, **kwargs)
        self.data = self.data.transpose((0, 2, 3, 1))  # NCHW -> NHWC (SVHN)
        self.data, self.labels = process_data(self.data, self.labels, device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
